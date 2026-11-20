#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Sequential Bayesian Optimization via Adaptive Connecting Samples (SnAKe) and
:math:`\epsilon`-SnAKe for Bayesian optimization under input transition costs
[folch2022snake]_.

SnAKe plans lookahead query trajectories by:
1. Generating a batch of :math:`B` prospective optima via Thompson sampling
   (maximizing continuous Matheron posterior sample paths, with a discrete
   posterior sample fallback when pathwise sampling is unsupported).
2. Applying :math:`\epsilon`-point deletion in normalized :math:`[0, 1]^d` space
   to prune candidates that collapse onto already-evaluated inputs or batch
   duplicates, replacing them with space-filling exploratory samples.
3. Ordering the resulting query cloud from the current operating point via an
   open Traveling Salesman Problem (Open-TSP) solver to minimize cumulative
   input movement costs.

References

.. [folch2022snake]
    J. P. Folch, S. Zhang, R. M. Lee, B. Shafei, D. Walz, C. Tsay,
    M. van der Wilk, and R. Misener. SnAKe: Bayesian Optimization with
    Pathwise Exploration. Advances in Neural Information Processing
    Systems (NeurIPS), 35:35226-35239, 2022.

Contributor: M-Colley
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model, ModelList
from botorch.models.transforms.input import ChainedInputTransform, Normalize
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.sampling.pathwise.paths import PathList, SamplePath
from botorch.sampling.pathwise.utils import get_train_inputs, get_train_targets
from botorch.utils.sampling import draw_sobol_samples, manual_seed, sample_simplex
from botorch.utils.transforms import is_ensemble, normalize, unnormalize
from gpytorch.kernels import Kernel
from torch import Size, Tensor

__all__ = [
    "SnAKe",
    "apply_epsilon_point_deletion",
    "sample_thompson_optima",
    "solve_open_tsp_path",
]

_VALID_EPSILON_MODES = {"adaptive", "bo_rescaling"}


def _validate_bounds(bounds: Tensor) -> None:
    r"""Validate that ``bounds`` is a finite ``2 x d`` box with ``ub > lb``.

    Args:
        bounds: A ``2 x d``-dim tensor of lower and upper domain bounds.
    """
    if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] < 1:
        raise ValueError(
            f"`bounds` must be a `2 x d` tensor with `d >= 1`, got {bounds.shape}."
        )
    if not torch.isfinite(bounds).all():
        raise ValueError("`bounds` must contain only finite values.")
    if not torch.all(bounds[1] > bounds[0]):
        raise ValueError(
            "Upper bounds `bounds[1]` must be strictly greater than `bounds[0]`."
        )


def _validate_epsilon(epsilon: float | str | None) -> None:
    r"""Validate the ``epsilon`` point-deletion configuration.

    Args:
        epsilon: Either ``None``, a non-negative float, or one of
            ``{"adaptive", "bo_rescaling"}``.
    """
    if epsilon is None:
        return
    if isinstance(epsilon, bool):
        raise ValueError("`epsilon` cannot be a boolean.")
    if isinstance(epsilon, (int, float)):
        if epsilon < 0.0:
            raise ValueError(f"`epsilon` must be non-negative, got {epsilon}.")
        return
    if isinstance(epsilon, str):
        if epsilon.lower() not in _VALID_EPSILON_MODES:
            raise ValueError(
                f"Unsupported `epsilon` mode {epsilon!r}. Expected a non-negative "
                f"float, None, or one of {sorted(_VALID_EPSILON_MODES)}."
            )
        return
    raise ValueError(f"Invalid `epsilon` type: {type(epsilon).__name__}.")


def _validate_transition_cost_args(
    d: int,
    transition_weights: Tensor | None,
    transition_metric: Tensor | None,
) -> None:
    r"""Validate optional per-parameter weights or PSD metric matrix.

    Args:
        d: Input feature dimension.
        transition_weights: Optional ``(d,)`` non-negative coordinate weights.
        transition_metric: Optional ``(d, d)`` symmetric PSD matrix ``W``.
    """
    if transition_weights is not None and transition_metric is not None:
        raise ValueError(
            "Cannot specify both `transition_weights` and `transition_metric`."
        )
    if transition_weights is not None:
        if transition_weights.ndim != 1 or transition_weights.shape[0] != d:
            raise ValueError(
                f"`transition_weights` must have shape `({d},)`, "
                f"got {transition_weights.shape}."
            )
        if (transition_weights < 0).any() or not (transition_weights > 0).any():
            raise ValueError(
                "`transition_weights` must be non-negative with at least one "
                "strictly positive entry."
            )
    if transition_metric is not None:
        if transition_metric.ndim != 2 or transition_metric.shape != (d, d):
            raise ValueError(
                f"`transition_metric` must have shape `({d}, {d})`, "
                f"got {transition_metric.shape}."
            )
        w_sym = 0.5 * (transition_metric + transition_metric.transpose(-1, -2))
        if not torch.allclose(transition_metric, w_sym, atol=1e-5, rtol=1e-4):
            raise ValueError("`transition_metric` must be symmetric.")
        min_eig = torch.linalg.eigvalsh(w_sym.to(dtype=torch.float64)).min().item()
        if min_eig < -1e-6:
            raise ValueError(
                "`transition_metric` must be positive semi-definite "
                f"(min eigenvalue: {min_eig:.3e})."
            )


def _has_normalize_input_transform(model: Model) -> bool:
    r"""Check whether ``model`` applies a ``Normalize`` input transform.

    Args:
        model: A BoTorch model to inspect.

    Returns:
        True if ``model.input_transform`` is or contains ``Normalize``.
    """
    tf = getattr(model, "input_transform", None)
    if isinstance(tf, Normalize):
        return True
    if isinstance(tf, ChainedInputTransform):
        return any(isinstance(sub_tf, Normalize) for sub_tf in tf.values())
    return False


def _extract_observed_inputs(model: Model | None, bounds: Tensor) -> Tensor | None:
    r"""Extract untransformed observed design points ``(N, d)`` from ``model``.

    Args:
        model: Optional BoTorch model with training inputs.
        bounds: A ``2 x d``-dim tensor providing target dtype/device and ``d``.

    Returns:
        A ``(N, d)`` tensor of observed points, or ``None`` if unavailable.
    """
    if model is None:
        return None
    d = bounds.shape[-1]
    try:
        raw_inputs = get_train_inputs(model, transformed=False)
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
        return None

    tensors: list[Tensor] = []
    if isinstance(raw_inputs, list):
        for item in raw_inputs:
            if isinstance(item, tuple) and len(item) > 0 and torch.is_tensor(item[0]):
                x = item[0]
                if x.shape[-1] == d and x.numel() > 0:
                    if x.ndim > 2:
                        x = x[(0,) * (x.ndim - 2)]
                    tensors.append(x.reshape(-1, d))
    elif (
        isinstance(raw_inputs, tuple)
        and len(raw_inputs) > 0
        and torch.is_tensor(raw_inputs[0])
    ):
        x = raw_inputs[0]
        if x.shape[-1] == d and x.numel() > 0:
            if x.ndim > 2:
                x = x[(0,) * (x.ndim - 2)]
            tensors.append(x.reshape(-1, d))

    if not tensors:
        return None
    X_all = torch.cat(tensors, dim=0).detach()
    return X_all.to(device=bounds.device, dtype=bounds.dtype)


def _extract_last_observed_point(model: Model | None, bounds: Tensor) -> Tensor | None:
    r"""Extract the most recently observed input ``(1, d)`` from ``model``.

    Args:
        model: Optional BoTorch model with training inputs.
        bounds: A ``2 x d``-dim tensor providing target dtype/device and ``d``.

    Returns:
        A ``(1, d)`` tensor for ``train_inputs[-1]``, or ``None`` if unavailable.
    """
    X_obs = _extract_observed_inputs(model=model, bounds=bounds)
    if X_obs is None or X_obs.shape[0] == 0:
        return None
    return X_obs[-1:].clone()


def _extract_negated_target_bounds(
    model: Model, num_outputs: int, bounds: Tensor
) -> tuple[Tensor, Tensor] | None:
    r"""Extract per-objective bounds on ``-Y`` for Chebyshev scalarization.

    Args:
        model: A fitted multi-output BoTorch model.
        num_outputs: Number of model outputs ``m``.
        bounds: A ``2 x d``-dim tensor providing target dtype and device.

    Returns:
        Tuple ``(y_min, y_scale)`` of ``(m,)`` tensors, or ``None`` if unavailable.
    """
    try:
        raw_targets = get_train_targets(model, transformed=False)
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
        return None

    tkwargs = {"device": bounds.device, "dtype": bounds.dtype}
    if isinstance(raw_targets, list):
        if len(raw_targets) != num_outputs:
            return None
        mins: list[Tensor] = []
        maxs: list[Tensor] = []
        for y_sub in raw_targets:
            if not torch.is_tensor(y_sub) or y_sub.numel() == 0:
                return None
            neg_y = (-y_sub.detach()).to(**tkwargs)
            mins.append(neg_y.min())
            maxs.append(neg_y.max())
        y_min = torch.stack(mins)
        y_max = torch.stack(maxs)
    elif torch.is_tensor(raw_targets) and raw_targets.numel() > 0:
        neg_y = (-raw_targets.detach()).to(**tkwargs)
        if neg_y.shape[-1] == num_outputs and neg_y.ndim >= 2:
            flat_y = neg_y.reshape(-1, num_outputs)
        elif neg_y.ndim >= 2 and neg_y.shape[-2] == num_outputs:
            flat_y = neg_y.transpose(-2, -1).reshape(-1, num_outputs)
        else:
            return None
        y_min = flat_y.min(dim=0).values
        y_max = flat_y.max(dim=0).values
    else:
        return None

    y_scale = (y_max - y_min).clamp_min(1e-8)
    return y_min, y_scale


def _compute_adaptive_epsilon(
    model: Model | None,
    bounds: Tensor,
    X_observed: Tensor | None,
) -> float:
    r"""Compute lengthscale-aware normalized deletion radius for :math:`\epsilon`-SnAKe.

    Following [folch2022snake]_ and ``cog-imperial/SnAKe``'s ``bo_rescaling`` mode,
    sets :math:`\epsilon` from the GP kernel lengthscales mapped into normalized
    :math:`[0, 1]^d` coordinates (``0.5 * mean(lengthscale_norm)``), falling back
    to :math:`N^{-1/d}` when the model has no kernel lengthscales.

    Args:
        model: Optional BoTorch model whose kernel lengthscales are inspected.
        bounds: A ``2 x d``-dim tensor of domain bounds.
        X_observed: Optional ``(N, d)`` tensor of observed inputs.

    Returns:
        Positive float deletion radius in normalized ``[0, 1]^d`` Euclidean units.
    """
    d = bounds.shape[-1]
    span = (bounds[1] - bounds[0]).detach().to(dtype=torch.float64).clamp_min(1e-12)

    norm_lengthscales: list[Tensor] = []
    if model is not None:
        submodels = list(model.models) if isinstance(model, ModelList) else [model]
        for sub_m in submodels:
            is_norm = _has_normalize_input_transform(sub_m)
            for mod in sub_m.modules():
                if isinstance(mod, Kernel) and mod.has_lengthscale:
                    ls = mod.lengthscale
                    if ls is None:
                        continue
                    ls_det = ls.detach().to(device=span.device, dtype=torch.float64)
                    if ls_det.shape[-1] == 1 and d > 1:
                        ls_det = ls_det.expand(*ls_det.shape[:-1], d)
                    if not is_norm and ls_det.shape[-1] == d:
                        ls_det = ls_det / span
                    norm_lengthscales.append(ls_det.reshape(-1))

    if norm_lengthscales:
        mean_ls = torch.cat(norm_lengthscales).mean().item()
        max_diam = math.sqrt(float(d)) if d > 1 else 1.0
        return float(min(max(0.5 * mean_ls, 1e-5), max_diam))

    n_obs = 0 if X_observed is None else int(X_observed.shape[0])
    if n_obs >= 1:
        return float(n_obs ** (-1.0 / d))
    return 0.1


def _make_scalarizer(
    model: Model,
    budget: int,
    bounds: Tensor,
    posterior_transform: PosteriorTransform | None,
    chebyshev_alpha: float,
    seed: int | None,
) -> Callable[[Tensor, Tensor], Tensor]:
    r"""Build a vectorized outcome scalarizer mapping ``(B, q, m) -> (B, q)``.

    Args:
        model: Fitted BoTorch model with ``m = model.num_outputs`` outcomes.
        budget: Number of Thompson samples ``B``.
        bounds: A ``2 x d``-dim tensor of domain bounds.
        posterior_transform: Optional explicit ``PosteriorTransform``.
        chebyshev_alpha: Augmentation weight for random ParEGO/Chebyshev
            scalarization when ``m > 1`` and ``posterior_transform is None``.
        seed: Optional random seed for simplex weight generation.

    Returns:
        Callable ``scalarize(Y, X)`` returning a ``(B, q)`` scalarized tensor.
    """
    num_outputs = model.num_outputs
    if posterior_transform is not None:

        def _transform_scalarizer(Y: Tensor, X: Tensor) -> Tensor:
            X_exp = X.expand(Y.shape[:-1] + (X.shape[-1],))
            val = posterior_transform.evaluate(Y=Y, X=X_exp)
            if val.ndim == Y.ndim and val.shape[-1] == 1:
                return val.squeeze(-1)
            if val.ndim == Y.ndim:
                return val.mean(dim=-1)
            return val

        return _transform_scalarizer

    if num_outputs == 1:

        def _single_output_scalarizer(Y: Tensor, X: Tensor) -> Tensor:
            return Y.squeeze(-1)

        return _single_output_scalarizer

    weights = sample_simplex(
        d=num_outputs,
        n=budget,
        qmc=True,
        seed=seed,
        device=bounds.device,
        dtype=bounds.dtype,
    )
    target_bounds = _extract_negated_target_bounds(
        model=model, num_outputs=num_outputs, bounds=bounds
    )

    def _chebyshev_scalarizer(Y: Tensor, X: Tensor) -> Tensor:
        neg_y = -Y
        if target_bounds is not None:
            y_min, y_scale = target_bounds
            neg_y = (neg_y - y_min) / y_scale
        w = weights.unsqueeze(1)
        prod = w * neg_y
        return -(prod.max(dim=-1).values + chebyshev_alpha * prod.sum(dim=-1))

    return _chebyshev_scalarizer


def _evaluate_matheron_paths(
    paths: SamplePath,
    model: Model,
    X: Tensor,
) -> Tensor:
    r"""Evaluate ``B`` Matheron sample paths at ``X`` with output shape ``(B, q, m)``.

    Args:
        paths: Sample paths returned by ``draw_matheron_paths``.
        model: Underlying BoTorch model.
        X: Input tensor of shape ``(q, d)`` or ``(B, q, d)``.

    Returns:
        Tensor of shape ``(B, q, m)`` containing sample path values.
    """
    if isinstance(paths, PathList):
        raw_list = paths(X)
        stacked = torch.stack(raw_list, dim=-1)
        if stacked.ndim == 4:
            stacked = stacked.mean(dim=1)
        return stacked

    if model.num_outputs == 1:
        raw = paths(X)
        if is_ensemble(model) and raw.ndim == 3:
            raw = raw.mean(dim=1)
        return raw.unsqueeze(-1)

    raw = paths(X.unsqueeze(-3)).transpose(-1, -2)
    if raw.ndim == 4:
        raw = raw.mean(dim=1)
    return raw


def sample_thompson_optima(
    model: Model,
    bounds: Tensor,
    budget: int,
    posterior_transform: PosteriorTransform | None = None,
    discrete_candidates: bool = False,
    num_candidates: int = 1024,
    num_restarts: int = 8,
    num_opt_steps: int = 50,
    lr: float = 0.05,
    chebyshev_alpha: float = 0.05,
    warm_start_points: Tensor | None = None,
    seed: int | None = None,
) -> Tensor:
    r"""Generate a lookahead batch of :math:`B` Thompson-sampled optima.

    Draws :math:`B` posterior realizations from ``model`` and maximizes each
    realization over ``bounds``. By default, continuous Matheron sample paths are
    drawn via ``draw_matheron_paths`` and maximized with multi-start Sobol
    initialization (optionally augmented with ``warm_start_points``) followed by
    projected Adam gradient ascent. When pathwise sampling raises
    ``UnsupportedError`` or ``NotImplementedError`` (or when
    ``discrete_candidates=True``), falls back to drawing :math:`B` joint posterior
    samples on a Sobol candidate pool via ``model.posterior(X_cand).rsample([B])``.

    For multi-output models without an explicit ``posterior_transform``, each
    Thompson sample is scalarized with a distinct QMC-sampled augmented Chebyshev
    (ParEGO) weight vector so SnAKe naturally explores Pareto fronts in MOBO.

    Args:
        model: A fitted BoTorch ``Model`` (single-output or multi-output).
        bounds: A ``2 x d``-dim tensor of lower and upper bounds.
        budget: Number of Thompson optima ``B >= 1`` to sample.
        posterior_transform: Optional ``PosteriorTransform`` applied to outcomes.
        discrete_candidates: If True, bypass continuous Matheron paths and sample
            directly over a discrete Sobol candidate pool.
        num_candidates: Number of Sobol candidates in the initialization/fallback
            pool. Defaults to 1024.
        num_restarts: Number of top candidates per path refined via gradient
            ascent in pathwise mode. Defaults to 8.
        num_opt_steps: Number of projected Adam steps for pathwise maximization.
            Defaults to 50.
        lr: Learning rate for Adam in normalized ``[0, 1]^d`` space.
            Defaults to 0.05.
        chebyshev_alpha: Augmentation coefficient for multi-output Chebyshev
            scalarization when ``posterior_transform is None``. Defaults to 0.05.
        warm_start_points: Optional ``(k, d)`` tensor of unvisited points from a
            previous planned trajectory used to warm-start candidate selection.
        seed: Optional integer random seed for reproducibility.

    Returns:
        A ``(B, d)``-dim tensor of sampled Thompson optima within ``bounds``.
    """
    _validate_bounds(bounds)
    if budget < 1:
        raise ValueError(f"`budget` must be >= 1, got {budget}.")
    if num_candidates < 1:
        raise ValueError(f"`num_candidates` must be >= 1, got {num_candidates}.")
    if num_restarts < 1:
        raise ValueError(f"`num_restarts` must be >= 1, got {num_restarts}.")
    if num_opt_steps < 0:
        raise ValueError(f"`num_opt_steps` must be >= 0, got {num_opt_steps}.")
    if lr <= 0.0:
        raise ValueError(f"`lr` must be > 0, got {lr}.")

    d = bounds.shape[-1]
    model.eval()

    X_pool = draw_sobol_samples(
        bounds=bounds, n=num_candidates, q=1, seed=seed
    ).squeeze(-2)
    if warm_start_points is not None and warm_start_points.numel() > 0:
        if warm_start_points.ndim != 2 or warm_start_points.shape[-1] != d:
            raise ValueError(
                f"`warm_start_points` must have shape `(k, {d})`, "
                f"got {warm_start_points.shape}."
            )
        warm_clamped = (
            warm_start_points.detach()
            .to(device=bounds.device, dtype=bounds.dtype)
            .clamp(min=bounds[0], max=bounds[1])
        )
        X_pool = torch.cat([X_pool, warm_clamped], dim=0)

    scalarize = _make_scalarizer(
        model=model,
        budget=budget,
        bounds=bounds,
        posterior_transform=posterior_transform,
        chebyshev_alpha=chebyshev_alpha,
        seed=seed,
    )

    paths: SamplePath | None = None
    if not discrete_candidates:
        try:
            with torch.no_grad(), manual_seed(seed=seed):
                paths = draw_matheron_paths(model=model, sample_shape=Size([budget]))
        except (UnsupportedError, NotImplementedError):
            paths = None

    if paths is not None:
        with torch.no_grad():
            pool_y = _evaluate_matheron_paths(paths=paths, model=model, X=X_pool)
            pool_vals = scalarize(pool_y, X_pool)
            r = min(num_restarts, X_pool.shape[0])
            top_vals, top_idx = torch.topk(pool_vals, k=r, dim=-1)
            X_init = X_pool[top_idx]

        if num_opt_steps == 0:
            return X_init[:, 0, :].clamp(min=bounds[0], max=bounds[1]).detach()

        Z_init = normalize(X_init, bounds=bounds).clamp(0.0, 1.0)
        with torch.enable_grad():
            Z_var = Z_init.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([Z_var], lr=lr)
            for _ in range(num_opt_steps):
                optimizer.zero_grad()
                X_cur = unnormalize(Z_var, bounds=bounds)
                y_cur = _evaluate_matheron_paths(paths=paths, model=model, X=X_cur)
                vals_cur = scalarize(y_cur, X_cur)
                loss = -vals_cur.sum()
                (Z_var.grad,) = torch.autograd.grad(loss, Z_var)
                optimizer.step()
                with torch.no_grad():
                    Z_var.clamp_(0.0, 1.0)

        with torch.no_grad():
            X_final = unnormalize(Z_var, bounds=bounds)
            y_final = _evaluate_matheron_paths(paths=paths, model=model, X=X_final)
            vals_final = scalarize(y_final, X_final)
            max_final_val, max_r = vals_final.max(dim=-1)
            batch_idx = torch.arange(budget, device=bounds.device)
            X_best_r = X_final[batch_idx, max_r]
            improved = max_final_val > top_vals[:, 0]
            X_opt = torch.where(improved.unsqueeze(-1), X_best_r, X_init[:, 0, :])
            return X_opt.clamp(min=bounds[0], max=bounds[1]).detach()

    # Fallback / discrete posterior Thompson sampling mode.
    with torch.no_grad(), manual_seed(seed=seed):
        vals_disc: Tensor | None = None
        if posterior_transform is not None:
            try:
                posterior = model.posterior(
                    X_pool, posterior_transform=posterior_transform
                )
                samples = posterior.rsample(sample_shape=Size([budget]))
                if samples.ndim == 4:
                    samples = samples.mean(dim=1)
                if samples.ndim == 3 and samples.shape[-1] == 1:
                    vals_disc = samples.squeeze(-1)
                elif samples.ndim == 2:
                    vals_disc = samples
                else:
                    vals_disc = samples.mean(dim=-1)
            except (
                AttributeError,
                NotImplementedError,
                TypeError,
                UnsupportedError,
            ):
                vals_disc = None

        if vals_disc is None:
            raw_samples = model.posterior(X_pool).rsample(sample_shape=Size([budget]))
            if raw_samples.ndim == 4:
                raw_samples = raw_samples.mean(dim=1)
            if raw_samples.ndim == 2:
                raw_samples = raw_samples.unsqueeze(-1)
            vals_disc = scalarize(raw_samples, X_pool)

        best_idx = vals_disc.argmax(dim=-1)
        return X_pool[best_idx].clone().detach()


def apply_epsilon_point_deletion(
    X_candidates: Tensor,
    bounds: Tensor,
    X_observed: Tensor | None = None,
    epsilon: float | str | None = "adaptive",
    model: Model | None = None,
    seed: int | None = None,
) -> Tensor:
    r"""Apply :math:`\epsilon`-point deletion to prevent local trapping in SnAKe.

    Implements Section 3.2 of [folch2022snake]_: sequentially inspects each
    prospective query point :math:`x_s \in X_{\text{candidates}}` in normalized
    :math:`[0, 1]^d` coordinates. Whenever the Euclidean distance from :math:`x_s`
    to any previously evaluated point in ``X_observed`` or any already-accepted
    batch point is ``<= epsilon``, :math:`x_s` is deleted and replaced by a fresh
    space-filling uniform random sample inside ``bounds``.

    Args:
        X_candidates: A ``(B, d)``-dim tensor of prospective Thompson optima.
        bounds: A ``2 x d``-dim tensor of lower and upper domain bounds.
        X_observed: Optional ``(N, d)``-dim tensor of already-evaluated inputs. If
            ``None`` and ``model`` is provided, extracted from ``model``.
        epsilon: Deletion radius specification:
            - ``0.0`` or ``None``: Pure unperturbed SnAKe (no deletion).
            - Positive ``float``: Fixed normalized Euclidean radius in ``[0, 1]^d``.
            - ``"adaptive"`` or ``"bo_rescaling"``: Lengthscale-aware radius
              ``0.5 * mean(lengthscale)`` in normalized space (with ``N^(-1/d)``
              fallback when ``model`` has no kernel lengthscales).
        model: Optional BoTorch model used for lengthscale extraction in adaptive
            modes and as a fallback source for ``X_observed``.
        seed: Optional integer random seed for replacement point generation.

    Returns:
        A ``(B, d)``-dim tensor of filtered/replenished query points in ``bounds``.
    """
    _validate_bounds(bounds)
    _validate_epsilon(epsilon)
    d = bounds.shape[-1]
    if X_candidates.ndim != 2 or X_candidates.shape[-1] != d:
        raise ValueError(
            f"`X_candidates` must have shape `(B, {d})`, got {X_candidates.shape}."
        )
    if X_candidates.shape[0] == 0:
        return X_candidates.clone()

    X_out = (
        X_candidates.detach()
        .to(device=bounds.device, dtype=bounds.dtype)
        .clamp(min=bounds[0], max=bounds[1])
        .clone()
    )

    if epsilon is None or (isinstance(epsilon, (int, float)) and epsilon == 0.0):
        return X_out

    if X_observed is None and model is not None:
        X_observed = _extract_observed_inputs(model=model, bounds=bounds)
    elif X_observed is not None:
        if X_observed.ndim != 2 or X_observed.shape[-1] != d:
            raise ValueError(
                f"`X_observed` must have shape `(N, {d})`, got {X_observed.shape}."
            )
        X_observed = X_observed.detach().to(device=bounds.device, dtype=bounds.dtype)

    if isinstance(epsilon, str):
        eps_val = _compute_adaptive_epsilon(
            model=model, bounds=bounds, X_observed=X_observed
        )
    else:
        eps_val = float(epsilon)

    if eps_val <= 0.0:
        return X_out

    Z_cand = normalize(X_out, bounds=bounds).clamp(0.0, 1.0)
    if X_observed is not None and X_observed.shape[0] > 0:
        Z_ref = normalize(X_observed, bounds=bounds).clamp(0.0, 1.0)
    else:
        Z_ref = torch.empty((0, d), device=bounds.device, dtype=bounds.dtype)

    num_rejection_pool = 64
    with manual_seed(seed=seed):
        for i in range(Z_cand.shape[0]):
            z_i = Z_cand[i : i + 1]
            needs_replacement = False
            if Z_ref.shape[0] > 0:
                min_dist = torch.cdist(z_i, Z_ref, p=2).min().item()
                needs_replacement = min_dist <= eps_val

            if needs_replacement:
                z_pool = torch.rand(
                    (num_rejection_pool, d),
                    device=bounds.device,
                    dtype=bounds.dtype,
                )
                dists = torch.cdist(z_pool, Z_ref, p=2).min(dim=-1).values
                valid_mask = dists > eps_val
                if valid_mask.any():
                    first_valid = int(valid_mask.nonzero(as_tuple=False)[0].item())
                    z_new = z_pool[first_valid : first_valid + 1]
                else:
                    best_sep = int(dists.argmax().item())
                    z_new = z_pool[best_sep : best_sep + 1]
                Z_cand[i : i + 1] = z_new
                z_i = z_new

            Z_ref = torch.cat([Z_ref, z_i], dim=0)

    return unnormalize(Z_cand, bounds=bounds).clamp(min=bounds[0], max=bounds[1])


def _compute_pairwise_transition_cost(
    X_a: Tensor,
    X_b: Tensor,
    transition_weights: Tensor | None = None,
    transition_metric: Tensor | None = None,
) -> Tensor:
    r"""Compute pairwise transition costs ``d(X_a[i], X_b[j])`` in float64.

    Args:
        X_a: A ``(n_a, d)``-dim tensor of source points.
        X_b: A ``(n_b, d)``-dim tensor of destination points.
        transition_weights: Optional ``(d,)`` per-parameter scaling weights ``w``
            defining ``d(u, v) = ||w * (u - v)||_2``.
        transition_metric: Optional ``(d, d)`` symmetric PSD matrix ``W``
            defining ``d(u, v) = sqrt((u - v)^T W (u - v))``.

    Returns:
        A ``(n_a, n_b)``-dim float64 tensor of non-negative transition costs.
    """
    a64 = X_a.detach().to(device="cpu", dtype=torch.float64)
    b64 = X_b.detach().to(device="cpu", dtype=torch.float64)

    if transition_weights is not None:
        w64 = transition_weights.detach().to(device="cpu", dtype=torch.float64)
        a64 = a64 * w64
        b64 = b64 * w64
        return torch.cdist(a64, b64, p=2)

    if transition_metric is not None:
        W64 = transition_metric.detach().to(device="cpu", dtype=torch.float64)
        W_sym = 0.5 * (W64 + W64.transpose(-1, -2))
        diff = a64.unsqueeze(1) - b64.unsqueeze(0)
        quad = torch.einsum("ijd,dk,ijk->ij", diff, W_sym, diff)
        return quad.clamp_min(0.0).sqrt()

    return torch.cdist(a64, b64, p=2)


def _solve_held_karp_open_tsp(
    start_cost: Tensor,
    pair_cost: Tensor,
) -> list[int]:
    r"""Solve anchored Open-TSP exactly via vectorized Held-Karp dynamic programming.

    Args:
        start_cost: A ``(B,)`` float64 CPU tensor of costs from ``x_current``.
        pair_cost: A ``(B, B)`` float64 CPU tensor of inter-point costs.

    Returns:
        List of ``B`` permutation indices minimizing cumulative transition cost.
    """
    B = int(start_cost.shape[0])
    if B == 1:
        return [0]

    num_states = 1 << B
    dp = torch.full((num_states, B), float("inf"), dtype=torch.float64)
    parent = torch.full((num_states, B), -1, dtype=torch.long)

    for i in range(B):
        dp[1 << i, i] = start_cost[i]

    all_masks = torch.arange(num_states, dtype=torch.long)
    popcounts = torch.zeros(num_states, dtype=torch.long)
    for bit in range(B):
        popcounts += (all_masks >> bit) & 1

    for k in range(1, B):
        masks_k = all_masks[popcounts == k]
        for nxt in range(B):
            bit_mask = 1 << nxt
            valid = (masks_k & bit_mask) == 0
            prev_masks = masks_k[valid]
            if prev_masks.numel() == 0:
                continue
            next_masks = prev_masks | bit_mask
            cand_costs = dp[prev_masks] + pair_cost[:, nxt]
            best_costs, best_prev = cand_costs.min(dim=1)
            dp[next_masks, nxt] = best_costs
            parent[next_masks, nxt] = best_prev

    full_mask = num_states - 1
    curr_node = int(dp[full_mask].argmin().item())
    curr_mask = full_mask
    rev_order: list[int] = []
    while curr_node != -1:
        rev_order.append(curr_node)
        prev_node = int(parent[curr_mask, curr_node].item())
        curr_mask = curr_mask ^ (1 << curr_node)
        curr_node = prev_node

    rev_order.reverse()
    return rev_order


def _solve_2opt_open_tsp(
    start_cost: Tensor,
    pair_cost: Tensor,
    max_passes: int = 200,
) -> list[int]:
    r"""Solve anchored Open-TSP heuristically via Nearest-Neighbor + 2-opt search.

    Args:
        start_cost: A ``(B,)`` float64 CPU tensor of costs from ``x_current``.
        pair_cost: A ``(B, B)`` float64 CPU tensor of inter-point costs.
        max_passes: Maximum number of 2-opt improvement sweeps.

    Returns:
        List of ``B`` permutation indices ordering the trajectory.
    """
    B = int(start_cost.shape[0])
    if B == 1:
        return [0]

    # Build (B + 1) x (B + 1) augmented symmetric cost matrix with 0 = x_current.
    C = torch.zeros((B + 1, B + 1), dtype=torch.float64)
    C[0, 1:] = start_cost
    C[1:, 0] = start_cost
    C[1:, 1:] = pair_cost

    # Nearest-neighbor initialization starting from node 0 (x_current).
    unvisited = [True] * B
    curr = int(start_cost.argmin().item())
    unvisited[curr] = False
    tour = [0, curr + 1]
    for _ in range(B - 1):
        row = pair_cost[curr].clone()
        for idx, is_free in enumerate(unvisited):
            if not is_free:
                row[idx] = float("inf")
        nxt = int(row.argmin().item())
        unvisited[nxt] = False
        tour.append(nxt + 1)
        curr = nxt

    C_list = C.tolist()
    tol = 1e-12
    for _ in range(max_passes):
        improved = False
        for i in range(1, B):
            prev_node = tour[i - 1]
            node_i = tour[i]
            c_prev_i = C_list[prev_node][node_i]
            for j in range(i + 1, B + 1):
                node_j = tour[j]
                if j < B:
                    next_node = tour[j + 1]
                    old_edges = c_prev_i + C_list[node_j][next_node]
                    new_edges = C_list[prev_node][node_j] + C_list[node_i][next_node]
                else:
                    # Open endpoint: reversing suffix tour[i : B + 1] only
                    # replaces edge (tour[i - 1], tour[i]) with (tour[i - 1], tour[B]).
                    old_edges = c_prev_i
                    new_edges = C_list[prev_node][node_j]

                if new_edges + tol < old_edges:
                    tour[i : j + 1] = reversed(tour[i : j + 1])
                    node_i = tour[i]
                    c_prev_i = C_list[prev_node][node_i]
                    improved = True
        if not improved:
            break

    return [node - 1 for node in tour[1:]]


def solve_open_tsp_path(
    X_candidates: Tensor,
    x_current: Tensor | None = None,
    model: Model | None = None,
    transition_weights: Tensor | None = None,
    transition_metric: Tensor | None = None,
    exact_threshold: int = 13,
) -> tuple[Tensor, Tensor]:
    r"""Order prospective query points via Open-TSP to minimize movement cost.

    Orders the :math:`B` candidate points in ``X_candidates`` starting from
    :math:`x^{(0)} = x_{\text{current}}` (without returning to the origin) to
    minimize cumulative transition cost
    :math:`\sum_{k=0}^{B-1} d(x^{(k+1)}, x^{(k)})` [folch2022snake]_. Uses exact
    Held-Karp dynamic programming when ``B <= exact_threshold`` (default 13) and
    nearest-neighbor construction followed by open-path 2-opt local search for
    larger lookahead batches.

    Args:
        X_candidates: A ``(B, d)``-dim tensor of prospective query points.
        x_current: Optional ``(d,)`` or ``(1, d)``-dim tensor giving the current
            operating point. Defaults to the last observed point
            ``model.train_inputs[-1]`` when ``model`` is provided.
        model: Optional BoTorch model used to infer ``x_current`` when omitted.
        transition_weights: Optional ``(d,)``-dim tensor of non-negative per-input
            movement weights ``w`` defining ``d(u, v) = ||w * (u - v)||_2``.
        transition_metric: Optional ``(d, d)``-dim symmetric positive
            semi-definite matrix ``W`` defining Mahalanobis transition distance
            ``d(u, v) = sqrt((u - v)^T W (u - v))``.
        exact_threshold: Maximum batch size ``B`` solved exactly via Held-Karp DP
            before switching to nearest-neighbor + 2-opt. Defaults to 13.

    Returns:
        A tuple ``(ordered_path, order_indices)`` where ``ordered_path`` has shape
        ``(B, d)`` and ``order_indices`` is a ``(B,)`` long tensor satisfying
        ``ordered_path == X_candidates[order_indices]``.
    """
    if X_candidates.ndim != 2 or X_candidates.shape[0] < 1:
        raise ValueError(
            "`X_candidates` must be a 2D tensor with `B >= 1` rows, "
            f"got {X_candidates.shape}."
        )
    if exact_threshold < 1:
        raise ValueError(f"`exact_threshold` must be >= 1, got {exact_threshold}.")

    B, d = X_candidates.shape
    _validate_transition_cost_args(
        d=d,
        transition_weights=transition_weights,
        transition_metric=transition_metric,
    )

    if x_current is None:
        if model is not None:
            ref_bounds = torch.zeros(
                (2, d), device=X_candidates.device, dtype=X_candidates.dtype
            )
            ref_bounds[1] = 1.0
            x_curr_2d = _extract_last_observed_point(model=model, bounds=ref_bounds)
        else:
            x_curr_2d = None
        if x_curr_2d is None:
            raise ValueError(
                "`x_current` must be provided when `model` is None or has no "
                "training inputs."
            )
    else:
        if x_current.shape == (d,):
            x_curr_2d = x_current.unsqueeze(0)
        elif x_current.shape == (1, d):
            x_curr_2d = x_current
        else:
            raise ValueError(
                f"`x_current` must have shape `({d},)` or `(1, {d})`, "
                f"got {x_current.shape}."
            )
        x_curr_2d = x_curr_2d.to(device=X_candidates.device, dtype=X_candidates.dtype)

    start_cost = _compute_pairwise_transition_cost(
        X_a=x_curr_2d,
        X_b=X_candidates,
        transition_weights=transition_weights,
        transition_metric=transition_metric,
    ).squeeze(0)
    pair_cost = _compute_pairwise_transition_cost(
        X_a=X_candidates,
        X_b=X_candidates,
        transition_weights=transition_weights,
        transition_metric=transition_metric,
    )

    if B <= exact_threshold:
        order_list = _solve_held_karp_open_tsp(
            start_cost=start_cost, pair_cost=pair_cost
        )
    else:
        order_list = _solve_2opt_open_tsp(start_cost=start_cost, pair_cost=pair_cost)

    order_indices = torch.tensor(
        order_list, dtype=torch.long, device=X_candidates.device
    )
    ordered_path = X_candidates.index_select(dim=0, index=order_indices)
    return ordered_path, order_indices


class SnAKe:
    r"""Stateful receding-horizon policy for SnAKe and :math:`\epsilon`-SnAKe.

    Implements the full online trajectory planning loop of [folch2022snake]_:
    at replanning steps, draws a remaining-horizon batch of Thompson optima
    (warm-started with unvisited points from the previous planned trajectory),
    applies :math:`\epsilon`-point deletion against observed and batch points, and
    orders the lookahead trajectory from the current input via Open-TSP.

    Supports both step-by-step receding-horizon replanning (``replan_interval=1``)
    and truncated/periodic replanning (``replan_interval > 1``), where intermediate
    ``suggest`` calls pop the next point directly from the cached trajectory.
    """

    def __init__(
        self,
        budget: int = 20,
        epsilon: float | str | None = "adaptive",
        replan_interval: int = 1,
        posterior_transform: PosteriorTransform | None = None,
        transition_weights: Tensor | None = None,
        transition_metric: Tensor | None = None,
        discrete_candidates: bool = False,
        num_candidates: int = 1024,
        num_restarts: int = 8,
        num_opt_steps: int = 50,
        lr: float = 0.05,
        chebyshev_alpha: float = 0.05,
        exact_tsp_threshold: int = 13,
        warm_start: bool = True,
        seed: int | None = None,
    ) -> None:
        r"""Initialize the ``SnAKe`` policy.

        Args:
            budget: Total evaluation horizon ``T >= 1`` used for receding-horizon
                lookahead ``B_t = max(1, T - t)`` when ``remaining_budget`` is not
                explicitly supplied to ``suggest``. Defaults to 20.
            epsilon: Point-deletion radius configuration (``None`` or ``0.0`` for
                pure SnAKe, positive ``float`` for fixed normalized radius, or
                ``"adaptive"`` / ``"bo_rescaling"`` for GP lengthscale rescaling).
            replan_interval: Number of ``suggest`` steps between full GP Thompson
                + TSP replans (``1`` replans at every step; ``> 1`` pops cached
                waypoints between replans). Defaults to 1.
            posterior_transform: Optional ``PosteriorTransform`` for scalarizing
                or transforming model outputs.
            transition_weights: Optional ``(d,)`` tensor of per-parameter movement
                cost weights.
            transition_metric: Optional ``(d, d)`` symmetric PSD matrix defining
                Mahalanobis movement cost.
            discrete_candidates: If True, always use discrete candidate Thompson
                sampling instead of continuous Matheron paths.
            num_candidates: Number of Sobol candidates per Thompson draw.
            num_restarts: Number of multi-start restarts for pathwise Adam ascent.
            num_opt_steps: Number of projected Adam gradient steps per path.
            lr: Adam learning rate in normalized ``[0, 1]^d`` space.
            chebyshev_alpha: Augmentation weight for random multi-output Chebyshev
                scalarization when ``posterior_transform is None``.
            exact_tsp_threshold: Maximum lookahead size ``B`` solved via exact
                Held-Karp DP before switching to 2-opt. Defaults to 13.
            warm_start: If True, warm-start Thompson candidate optimization with
                unvisited points from the previous planned path. Defaults to True.
            seed: Optional base integer seed for reproducible planning.
        """
        if budget < 1:
            raise ValueError(f"`budget` must be >= 1, got {budget}.")
        if replan_interval < 1:
            raise ValueError(f"`replan_interval` must be >= 1, got {replan_interval}.")
        if num_candidates < 1:
            raise ValueError(f"`num_candidates` must be >= 1, got {num_candidates}.")
        if num_restarts < 1:
            raise ValueError(f"`num_restarts` must be >= 1, got {num_restarts}.")
        if num_opt_steps < 0:
            raise ValueError(f"`num_opt_steps` must be >= 0, got {num_opt_steps}.")
        if lr <= 0.0:
            raise ValueError(f"`lr` must be > 0, got {lr}.")
        if exact_tsp_threshold < 1:
            raise ValueError(
                f"`exact_tsp_threshold` must be >= 1, got {exact_tsp_threshold}."
            )
        _validate_epsilon(epsilon)
        if transition_weights is not None and transition_metric is not None:
            raise ValueError(
                "Cannot specify both `transition_weights` and `transition_metric`."
            )

        self.budget = budget
        self.epsilon = epsilon
        self.replan_interval = replan_interval
        self.posterior_transform = posterior_transform
        self.transition_weights = transition_weights
        self.transition_metric = transition_metric
        self.discrete_candidates = discrete_candidates
        self.num_candidates = num_candidates
        self.num_restarts = num_restarts
        self.num_opt_steps = num_opt_steps
        self.lr = lr
        self.chebyshev_alpha = chebyshev_alpha
        self.exact_tsp_threshold = exact_tsp_threshold
        self.warm_start = warm_start
        self.seed = seed

        self._step_count: int = 0
        self._steps_since_replan: int = 0
        self._cached_path: Tensor | None = None
        self._last_x_suggested: Tensor | None = None

    @property
    def step_count(self) -> int:
        r"""Number of ``suggest`` steps executed since initialization or ``reset``."""
        return self._step_count

    @property
    def cached_path(self) -> Tensor | None:
        r"""Remaining unvisited waypoints from the most recent planned trajectory."""
        return None if self._cached_path is None else self._cached_path.clone()

    def reset(self) -> None:
        r"""Clear cached trajectory and step counters to restart planning."""
        self._step_count = 0
        self._steps_since_replan = 0
        self._cached_path = None
        self._last_x_suggested = None

    def suggest(
        self,
        model: Model,
        bounds: Tensor,
        x_current: Tensor | None = None,
        remaining_budget: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Suggest the next query point and lookahead trajectory.

        Args:
            model: Fitted BoTorch ``Model`` conditioned on observations so far.
            bounds: A ``2 x d``-dim tensor of lower and upper domain bounds.
            x_current: Optional ``(d,)`` or ``(1, d)``-dim tensor specifying the
                current input position. Defaults to ``model.train_inputs[-1]`` (or
                the previous suggested point / domain center if unavailable).
            remaining_budget: Optional explicit lookahead horizon ``B >= 1``.
                Defaults to ``max(1, self.budget - self.step_count)``.

        Returns:
            A tuple ``(next_x, full_planned_path)`` of shapes ``(1, d)`` and
            ``(B, d)``, where ``next_x == full_planned_path[:1]``.
        """
        _validate_bounds(bounds)
        d = bounds.shape[-1]
        _validate_transition_cost_args(
            d=d,
            transition_weights=self.transition_weights,
            transition_metric=self.transition_metric,
        )

        if remaining_budget is not None:
            if remaining_budget < 1:
                raise ValueError(
                    f"`remaining_budget` must be >= 1, got {remaining_budget}."
                )
            horizon = remaining_budget
        else:
            horizon = max(1, self.budget - self._step_count)

        # Resolve current operating point x_curr of shape (1, d).
        if x_current is not None:
            if x_current.shape == (d,):
                x_curr = x_current.unsqueeze(0)
            elif x_current.shape == (1, d):
                x_curr = x_current
            else:
                raise ValueError(
                    f"`x_current` must have shape `({d},)` or `(1, {d})`, "
                    f"got {x_current.shape}."
                )
            x_curr = x_curr.to(device=bounds.device, dtype=bounds.dtype)
        else:
            x_curr = _extract_last_observed_point(model=model, bounds=bounds)
            if x_curr is None:
                if (
                    self._last_x_suggested is not None
                    and self._last_x_suggested.shape == (1, d)
                ):
                    x_curr = self._last_x_suggested.to(
                        device=bounds.device, dtype=bounds.dtype
                    )
                else:
                    x_curr = bounds.mean(dim=0, keepdim=True)

        can_use_cache = (
            self.replan_interval > 1
            and 0 < self._steps_since_replan < self.replan_interval
            and self._cached_path is not None
            and self._cached_path.shape[0]
            >= (horizon if remaining_budget is not None else 1)
            and self._cached_path.shape[-1] == d
        )
        if can_use_cache:
            assert self._cached_path is not None
            planned_path = self._cached_path.to(
                device=bounds.device, dtype=bounds.dtype
            )
            if remaining_budget is not None:
                planned_path = planned_path[:horizon]
            next_x = planned_path[:1].clone()
            self._cached_path = planned_path[1:].detach().clone()
            self._last_x_suggested = next_x.detach().clone()
            self._steps_since_replan += 1
            self._step_count += 1
            return next_x, planned_path

        warm_pts: Tensor | None = None
        if (
            self.warm_start
            and self._cached_path is not None
            and self._cached_path.shape[0] > 0
            and self._cached_path.shape[-1] == d
        ):
            warm_pts = self._cached_path.to(device=bounds.device, dtype=bounds.dtype)

        step_seed = None if self.seed is None else self.seed + self._step_count
        X_ts = sample_thompson_optima(
            model=model,
            bounds=bounds,
            budget=horizon,
            posterior_transform=self.posterior_transform,
            discrete_candidates=self.discrete_candidates,
            num_candidates=self.num_candidates,
            num_restarts=self.num_restarts,
            num_opt_steps=self.num_opt_steps,
            lr=self.lr,
            chebyshev_alpha=self.chebyshev_alpha,
            warm_start_points=warm_pts,
            seed=step_seed,
        )
        X_obs = _extract_observed_inputs(model=model, bounds=bounds)
        X_filtered = apply_epsilon_point_deletion(
            X_candidates=X_ts,
            bounds=bounds,
            X_observed=X_obs,
            epsilon=self.epsilon,
            model=model,
            seed=step_seed,
        )
        ordered_path, _ = solve_open_tsp_path(
            X_candidates=X_filtered,
            x_current=x_curr,
            model=model,
            transition_weights=self.transition_weights,
            transition_metric=self.transition_metric,
            exact_threshold=self.exact_tsp_threshold,
        )

        next_x = ordered_path[:1].clone()
        self._cached_path = ordered_path[1:].detach().clone()
        self._last_x_suggested = next_x.detach().clone()
        self._steps_since_replan = 1
        self._step_count += 1
        return next_x, ordered_path
