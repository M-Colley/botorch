#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
LineBO: Adaptive Bayesian Optimization in High Dimensions via One-Dimensional
Subspaces [kirschner2019linebo]_.

LineBO decomposes high-dimensional Bayesian optimization over
:math:`\mathcal{X} \subset \mathbb{R}^d` into a sequence of one-dimensional
acquisition optimization subproblems along iteratively selected affine lines
:math:`\mathcal{L}(x_t, v_t) = \{x_t + \alpha v_t : \alpha \in [\alpha_{\min},
\alpha_{\max}]\} \cap \mathcal{X}`.

This module provides:

1. ``compute_line_bounds``: Exact scalar interval intersection of a line through
   a pivot with box bounds (and optional local step cap ``max_step``).
2. ``LineAcquisitionFunction``: A 1D ``AcquisitionFunction`` wrapper around any
   single- or multi-objective BoTorch ``AcquisitionFunction`` with full autograd,
   ``_log`` propagation, and ``X_pending`` support.
3. Three line-selection strategies from [kirschner2019linebo]_:
   - ``"coordinate"`` (``CoordinateLineBO`` / ``sample_coordinate_direction``)
   - ``"random"`` (``RandomLineBO`` / ``sample_random_direction``)
   - ``"ascent"`` (``AscentLineBO`` / ``sample_ascent_direction``), supporting both
     Thompson-sampled derivative posterior directions via ``draw_matheron_paths``
     (with finite-difference joint posterior sampling fallback) and deterministic
     posterior mean gradients.
4. ``LineBO`` and ``optimize_linebo_step``: Stateful and functional helpers for
   running 1D subspace BO steps with either ``"best_posterior_mean"`` or
   ``"last_observed"`` pivot selection.

References

.. [kirschner2019linebo]
    J. Kirschner, M. Mutný, N. Hiller, R. Ischebeck, and A. Krause.
    Adaptive and Safe Bayesian Optimization in High Dimensions via
    One-Dimensional Subspaces. Proceedings of the 36th International
    Conference on Machine Learning (ICML), PMLR 97:3429-3438, 2019.

Contributor: M-Colley
"""

from __future__ import annotations

import warnings

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import (
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)
from botorch.exceptions import BotorchWarning, UnsupportedError
from botorch.models.model import Model, ModelList
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.pathwise.posterior_samplers import (
    draw_matheron_paths,
    MatheronPath,
)
from botorch.sampling.pathwise.utils import get_train_inputs
from torch import Tensor
from torch.nn import Module

__all__ = [
    "AscentLineBO",
    "CoordinateLineBO",
    "LineAcquisitionFunction",
    "LineBO",
    "RandomLineBO",
    "compute_line_bounds",
    "optimize_linebo_step",
    "sample_ascent_direction",
    "sample_coordinate_direction",
    "sample_random_direction",
    "select_linebo_pivot",
]

_DIRECTION_EPS = 1e-12
_BOUNDS_ATOL = 1e-6


def _validate_vector_1d(vec: Tensor, name: str) -> Tensor:
    r"""Validate and normalize shape of a ``(d,)`` or ``(1, d)`` vector to ``(d,)``.

    Args:
        vec: Input tensor expected to represent a single ``d``-dimensional vector.
        name: Argument name for error messages.

    Returns:
        A 1D tensor of shape ``(d,)``.
    """
    if vec.ndim == 2 and vec.shape[0] == 1:
        vec = vec.squeeze(0)
    if vec.ndim != 1 or vec.shape[0] < 1:
        raise ValueError(
            f"`{name}` must have shape `(d,)` (or `(1, d)`), got {tuple(vec.shape)}."
        )
    return vec


def _make_local_seed(
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> int | None:
    r"""Resolve an optional integer seed from ``seed`` or ``generator``.

    Args:
        generator: Optional PyTorch random number generator.
        seed: Optional explicit integer seed.

    Returns:
        An integer seed if either ``seed`` or ``generator`` is provided, else ``None``.
    """
    if seed is not None and generator is not None:
        raise ValueError("Specify at most one of `generator` or `seed`.")
    if seed is not None:
        return int(seed)
    if generator is not None:
        return int(
            torch.randint(
                0,
                2**31 - 1,
                (1,),
                generator=generator,
                device=generator.device,
            ).item()
        )
    return None


def _resolve_posterior_transform(
    model: Model,
    posterior_transform: PosteriorTransform | None,
    scalarization_weights: Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
) -> PosteriorTransform | None:
    r"""Build an effective single-outcome ``PosteriorTransform`` when needed.

    Args:
        model: The BoTorch ``Model``.
        posterior_transform: Optional explicit ``PosteriorTransform``.
        scalarization_weights: Optional 1D tensor of length ``model.num_outputs``.
        dtype: Target floating-point dtype.
        device: Target torch device.

    Returns:
        A ``PosteriorTransform`` yielding a scalarized single-output posterior, or
        ``None`` for single-output models without a transform.
    """
    if posterior_transform is not None and scalarization_weights is not None:
        raise ValueError(
            "Specify at most one of `posterior_transform` or `scalarization_weights`."
        )
    if scalarization_weights is not None:
        if scalarization_weights.ndim != 1:
            raise ValueError(
                "`scalarization_weights` must be a 1D tensor, got shape "
                f"{tuple(scalarization_weights.shape)}."
            )
        num_outputs = getattr(model, "num_outputs", 1)
        if scalarization_weights.shape[0] != num_outputs:
            raise ValueError(
                f"`scalarization_weights` length ({scalarization_weights.shape[0]}) "
                f"must match `model.num_outputs` ({num_outputs})."
            )
        return ScalarizedPosteriorTransform(
            weights=scalarization_weights.to(dtype=dtype, device=device)
        )
    if posterior_transform is not None:
        return posterior_transform
    num_outputs = getattr(model, "num_outputs", 1)
    if num_outputs > 1:
        weights = torch.full(
            (num_outputs,), 1.0 / num_outputs, dtype=dtype, device=device
        )
        return ScalarizedPosteriorTransform(weights=weights)
    return None


def _extract_train_X(
    model: Model | None = None,
    train_X: Tensor | None = None,
) -> Tensor:
    r"""Extract untransformed ``(n, d)`` observed inputs from ``train_X`` or ``model``.

    Args:
        model: Optional fitted BoTorch ``Model``.
        train_X: Optional explicit ``(n, d)`` tensor of observed design points.

    Returns:
        A 2D tensor of shape ``(n, d)`` in the original input space.
    """
    if train_X is not None:
        if train_X.ndim != 2 or train_X.shape[0] < 1:
            raise ValueError(
                f"`train_X` must have shape `(n, d)` with n >= 1, got "
                f"{tuple(train_X.shape)}."
            )
        return train_X.detach()

    if model is None:
        raise ValueError(
            "Either `pivot`, `train_X`, or an acquisition function with a fitted "
            "`model` must be provided."
        )

    try:
        raw_inputs = get_train_inputs(model, transformed=False)
    except Exception as exc:
        raise ValueError(
            "Could not extract training inputs from `model`. Please pass `pivot` or "
            "`train_X` explicitly."
        ) from exc

    if isinstance(model, ModelList):
        if not raw_inputs or not raw_inputs[0]:
            raise ValueError("`ModelList` does not contain any training inputs.")
        X = raw_inputs[0][0]
    else:
        if not raw_inputs:
            raise ValueError("`model` does not contain any training inputs.")
        X = raw_inputs[0]

    while X.ndim > 2:
        X = X[0]
    if X.ndim != 2 or X.shape[0] < 1:
        raise ValueError(
            f"Extracted training inputs must have shape `(n, d)` with n >= 1, got "
            f"{tuple(X.shape)}."
        )
    return X.detach()


def compute_line_bounds(
    pivot: Tensor,
    direction: Tensor,
    bounds: Tensor,
    max_step: float | None = None,
    atol: float = _BOUNDS_ATOL,
) -> Tensor:
    r"""Compute exact 1D box-intersection bounds for an affine line through ``pivot``.

    Computes the maximal scalar interval ``[alpha_min, alpha_max]`` such that
    ``bounds[0] <= pivot + alpha * direction <= bounds[1]`` for all
    ``alpha in [alpha_min, alpha_max]``, optionally intersected with the local
    step constraint ``[-max_step, max_step]``.

    Args:
        pivot: A ``(d,)`` (or ``(1, d)``) tensor giving the reference point on the
            line. Must lie inside ``bounds`` up to tolerance ``atol``.
        direction: A ``(d,)`` (or ``(1, d)``) non-zero direction vector (typically
            unit norm). Zero or near-zero components (``|direction_i| <= 1e-12``)
            are handled safely without division by zero.
        bounds: A ``2 x d`` tensor of lower (``bounds[0]``) and upper
            (``bounds[1]``) box bounds.
        max_step: Optional positive scalar capping ``|alpha| <= max_step`` for
            locally bounded / smooth HITL line sweeps.
        atol: Numerical tolerance used when validating ``bounds[0] <= pivot <=
            bounds[1]``.

    Returns:
        A ``2 x 1`` tensor ``[[alpha_min], [alpha_max]]`` compatible with
        ``botorch.optim.optimize_acqf``.
    """
    if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] < 1:
        raise ValueError(
            f"`bounds` must have shape `(2, d)` with d >= 1, got {tuple(bounds.shape)}."
        )
    if torch.any(bounds[0] > bounds[1]):
        raise ValueError(
            f"Lower bounds must not exceed upper bounds, got bounds={bounds}."
        )

    d = bounds.shape[1]
    pivot = _validate_vector_1d(pivot, "pivot").to(
        dtype=bounds.dtype, device=bounds.device
    )
    direction = _validate_vector_1d(direction, "direction").to(
        dtype=bounds.dtype, device=bounds.device
    )
    if pivot.shape[0] != d or direction.shape[0] != d:
        raise ValueError(
            f"Dimension mismatch: `pivot` ({pivot.shape[0]}), `direction` "
            f"({direction.shape[0]}), and `bounds` ({d}) must share dimension `d`."
        )

    dir_norm = torch.linalg.vector_norm(direction)
    if dir_norm <= _DIRECTION_EPS or not torch.isfinite(dir_norm):
        raise ValueError("`direction` must be finite and have non-zero norm.")

    if torch.any(pivot < bounds[0] - atol) or torch.any(pivot > bounds[1] + atol):
        raise ValueError(
            f"`pivot` must lie within `bounds` (up to tolerance {atol}), got "
            f"pivot={pivot} and bounds={bounds}."
        )

    pivot_clamped = pivot.clamp(min=bounds[0], max=bounds[1])
    active = direction.abs() > _DIRECTION_EPS
    if not torch.any(active):
        raise ValueError("`direction` has no active components above 1e-12.")

    safe_dir = torch.where(active, direction, torch.ones_like(direction))
    t_low = (bounds[0] - pivot_clamped) / safe_dir
    t_high = (bounds[1] - pivot_clamped) / safe_dir

    alpha_low = torch.where(
        active,
        torch.minimum(t_low, t_high),
        torch.full_like(t_low, -torch.inf),
    ).max()
    alpha_high = torch.where(
        active,
        torch.maximum(t_low, t_high),
        torch.full_like(t_high, torch.inf),
    ).min()

    if max_step is not None:
        if max_step <= 0:
            raise ValueError(f"`max_step` must be positive, got {max_step}.")
        step_cap = torch.as_tensor(max_step, dtype=bounds.dtype, device=bounds.device)
        alpha_low = torch.maximum(alpha_low, -step_cap)
        alpha_high = torch.minimum(alpha_high, step_cap)

    # Guard against sub-ulp numerical inversion when pivot sits on a box corner.
    if alpha_low > alpha_high and (alpha_low - alpha_high) <= atol:
        mid = 0.5 * (alpha_low + alpha_high)
        alpha_low = mid
        alpha_high = mid

    return torch.stack([alpha_low.unsqueeze(0), alpha_high.unsqueeze(0)], dim=0)


class LineAcquisitionFunction(AcquisitionFunction):
    r"""One-dimensional subspace wrapper around a ``d``-dimensional acquisition.

    Given a pivot :math:`x_0 \in \mathbb{R}^d` and normalized direction
    :math:`v \in \mathbb{S}^{d-1}`, evaluates a base acquisition function along
    the affine 1D subspace :math:`x(\alpha) = x_0 + \alpha v`, where
    :math:`\alpha` has shape ``batch_shape x q x 1``.
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        pivot: Tensor,
        direction: Tensor,
    ) -> None:
        r"""Initialize the 1D subspace acquisition wrapper.

        Args:
            acq_function: Base BoTorch ``AcquisitionFunction`` defined on ``d``-dim
                inputs ``batch_shape x q x d``.
            pivot: A ``(d,)`` (or ``(1, d)``) reference point :math:`x_0`.
            direction: A ``(d,)`` (or ``(1, d)``) non-zero direction vector, which
                is normalized to unit Euclidean norm internally.
        """
        Module.__init__(self)
        self.acq_function: AcquisitionFunction = acq_function
        self._log: bool = bool(getattr(acq_function, "_log", False))

        pivot_1d = _validate_vector_1d(pivot, "pivot").detach().clone()
        direction_1d = (
            _validate_vector_1d(direction, "direction")
            .to(dtype=pivot_1d.dtype, device=pivot_1d.device)
            .detach()
            .clone()
        )
        if pivot_1d.shape != direction_1d.shape:
            raise ValueError(
                f"`pivot` shape {tuple(pivot_1d.shape)} and `direction` shape "
                f"{tuple(direction_1d.shape)} must match."
            )
        norm = torch.linalg.vector_norm(direction_1d)
        if norm <= _DIRECTION_EPS or not torch.isfinite(norm):
            raise ValueError("`direction` must be finite and have non-zero norm.")
        direction_1d = direction_1d / norm

        self.d: int = pivot_1d.shape[0]
        self.register_buffer("pivot", pivot_1d)
        self.register_buffer("direction", direction_1d)

        base_pending = getattr(acq_function, "X_pending", None)
        self._base_X_pending: Tensor | None = (
            base_pending.detach().clone() if torch.is_tensor(base_pending) else None
        )
        self._line_X_pending: Tensor | None = None

    @property
    def model(self) -> Model | None:
        r"""Return the underlying model of ``self.acq_function`` if present."""
        return getattr(self.acq_function, "model", None)

    @property
    def X_pending(self) -> Tensor | None:
        r"""Return 1D pending coordinates ``alpha_pending`` on the line if set."""
        return self._line_X_pending

    @X_pending.setter
    def X_pending(self, X_pending: Tensor | None) -> None:
        r"""Set 1D pending coordinates and update the wrapped acquisition function."""
        self.set_X_pending(X_pending)

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        r"""Inform the acquisition function about pending 1D line coordinates.

        Args:
            X_pending: Optional ``n x 1`` tensor of pending scalar coordinates
                ``alpha`` along the active line (or ``None`` to clear line-level
                pending points while preserving any pre-existing full-space
                pending points on ``acq_function``).
        """
        if X_pending is None:
            self._line_X_pending = None
            self.acq_function.set_X_pending(self._base_X_pending)
            return

        if X_pending.ndim < 2 or X_pending.shape[-1] != 1:
            raise ValueError(
                "`X_pending` for `LineAcquisitionFunction` must have shape "
                f"`(... x n x 1)`, got {tuple(X_pending.shape)}."
            )
        if X_pending.requires_grad:
            warnings.warn(
                "Pending points require a gradient but the acquisition function "
                "will not provide a gradient to these points.",
                BotorchWarning,
                stacklevel=2,
            )
        alpha_pending = X_pending.detach().clone()
        self._line_X_pending = alpha_pending
        full_pending = self.project_to_full_space(alpha_pending)
        if self._base_X_pending is not None:
            base = self._base_X_pending.to(
                dtype=full_pending.dtype, device=full_pending.device
            )
            if full_pending.ndim > base.ndim:
                base = base.expand(*full_pending.shape[:-2], *base.shape[-2:])
            full_pending = torch.cat([base, full_pending], dim=-2)
        self.acq_function.set_X_pending(full_pending)

    def project_to_full_space(self, alpha: Tensor) -> Tensor:
        r"""Map scalar line coordinates ``alpha`` to ``d``-dimensional points.

        Args:
            alpha: Scalar tensor of shape ``(..., 1)`` or 0D scalar tensor.

        Returns:
            Full-space tensor ``pivot + alpha * direction`` of shape ``(..., d)``
            (or ``(d,)`` when ``alpha`` is 0D).
        """
        squeeze_out = alpha.ndim == 0
        if squeeze_out:
            alpha = alpha.unsqueeze(-1)
        elif alpha.shape[-1] != 1:
            raise ValueError(
                "Expected `alpha` to have trailing dimension 1, got shape "
                f"{tuple(alpha.shape)}."
            )
        pivot = self.pivot.to(dtype=alpha.dtype, device=alpha.device)
        direction = self.direction.to(dtype=alpha.dtype, device=alpha.device)
        X_full = pivot + alpha * direction
        return X_full.squeeze(0) if squeeze_out and X_full.ndim == 2 else X_full

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the wrapped acquisition function at ``pivot + X * direction``.

        Args:
            X: A ``batch_shape x q x 1`` tensor of scalar coordinates ``alpha``
                along the 1D line.

        Returns:
            A ``batch_shape``-dim tensor of acquisition values.
        """
        if X.ndim < 2 or X.shape[-1] != 1:
            raise ValueError(
                "`LineAcquisitionFunction` expects input `X` of shape "
                f"`batch_shape x q x 1`, got {tuple(X.shape)}."
            )
        X_full = self.project_to_full_space(X)
        return self.acq_function(X_full)


def sample_coordinate_direction(
    d: int,
    coordinate_index: int | None = None,
    generator: torch.Generator | None = None,
    seed: int | None = None,
    dtype: torch.dtype = torch.double,
    device: torch.device | None = None,
) -> Tensor:
    r"""Select a canonical coordinate basis vector :math:`e_i \in \mathbb{R}^d`.

    Implements the direction oracle of CoordinateLineBO [kirschner2019linebo]_,
    either cycling deterministically via ``coordinate_index % d`` or sampling
    uniformly from :math:`\{e_1, \dots, e_d\}`.

    Args:
        d: Ambient input dimension (``d >= 1``).
        coordinate_index: Optional non-negative / integer step index. When
            provided, selects coordinate ``coordinate_index % d`` deterministically.
            When ``None``, samples an index uniformly at random.
        generator: Optional ``torch.Generator`` for uniform random axis selection.
        seed: Optional integer seed (mutually exclusive with ``generator``).
        dtype: Output tensor floating-point dtype.
        device: Output tensor torch device.

    Returns:
        A unit basis vector of shape ``(d,)``.
    """
    if d < 1:
        raise ValueError(f"`d` must be >= 1, got {d}.")
    if coordinate_index is not None:
        if seed is not None or generator is not None:
            # Validate mutual exclusion even if deterministic index is supplied.
            _make_local_seed(generator=generator, seed=seed)
        idx = int(coordinate_index) % d
    else:
        if seed is not None and generator is not None:
            raise ValueError("Specify at most one of `generator` or `seed`.")
        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            idx = int(torch.randint(0, d, (1,), generator=gen, device="cpu").item())
        elif generator is not None:
            idx = int(
                torch.randint(
                    0, d, (1,), generator=generator, device=generator.device
                ).item()
            )
        else:
            idx = int(torch.randint(0, d, (1,)).item())

    direction = torch.zeros(d, dtype=dtype, device=device)
    direction[idx] = 1.0
    return direction


def sample_random_direction(
    d: int,
    generator: torch.Generator | None = None,
    seed: int | None = None,
    dtype: torch.dtype = torch.double,
    device: torch.device | None = None,
) -> Tensor:
    r"""Sample a direction uniformly from the unit sphere :math:`\mathbb{S}^{d-1}`.

    Implements the direction oracle of RandomLineBO [kirschner2019linebo]_ via
    normalized isotropic Gaussian draws :math:`z / \|z\|_2` with
    :math:`z \sim \mathcal{N}(0, I_d)`.

    Args:
        d: Ambient input dimension (``d >= 1``).
        generator: Optional ``torch.Generator`` for reproducible sampling.
        seed: Optional integer seed (mutually exclusive with ``generator``).
        dtype: Output tensor floating-point dtype.
        device: Output tensor torch device.

    Returns:
        A unit vector of shape ``(d,)``.
    """
    if d < 1:
        raise ValueError(f"`d` must be >= 1, got {d}.")
    if seed is not None and generator is not None:
        raise ValueError("Specify at most one of `generator` or `seed`.")

    local_gen = generator
    if seed is not None:
        local_gen = torch.Generator(device="cpu")
        local_gen.manual_seed(int(seed))

    for _ in range(16):
        if local_gen is not None:
            raw = torch.randn(
                d, generator=local_gen, dtype=dtype, device=local_gen.device
            ).to(device=device)
        else:
            raw = torch.randn(d, dtype=dtype, device=device)
        norm = torch.linalg.vector_norm(raw)
        if norm > _DIRECTION_EPS and torch.isfinite(norm):
            return raw / norm

    fallback = torch.zeros(d, dtype=dtype, device=device)
    fallback[0] = 1.0
    return fallback


def _sample_gradient_matheron(
    model: Model,
    pivot: Tensor,
    effective_transform: PosteriorTransform | None,
    local_seed: int | None,
) -> Tensor:
    r"""Draw a pathwise posterior sample via Matheron's rule and differentiate at pivot.

    Args:
        model: A fitted BoTorch GP model supported by ``draw_matheron_paths``.
        pivot: 1D tensor of shape ``(d,)``.
        effective_transform: Optional ``ScalarizedPosteriorTransform`` or ``None``.
        local_seed: Optional integer seed for reproducible path sampling.

    Returns:
        1D gradient tensor of shape ``(d,)`` at ``pivot``.
    """
    if isinstance(model, ModelList):
        raise UnsupportedError(
            "`ModelList` uses finite-difference joint posterior sampling."
        )
    if effective_transform is not None and not isinstance(
        effective_transform, ScalarizedPosteriorTransform
    ):
        raise UnsupportedError(
            "Non-scalarized `PosteriorTransform` uses finite-difference posterior "
            "sampling."
        )

    model.eval()
    with torch.no_grad():
        if local_seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(local_seed)
                paths = draw_matheron_paths(model, sample_shape=torch.Size([1]))
        else:
            paths = draw_matheron_paths(model, sample_shape=torch.Size([1]))

    if not isinstance(paths, MatheronPath):
        raise UnsupportedError(f"Expected `MatheronPath`, got {type(paths).__name__}.")

    with torch.enable_grad():
        x = pivot.detach().clone().unsqueeze(0).requires_grad_(True)
        num_outputs = getattr(model, "num_outputs", 1)
        if num_outputs == 1:
            y = paths(x).unsqueeze(-1)
        else:
            y = paths(x.unsqueeze(-3)).transpose(-1, -2)
        if effective_transform is not None:
            y = effective_transform.evaluate(Y=y, X=x)
        scalar_val = y.mean()
        (grad,) = torch.autograd.grad(scalar_val, x)
    return grad.squeeze(0).detach()


def _sample_gradient_finite_difference(
    model: Model,
    pivot: Tensor,
    effective_transform: PosteriorTransform | None,
    fd_step: float | None,
    local_seed: int | None,
) -> Tensor:
    r"""Sample a local gradient via central finite differences of a joint GP draw.

    Evaluates ``model.posterior`` jointly on the ``2 * d`` stencil points
    ``pivot +/- h * e_i`` so that ``rsample`` preserves cross-point posterior
    covariances across each central-difference pair.

    Args:
        model: A fitted BoTorch ``Model``.
        pivot: 1D tensor of shape ``(d,)``.
        effective_transform: Optional ``PosteriorTransform``.
        fd_step: Optional finite-difference step ``h > 0``.
        local_seed: Optional integer seed for reproducible posterior sampling.

    Returns:
        1D gradient tensor of shape ``(d,)`` at ``pivot``.
    """
    d = pivot.shape[0]
    if fd_step is not None:
        if fd_step <= 0:
            raise ValueError(f"`fd_step` must be positive, got {fd_step}.")
        h = float(fd_step)
    else:
        h = 1e-3 if pivot.dtype == torch.double else 1e-2

    eye = torch.eye(d, dtype=pivot.dtype, device=pivot.device)
    X_fd = torch.cat([pivot.unsqueeze(0) + h * eye, pivot.unsqueeze(0) - h * eye], 0)

    with torch.no_grad():
        posterior = model.posterior(X_fd, posterior_transform=effective_transform)
        if local_seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(local_seed)
                samples = posterior.rsample(sample_shape=torch.Size([1]))
        else:
            samples = posterior.rsample(sample_shape=torch.Size([1]))

    samples_q = samples.mean(dim=-1)
    while samples_q.ndim > 1:
        samples_q = samples_q.mean(dim=0)
    return (samples_q[:d] - samples_q[d:]) / (2.0 * h)


def sample_ascent_direction(
    model: Model,
    pivot: Tensor,
    sample_gradient: bool = True,
    posterior_transform: PosteriorTransform | None = None,
    scalarization_weights: Tensor | None = None,
    maximize: bool = True,
    fd_step: float | None = None,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> Tensor:
    r"""Compute a normalized GP ascent (or descent) direction at ``pivot``.

    Implements the local gradient direction oracle of AscentLineBO / DescentLineBO
    (Section 3.2 of [kirschner2019linebo]_):

    - When ``sample_gradient=True`` (default): draws a Thompson-sampled gradient
      from the GP derivative posterior by differentiating a posterior sample path
      drawn via ``draw_matheron_paths(model, sample_shape=torch.Size([1]))``,
      automatically falling back to central finite differences of a joint posterior
      sample ``model.posterior(pivot +/- h * e_i).rsample()`` when pathwise
      sampling is unsupported (e.g., ``ModelListGP``).
    - When ``sample_gradient=False``: computes the exact deterministic gradient of
      the GP posterior mean :math:`\nabla_x \mu(\text{pivot})`.

    If the resulting gradient norm is below ``1e-12``, falls back to a uniform
    random unit direction on :math:`\mathbb{S}^{d-1}`.

    Args:
        model: A fitted BoTorch ``Model``.
        pivot: A ``(d,)`` (or ``(1, d)``) tensor giving the point at which to
            evaluate the gradient.
        sample_gradient: If ``True``, Thompson-samples a gradient from the GP
            derivative posterior; if ``False``, uses the posterior mean gradient.
        posterior_transform: Optional ``PosteriorTransform`` mapping multi-output
            posteriors to a scalarized single-output posterior.
        scalarization_weights: Optional 1D tensor of weights of length
            ``model.num_outputs`` for linear scalarization of multi-output models
            (mutually exclusive with ``posterior_transform``).
        maximize: If ``True``, returns an ascent direction; if ``False``, negates
            the gradient to return a descent direction.
        fd_step: Optional positive step size ``h`` for the finite-difference
            fallback when ``sample_gradient=True``.
        generator: Optional ``torch.Generator`` for reproducible sampling.
        seed: Optional integer seed (mutually exclusive with ``generator``).

    Returns:
        A unit direction vector of shape ``(d,)``.
    """
    pivot_1d = _validate_vector_1d(pivot, "pivot")
    d = pivot_1d.shape[0]
    dtype = pivot_1d.dtype
    device = pivot_1d.device

    local_seed = _make_local_seed(generator=generator, seed=seed)
    effective_transform = _resolve_posterior_transform(
        model=model,
        posterior_transform=posterior_transform,
        scalarization_weights=scalarization_weights,
        dtype=dtype,
        device=device,
    )

    if sample_gradient:
        try:
            grad = _sample_gradient_matheron(
                model=model,
                pivot=pivot_1d,
                effective_transform=effective_transform,
                local_seed=local_seed,
            )
        except (
            NotImplementedError,
            UnsupportedError,
            TypeError,
            RuntimeError,
            AttributeError,
        ):
            grad = _sample_gradient_finite_difference(
                model=model,
                pivot=pivot_1d,
                effective_transform=effective_transform,
                fd_step=fd_step,
                local_seed=local_seed,
            )
    else:
        with torch.enable_grad():
            x = pivot_1d.detach().clone().unsqueeze(0).requires_grad_(True)
            posterior = model.posterior(x, posterior_transform=effective_transform)
            scalar_mean = posterior.mean.mean()
            (grad_x,) = torch.autograd.grad(scalar_mean, x)
            grad = grad_x.squeeze(0).detach()

    if not maximize:
        grad = -grad

    grad_norm = torch.linalg.vector_norm(grad)
    if grad_norm < _DIRECTION_EPS or not torch.isfinite(grad_norm):
        return sample_random_direction(
            d=d,
            seed=local_seed,
            dtype=dtype,
            device=device,
        )
    return grad / grad_norm


def select_linebo_pivot(
    model: Model | None = None,
    train_X: Tensor | None = None,
    pivot_selection: str = "best_posterior_mean",
    posterior_transform: PosteriorTransform | None = None,
    scalarization_weights: Tensor | None = None,
    maximize: bool = True,
    bounds: Tensor | None = None,
) -> Tensor:
    r"""Select the pivot point :math:`x_t` through which the next 1D line passes.

    Args:
        model: Fitted BoTorch ``Model`` (required for ``"best_posterior_mean"``,
            and used to extract ``train_X`` when ``train_X is None``).
        train_X: Optional explicit ``(n, d)`` tensor of observed inputs.
        pivot_selection: Strategy for choosing the pivot from observed points:
            - ``"best_posterior_mean"`` (default in [kirschner2019linebo]_):
              selects :math:`\arg\max_{x \in X_{\text{train}}} \mu(x)` (or
              :math:`\arg\min` when ``maximize=False``).
            - ``"last_observed"``: selects ``train_X[-1]``, producing connected
              1D line sweeps for smooth HITL / physical parameter transitions.
        posterior_transform: Optional ``PosteriorTransform`` for multi-output models.
        scalarization_weights: Optional 1D tensor of weights for multi-output models.
        maximize: Whether higher posterior mean values are preferred.
        bounds: Optional ``2 x d`` box bounds used to clamp sub-tolerance numerical
            roundoff on extracted training points.

    Returns:
        Selected pivot tensor of shape ``(d,)``.
    """
    X_obs = _extract_train_X(model=model, train_X=train_X)
    if bounds is not None:
        X_obs = X_obs.to(dtype=bounds.dtype, device=bounds.device)

    mode = pivot_selection.lower()
    if mode == "last_observed":
        pivot = X_obs[-1].clone()
    elif mode == "best_posterior_mean":
        if model is None:
            raise ValueError(
                "`model` is required when `pivot_selection='best_posterior_mean'`."
            )
        effective_transform = _resolve_posterior_transform(
            model=model,
            posterior_transform=posterior_transform,
            scalarization_weights=scalarization_weights,
            dtype=X_obs.dtype,
            device=X_obs.device,
        )
        with torch.no_grad():
            posterior = model.posterior(X_obs, posterior_transform=effective_transform)
            scores = posterior.mean.mean(dim=-1)
            while scores.ndim > 1:
                scores = scores.mean(dim=0)
            if not maximize:
                scores = -scores
            best_idx = int(torch.argmax(scores).item())
        pivot = X_obs[best_idx].clone()
    else:
        raise ValueError(
            f"Unknown `pivot_selection={pivot_selection!r}`. Expected "
            "`'best_posterior_mean'` or `'last_observed'`."
        )

    if bounds is not None:
        if torch.any(pivot < bounds[0] - _BOUNDS_ATOL) or torch.any(
            pivot > bounds[1] + _BOUNDS_ATOL
        ):
            raise ValueError(f"Selected pivot {pivot} lies outside `bounds` {bounds}.")
        pivot = pivot.clamp(min=bounds[0], max=bounds[1])
    return pivot


def optimize_linebo_step(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    strategy: str = "ascent",
    pivot: Tensor | None = None,
    pivot_selection: str = "best_posterior_mean",
    direction: Tensor | None = None,
    q: int = 1,
    max_step: float | None = None,
    coordinate_index: int | None = None,
    sample_gradient: bool = True,
    posterior_transform: PosteriorTransform | None = None,
    scalarization_weights: Tensor | None = None,
    maximize: bool | None = None,
    fd_step: float | None = None,
    num_restarts: int = 8,
    raw_samples: int = 128,
    sequential: bool = False,
    options: dict[str, bool | float | int | str] | None = None,
    train_X: Tensor | None = None,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Execute a single 1D subspace LineBO step for any BoTorch acquisition function.

    Selects (or validates) a pivot :math:`x_t` and unit direction :math:`v_t`,
    computes the exact 1D line-box intersection ``[alpha_min, alpha_max]``, and
    optimizes ``LineAcquisitionFunction(acq_function, pivot, direction)`` over
    1D via ``optimize_acqf``.

    Args:
        acq_function: A ``d``-dimensional single- or multi-objective BoTorch
            ``AcquisitionFunction``.
        bounds: A ``2 x d`` tensor of lower and upper box bounds.
        strategy: Line direction selection strategy when ``direction is None``:
            ``"ascent"`` (default), ``"random"``, or ``"coordinate"``.
        pivot: Optional ``(d,)`` pivot point inside ``bounds``. When ``None``,
            inferred from observed training data according to ``pivot_selection``.
        pivot_selection: Pivot rule used when ``pivot is None``:
            ``"best_posterior_mean"`` (default) or ``"last_observed"``.
        direction: Optional explicit ``(d,)`` non-zero direction vector (normalized
            internally). Overrides ``strategy`` when provided.
        q: Number of candidates to generate along the active 1D line (``q >= 1``).
        max_step: Optional positive scalar capping ``|alpha| <= max_step``.
        coordinate_index: Optional axis index for ``strategy="coordinate"``.
        sample_gradient: Whether ``strategy="ascent"`` Thompson-samples from the GP
            derivative posterior (``True``) or uses the posterior mean gradient
            (``False``).
        posterior_transform: Optional ``PosteriorTransform`` for multi-output models.
            Defaults to ``acq_function.posterior_transform`` when available.
        scalarization_weights: Optional 1D tensor of weights for multi-output models.
        maximize: Whether the surrogate objective is maximized (defaults to
            ``getattr(acq_function, "maximize", True)``).
        fd_step: Optional finite-difference step size for ``strategy="ascent"``
            fallback sampling.
        num_restarts: Number of L-BFGS-B restarts in ``optimize_acqf``.
        raw_samples: Number of initial Sobol samples in 1D for ``optimize_acqf``.
        sequential: Whether to optimize ``q > 1`` candidates sequentially along the
            active 1D line.
        options: Optional optimizer options passed to ``optimize_acqf``.
        train_X: Optional explicit ``(n, d)`` observed inputs used when ``pivot``
            is ``None``.
        generator: Optional ``torch.Generator`` for direction sampling.
        seed: Optional integer seed (mutually exclusive with ``generator``).

    Returns:
        A 3-tuple ``(candidate_X, acq_value, direction)`` where:
        - ``candidate_X`` has shape ``(q, d)`` on the line ``pivot + alpha * direction``
          within ``bounds``,
        - ``acq_value`` is the optimal acquisition value tensor, and
        - ``direction`` is the normalized ``(d,)`` unit direction vector.
    """
    if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] < 1:
        raise ValueError(
            f"`bounds` must have shape `(2, d)` with d >= 1, got {tuple(bounds.shape)}."
        )
    if q < 1:
        raise ValueError(f"`q` must be >= 1, got {q}.")

    d = bounds.shape[1]
    dtype = bounds.dtype
    device = bounds.device
    model = getattr(acq_function, "model", None)

    effective_maximize = (
        bool(getattr(acq_function, "maximize", True))
        if maximize is None
        else bool(maximize)
    )
    inferred_transform = (
        posterior_transform
        if posterior_transform is not None
        else (
            None
            if scalarization_weights is not None
            else getattr(acq_function, "posterior_transform", None)
        )
    )

    if pivot is None:
        pivot_1d = select_linebo_pivot(
            model=model,
            train_X=train_X,
            pivot_selection=pivot_selection,
            posterior_transform=inferred_transform,
            scalarization_weights=scalarization_weights,
            maximize=effective_maximize,
            bounds=bounds,
        )
    else:
        pivot_1d = _validate_vector_1d(pivot, "pivot").to(dtype=dtype, device=device)

    if direction is not None:
        dir_1d = _validate_vector_1d(direction, "direction").to(
            dtype=dtype, device=device
        )
        dir_norm = torch.linalg.vector_norm(dir_1d)
        if dir_norm <= _DIRECTION_EPS or not torch.isfinite(dir_norm):
            raise ValueError("`direction` must be finite and have non-zero norm.")
        unit_direction = dir_1d / dir_norm
    else:
        strat = strategy.lower()
        if strat == "coordinate":
            unit_direction = sample_coordinate_direction(
                d=d,
                coordinate_index=coordinate_index,
                generator=generator,
                seed=seed,
                dtype=dtype,
                device=device,
            )
        elif strat == "random":
            unit_direction = sample_random_direction(
                d=d,
                generator=generator,
                seed=seed,
                dtype=dtype,
                device=device,
            )
        elif strat == "ascent":
            if model is None:
                raise ValueError(
                    "`strategy='ascent'` requires `acq_function` to expose a fitted "
                    "`model`."
                )
            unit_direction = sample_ascent_direction(
                model=model,
                pivot=pivot_1d,
                sample_gradient=sample_gradient,
                posterior_transform=inferred_transform,
                scalarization_weights=scalarization_weights,
                maximize=effective_maximize,
                fd_step=fd_step,
                generator=generator,
                seed=seed,
            )
        else:
            raise ValueError(
                f"Unknown `strategy={strategy!r}`. Expected `'coordinate'`, "
                "`'random'`, or `'ascent'`."
            )

    line_bounds = compute_line_bounds(
        pivot=pivot_1d,
        direction=unit_direction,
        bounds=bounds,
        max_step=max_step,
    )
    line_acqf = LineAcquisitionFunction(
        acq_function=acq_function,
        pivot=pivot_1d,
        direction=unit_direction,
    )

    interval_width = float((line_bounds[1, 0] - line_bounds[0, 0]).item())
    if interval_width <= _DIRECTION_EPS:
        alpha_opt = line_bounds[0].expand(q, 1).clone()
        with torch.no_grad():
            acq_val = line_acqf(alpha_opt.unsqueeze(0)).squeeze(0)
            if sequential and q > 1:
                acq_val = acq_val.expand(q).clone()
    else:
        alpha_opt, acq_val_opt = optimize_acqf(
            acq_function=line_acqf,
            bounds=line_bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=sequential,
            options=options,
        )
        if acq_val_opt is None:  # pragma: no cover
            with torch.no_grad():
                acq_val = line_acqf(alpha_opt.unsqueeze(0)).squeeze(0)
        else:
            acq_val = acq_val_opt

    candidate_X = line_acqf.project_to_full_space(alpha_opt)
    candidate_X = candidate_X.clamp(min=bounds[0], max=bounds[1])
    return candidate_X, acq_val, line_acqf.direction.detach().clone()


class LineBO:
    r"""Stateful driver for iterative 1D subspace Bayesian optimization (LineBO).

    Tracks step count, coordinate cycling state, random generator state, and the
    most recent active 1D line ``(last_pivot, last_direction, last_line_bounds)``
    across optimization iterations [kirschner2019linebo]_.
    """

    def __init__(
        self,
        bounds: Tensor,
        strategy: str = "ascent",
        pivot_selection: str = "best_posterior_mean",
        max_step: float | None = None,
        cycle_coordinates: bool = True,
        sample_gradient: bool = True,
        posterior_transform: PosteriorTransform | None = None,
        scalarization_weights: Tensor | None = None,
        maximize: bool | None = None,
        fd_step: float | None = None,
        num_restarts: int = 8,
        raw_samples: int = 128,
        options: dict[str, bool | float | int | str] | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        r"""Initialize a stateful ``LineBO`` optimizer.

        Args:
            bounds: A ``2 x d`` tensor of lower and upper box bounds.
            strategy: Direction selection strategy: ``"ascent"`` (default),
                ``"random"``, or ``"coordinate"``.
            pivot_selection: Default pivot selection rule when ``pivot`` is omitted
                in ``step``: ``"best_posterior_mean"`` or ``"last_observed"``.
            max_step: Optional positive scalar capping ``|alpha| <= max_step``.
            cycle_coordinates: If ``True`` and ``strategy="coordinate"``, cycles
                deterministically through ``step_count % d``; if ``False``, samples
                coordinate axes uniformly at random.
            sample_gradient: If ``True`` and ``strategy="ascent"``, Thompson-samples
                directions from the GP derivative posterior; if ``False``, uses the
                posterior mean gradient.
            posterior_transform: Optional ``PosteriorTransform`` for multi-output
                models.
            scalarization_weights: Optional 1D scalarization weights for
                multi-output models.
            maximize: Optional boolean overriding ``acq_function.maximize``.
            fd_step: Optional finite-difference step size for ``strategy="ascent"``
                fallback sampling.
            num_restarts: Number of restarts passed to ``optimize_acqf``.
            raw_samples: Number of raw 1D Sobol samples passed to ``optimize_acqf``.
            options: Optional optimizer options dict passed to ``optimize_acqf``.
            generator: Optional ``torch.Generator`` for reproducible direction
                draws across steps.
            seed: Optional integer seed used to initialize an internal
                ``torch.Generator`` (mutually exclusive with ``generator``).
        """
        if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] < 1:
            raise ValueError(
                "`bounds` must have shape `(2, d)` with d >= 1, got "
                f"{tuple(bounds.shape)}."
            )
        if torch.any(bounds[0] > bounds[1]):
            raise ValueError(
                f"Lower bounds must not exceed upper bounds, got bounds={bounds}."
            )
        strat = strategy.lower()
        if strat not in {"coordinate", "random", "ascent"}:
            raise ValueError(
                f"Unknown `strategy={strategy!r}`. Expected `'coordinate'`, "
                "`'random'`, or `'ascent'`."
            )
        if pivot_selection.lower() not in {"best_posterior_mean", "last_observed"}:
            raise ValueError(
                f"Unknown `pivot_selection={pivot_selection!r}`. Expected "
                "`'best_posterior_mean'` or `'last_observed'`."
            )
        if max_step is not None and max_step <= 0:
            raise ValueError(f"`max_step` must be positive, got {max_step}.")
        if seed is not None and generator is not None:
            raise ValueError("Specify at most one of `generator` or `seed`.")

        self.bounds: Tensor = bounds.detach().clone()
        self.d: int = bounds.shape[1]
        self.strategy: str = strat
        self.pivot_selection: str = pivot_selection.lower()
        self.max_step: float | None = max_step
        self.cycle_coordinates: bool = cycle_coordinates
        self.sample_gradient: bool = sample_gradient
        self.posterior_transform: PosteriorTransform | None = posterior_transform
        self.scalarization_weights: Tensor | None = scalarization_weights
        self.maximize: bool | None = maximize
        self.fd_step: float | None = fd_step
        self.num_restarts: int = num_restarts
        self.raw_samples: int = raw_samples
        self.options: dict[str, bool | float | int | str] | None = options

        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            self.generator: torch.Generator | None = gen
        else:
            self.generator = generator

        self.step_count: int = 0
        self.last_pivot: Tensor | None = None
        self.last_direction: Tensor | None = None
        self.last_line_bounds: Tensor | None = None

    def step(
        self,
        acq_function: AcquisitionFunction,
        pivot: Tensor | None = None,
        direction: Tensor | None = None,
        q: int = 1,
        sequential: bool = False,
        train_X: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Perform one LineBO iteration and advance internal step state.

        Args:
            acq_function: The ``d``-dimensional BoTorch ``AcquisitionFunction``.
            pivot: Optional explicit ``(d,)`` pivot overriding ``pivot_selection``.
            direction: Optional explicit ``(d,)`` direction overriding ``strategy``.
            q: Number of candidates to generate on the active 1D line.
            sequential: Whether to optimize ``q > 1`` candidates sequentially.
            train_X: Optional explicit ``(n, d)`` observed inputs when ``pivot`` is
                ``None``.

        Returns:
            Tuple ``(candidate_X, acq_value, direction)`` for the current step.
        """
        effective_maximize = (
            bool(getattr(acq_function, "maximize", True))
            if self.maximize is None
            else bool(self.maximize)
        )
        inferred_transform = (
            self.posterior_transform
            if self.posterior_transform is not None
            else (
                None
                if self.scalarization_weights is not None
                else getattr(acq_function, "posterior_transform", None)
            )
        )
        model = getattr(acq_function, "model", None)

        if pivot is None:
            resolved_pivot = select_linebo_pivot(
                model=model,
                train_X=train_X,
                pivot_selection=self.pivot_selection,
                posterior_transform=inferred_transform,
                scalarization_weights=self.scalarization_weights,
                maximize=effective_maximize,
                bounds=self.bounds,
            )
        else:
            resolved_pivot = _validate_vector_1d(pivot, "pivot").to(
                dtype=self.bounds.dtype, device=self.bounds.device
            )

        coord_idx = (
            self.step_count
            if (self.strategy == "coordinate" and self.cycle_coordinates)
            else None
        )

        candidate_X, acq_val, unit_direction = optimize_linebo_step(
            acq_function=acq_function,
            bounds=self.bounds,
            strategy=self.strategy,
            pivot=resolved_pivot,
            pivot_selection=self.pivot_selection,
            direction=direction,
            q=q,
            max_step=self.max_step,
            coordinate_index=coord_idx,
            sample_gradient=self.sample_gradient,
            posterior_transform=self.posterior_transform,
            scalarization_weights=self.scalarization_weights,
            maximize=self.maximize,
            fd_step=self.fd_step,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            sequential=sequential,
            options=self.options,
            train_X=train_X,
            generator=None if coord_idx is not None else self.generator,
        )

        self.last_pivot = resolved_pivot.detach().clone()
        self.last_direction = unit_direction.detach().clone()
        self.last_line_bounds = compute_line_bounds(
            pivot=self.last_pivot,
            direction=self.last_direction,
            bounds=self.bounds,
            max_step=self.max_step,
        )
        self.step_count += 1
        return candidate_X, acq_val, unit_direction


class CoordinateLineBO(LineBO):
    r"""Coordinate-aligned LineBO variant from [kirschner2019linebo]_."""

    def __init__(
        self,
        bounds: Tensor,
        pivot_selection: str = "best_posterior_mean",
        max_step: float | None = None,
        cycle_coordinates: bool = True,
        posterior_transform: PosteriorTransform | None = None,
        scalarization_weights: Tensor | None = None,
        maximize: bool | None = None,
        num_restarts: int = 8,
        raw_samples: int = 128,
        options: dict[str, bool | float | int | str] | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        r"""Initialize ``CoordinateLineBO``.

        Args:
            bounds: A ``2 x d`` tensor of lower and upper box bounds.
            pivot_selection: Pivot selection strategy (``"best_posterior_mean"`` or
                ``"last_observed"``).
            max_step: Optional positive scalar capping ``|alpha| <= max_step``.
            cycle_coordinates: If ``True``, cycles through canonical basis vectors
                ``e_{(t mod d) + 1}``; if ``False``, samples axes uniformly.
            posterior_transform: Optional ``PosteriorTransform`` for pivot scoring.
            scalarization_weights: Optional 1D weights for multi-output pivot
                scoring.
            maximize: Optional boolean overriding ``acq_function.maximize``.
            num_restarts: Number of restarts passed to ``optimize_acqf``.
            raw_samples: Number of raw 1D Sobol samples passed to ``optimize_acqf``.
            options: Optional optimizer options dict passed to ``optimize_acqf``.
            generator: Optional ``torch.Generator`` when ``cycle_coordinates=False``.
            seed: Optional integer seed (mutually exclusive with ``generator``).
        """
        super().__init__(
            bounds=bounds,
            strategy="coordinate",
            pivot_selection=pivot_selection,
            max_step=max_step,
            cycle_coordinates=cycle_coordinates,
            posterior_transform=posterior_transform,
            scalarization_weights=scalarization_weights,
            maximize=maximize,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
            generator=generator,
            seed=seed,
        )


class RandomLineBO(LineBO):
    r"""Uniformly random spherical-direction LineBO variant [kirschner2019linebo]_."""

    def __init__(
        self,
        bounds: Tensor,
        pivot_selection: str = "best_posterior_mean",
        max_step: float | None = None,
        posterior_transform: PosteriorTransform | None = None,
        scalarization_weights: Tensor | None = None,
        maximize: bool | None = None,
        num_restarts: int = 8,
        raw_samples: int = 128,
        options: dict[str, bool | float | int | str] | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        r"""Initialize ``RandomLineBO``.

        Args:
            bounds: A ``2 x d`` tensor of lower and upper box bounds.
            pivot_selection: Pivot selection strategy (``"best_posterior_mean"`` or
                ``"last_observed"``).
            max_step: Optional positive scalar capping ``|alpha| <= max_step``.
            posterior_transform: Optional ``PosteriorTransform`` for pivot scoring.
            scalarization_weights: Optional 1D weights for multi-output pivot
                scoring.
            maximize: Optional boolean overriding ``acq_function.maximize``.
            num_restarts: Number of restarts passed to ``optimize_acqf``.
            raw_samples: Number of raw 1D Sobol samples passed to ``optimize_acqf``.
            options: Optional optimizer options dict passed to ``optimize_acqf``.
            generator: Optional ``torch.Generator`` for spherical direction draws.
            seed: Optional integer seed (mutually exclusive with ``generator``).
        """
        super().__init__(
            bounds=bounds,
            strategy="random",
            pivot_selection=pivot_selection,
            max_step=max_step,
            posterior_transform=posterior_transform,
            scalarization_weights=scalarization_weights,
            maximize=maximize,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
            generator=generator,
            seed=seed,
        )


class AscentLineBO(LineBO):
    r"""GP gradient-ascent LineBO variant from [kirschner2019linebo]_."""

    def __init__(
        self,
        bounds: Tensor,
        pivot_selection: str = "best_posterior_mean",
        max_step: float | None = None,
        sample_gradient: bool = True,
        posterior_transform: PosteriorTransform | None = None,
        scalarization_weights: Tensor | None = None,
        maximize: bool | None = None,
        fd_step: float | None = None,
        num_restarts: int = 8,
        raw_samples: int = 128,
        options: dict[str, bool | float | int | str] | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        r"""Initialize ``AscentLineBO``.

        Args:
            bounds: A ``2 x d`` tensor of lower and upper box bounds.
            pivot_selection: Pivot selection strategy (``"best_posterior_mean"`` or
                ``"last_observed"``).
            max_step: Optional positive scalar capping ``|alpha| <= max_step``.
            sample_gradient: If ``True``, Thompson-samples a gradient from the GP
                derivative posterior; if ``False``, uses the posterior mean
                gradient.
            posterior_transform: Optional ``PosteriorTransform`` for multi-output
                models.
            scalarization_weights: Optional 1D weights for multi-output models.
            maximize: Optional boolean overriding ``acq_function.maximize``.
            fd_step: Optional finite-difference step size for fallback derivative
                sampling.
            num_restarts: Number of restarts passed to ``optimize_acqf``.
            raw_samples: Number of raw 1D Sobol samples passed to ``optimize_acqf``.
            options: Optional optimizer options dict passed to ``optimize_acqf``.
            generator: Optional ``torch.Generator`` for derivative path sampling.
            seed: Optional integer seed (mutually exclusive with ``generator``).
        """
        super().__init__(
            bounds=bounds,
            strategy="ascent",
            pivot_selection=pivot_selection,
            max_step=max_step,
            sample_gradient=sample_gradient,
            posterior_transform=posterior_transform,
            scalarization_weights=scalarization_weights,
            maximize=maximize,
            fd_step=fd_step,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
            generator=generator,
            seed=seed,
        )
