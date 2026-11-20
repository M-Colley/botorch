#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Safe Exploration for Bayesian Optimization with Gaussian Processes (SafeOpt and
SafeOpt-MC). See [sui2015safeopt]_ and [berkenkamp2016safeopt]_.

SafeOpt maintains a high-probability safe subset ``S_t`` of a discretized candidate
domain ``X`` and selects queries that maximize epistemic uncertainty across the union
of potential maximizers ``M_t`` and potential safe-set expanders ``G_t``. Both the
Lipschitz-based formulation of [sui2015safeopt]_ and the pure Gaussian process
confidence-bound / rank-1 fantasy expansion formulation of [berkenkamp2016safeopt]_
(SafeOpt-MC with multiple decoupled or coupled safety constraints) are supported.

References

.. [sui2015safeopt]
    Y. Sui, A. Gotovos, J. W. Burdick, and A. Krause.
    Safe Exploration for Optimization with Gaussian Processes.
    Proceedings of the 32nd International Conference on Machine Learning (ICML),
    PMLR 37:997-1005, 2015.
.. [berkenkamp2016safeopt]
    F. Berkenkamp, A. P. Schoellig, and A. Krause.
    Safe Controller Optimization for Quadrotors with Gaussian Processes.
    Proceedings of the IEEE International Conference on Robotics and Automation
    (ICRA), pp. 491-496, 2016.

Contributor: M-Colley
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model, ModelList
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from botorch.posteriors.posterior_list import PosteriorList
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


@dataclass
class SafeOptSets:
    r"""Container for SafeOpt candidate subsets and confidence bounds.

    Attributes:
        safe_mask: Boolean tensor of shape ``(M,)`` indicating candidates in the
            certified safe set ``S_t``.
        maximizers_mask: Boolean tensor of shape ``(M,)`` indicating potential
            maximizers ``M_t \subseteq S_t``.
        expanders_mask: Boolean tensor of shape ``(M,)`` indicating potential
            safe-set expanders ``G_t \subseteq S_t``.
        lower: Lower confidence bounds ``l_t(x)`` of shape ``(M, m)``.
        upper: Upper confidence bounds ``u_t(x)`` of shape ``(M, m)``.
    """

    safe_mask: Tensor
    maximizers_mask: Tensor
    expanders_mask: Tensor
    lower: Tensor
    upper: Tensor


def _extract_all_output_covariances(
    posterior: Posterior,
    num_candidates: int,
) -> Tensor:
    r"""Extract per-output covariance matrices of shape ``(m, M, M)``."""
    if isinstance(posterior, PosteriorList):
        sub_covs = [
            _extract_all_output_covariances(p, num_candidates=num_candidates)
            for p in posterior.posteriors
        ]
        return torch.cat(sub_covs, dim=0)

    if isinstance(posterior, GPyTorchPosterior):
        mvn = posterior.distribution
        if not posterior._is_mt:
            return mvn.covariance_matrix.unsqueeze(0)
        num_outputs = posterior.mean.shape[-1]
        cov_full = mvn.covariance_matrix
        if getattr(mvn, "_interleaved", True):
            cov_4d = cov_full.reshape(
                num_candidates, num_outputs, num_candidates, num_outputs
            )
            return cov_4d.diagonal(dim1=1, dim2=3).permute(2, 0, 1).contiguous()
        cov_4d = cov_full.reshape(
            num_outputs, num_candidates, num_outputs, num_candidates
        )
        return cov_4d.diagonal(dim1=0, dim2=2).permute(2, 0, 1).contiguous()

    if hasattr(posterior, "distribution") and hasattr(
        posterior.distribution, "covariance_matrix"
    ):
        cov = posterior.distribution.covariance_matrix
        if cov.shape == (num_candidates, num_candidates):
            return cov.unsqueeze(0)

    raise NotImplementedError(
        f"Extracting full covariance matrices from posterior of type "
        f"{type(posterior).__name__} is not supported when `lipschitz=None`."
    )


class SafeOpt(AcquisitionFunction):
    r"""SafeOpt and SafeOpt-MC acquisition function over a discrete candidate set.

    Supports single-output safe optimization [sui2015safeopt]_ where the objective
    ``f(x)`` satisfies ``f(x) >= h_0``, as well as multi-output SafeOpt-MC
    [berkenkamp2016safeopt]_ where output ``0`` is maximized subject to safety lower
    bounds on either all ``m`` outputs (``len(safety_thresholds) == m``) or on the
    ``m - 1`` constraint outputs ``1, ..., m - 1``
    (``len(safety_thresholds) == m - 1``).
    """

    def __init__(
        self,
        model: Model,
        candidate_set: Tensor,
        safety_thresholds: float | Sequence[float] | Tensor,
        initial_safe_indices: Sequence[int] | Tensor | None = None,
        safe_seed: Tensor | None = None,
        beta: float | Tensor | Callable[[int], float | Tensor] = 2.0,
        lipschitz: float | Sequence[float] | Tensor | None = None,
        monotonic_bounds: bool = False,
        scaling: float | Sequence[float] | Tensor | str | None = None,
        x_current: Tensor | None = None,
        max_step: float | None = None,
        step: int = 1,
        prev_safe_mask: Tensor | None = None,
        prev_lower: Tensor | None = None,
        prev_upper: Tensor | None = None,
        candidate_match_atol: float = 1e-6,
    ) -> None:
        r"""Initialize SafeOpt / SafeOpt-MC.

        Args:
            model: A fitted BoTorch ``Model`` with ``m >= 1`` outputs. Output ``0``
                is the objective to maximize.
            candidate_set: A ``(M, d)`` tensor of discretized candidate points ``X``.
            safety_thresholds: Safety lower bound(s) ``h_k``. For ``m == 1``, a
                scalar or length-1 sequence/tensor constraining output ``0``. For
                ``m >= 2``, a sequence/tensor of length ``m`` (constraining all
                outputs ``0, ..., m - 1``) or length ``m - 1`` (constraining outputs
                ``1, ..., m - 1`` while leaving objective output ``0`` unconstrained;
                a scalar is also accepted when ``m == 2``).
            initial_safe_indices: Optional 1D integer index sequence/tensor or
                ``(M,)`` boolean mask specifying the initial safe seed ``S_0``.
                Exactly one of ``initial_safe_indices`` or ``safe_seed`` must be
                provided.
            safe_seed: Optional ``(k, d)`` or ``(d,)`` tensor of known-safe seed
                points mapped onto ``candidate_set`` to form ``S_0``.
            beta: Confidence interval scaling ``\beta_t > 0`` (or a callable mapping
                the 1-based ``step`` ``t`` to ``\beta_t``) such that
                ``Q_t(x) = [\mu_t(x) \pm \sqrt{\beta_t} \sigma_t(x)]``.
            lipschitz: Optional positive Lipschitz constant(s) ``L_k`` (scalar,
                length ``len(constrained_dims)``, or length ``m``). If provided, uses
                the Lipschitz safe-set and expander definitions of [sui2015safeopt]_.
                If ``None``, uses GP confidence bounds and exact rank-1 Gaussian
                fantasy expansion as in [berkenkamp2016safeopt]_.
            monotonic_bounds: If ``True``, maintains intersected confidence bounds
                ``C_t(x) = C_{t-1}(x) \cap Q_t(x)`` across steps.
            scaling: Optional positive per-output width normalization ``scale_k``
                (scalar, length-``m`` sequence/tensor, ``"auto"``, or ``None`` for
                unit scaling) in ``w_t(x) = \max_k (u_{t,k}(x) - l_{t,k}(x)) /
                scale_k``.
            x_current: Optional ``(d,)`` or ``(1, d)`` tensor representing the
                current safe parameter for local step-size limiting.
            max_step: Optional positive Euclidean radius restricting the next query
                to ``||x - x_current||_2 <= max_step``.
            step: Positive 1-based iteration counter ``t`` passed to callable
                ``beta``.
            prev_safe_mask: Optional ``(M,)`` boolean mask overriding ``S_{t-1}``.
            prev_lower: Optional ``(M, m)`` tensor overriding ``l_{t-1}(x)`` when
                ``monotonic_bounds=True``.
            prev_upper: Optional ``(M, m)`` tensor overriding ``u_{t-1}(x)`` when
                ``monotonic_bounds=True``.
            candidate_match_atol: Non-negative Euclidean tolerance used to match
                arbitrary query tensors ``X`` to ``candidate_set`` in ``forward``.
        """
        super().__init__(model=model)
        if candidate_set.ndim != 2 or candidate_set.shape[0] == 0:
            raise ValueError(
                "`candidate_set` must be a non-empty 2D tensor of shape `(M, d)`, "
                f"got shape {tuple(candidate_set.shape)}."
            )
        if candidate_set.shape[1] == 0:
            raise ValueError("`candidate_set` must have feature dimension `d >= 1`.")
        if step < 1:
            raise ValueError(f"`step` must be >= 1, got {step}.")
        if candidate_match_atol < 0:
            raise ValueError(
                f"`candidate_match_atol` must be >= 0, got {candidate_match_atol}."
            )

        self.register_buffer("candidate_set", candidate_set.clone())
        self._num_candidates: int = int(candidate_set.shape[0])
        self._dim: int = int(candidate_set.shape[1])
        self.num_outputs: int = int(model.num_outputs)
        self._step: int = int(step)
        self.beta = beta
        self.monotonic_bounds: bool = bool(monotonic_bounds)
        self.candidate_match_atol: float = float(candidate_match_atol)

        device = candidate_set.device
        dtype = candidate_set.dtype

        thresholds, constrained_dims = self._parse_safety_thresholds(
            safety_thresholds=safety_thresholds,
            num_outputs=self.num_outputs,
            device=device,
            dtype=dtype,
        )
        self.register_buffer("safety_thresholds", thresholds)
        self._constrained_dims: list[int] = constrained_dims

        # Validate beta at the initial step.
        self._get_sqrt_beta(self._step)

        initial_mask = self._parse_initial_safe_set(
            initial_safe_indices=initial_safe_indices,
            safe_seed=safe_seed,
        )
        self.register_buffer("_initial_safe_mask", initial_mask)

        if prev_safe_mask is not None:
            if prev_safe_mask.dtype != torch.bool or prev_safe_mask.shape != (
                self._num_candidates,
            ):
                raise ValueError(
                    "`prev_safe_mask` must be a boolean tensor of shape "
                    f"({self._num_candidates},)."
                )
            if not prev_safe_mask.any():
                raise ValueError(
                    "`prev_safe_mask` must contain at least one safe point."
                )
            init_prev_safe = prev_safe_mask.to(device=device) | initial_mask
        else:
            init_prev_safe = initial_mask.clone()
        self.register_buffer("_prev_safe_mask", init_prev_safe)

        lipschitz_tensor = self._parse_lipschitz(
            lipschitz=lipschitz,
            constrained_dims=self._constrained_dims,
            num_outputs=self.num_outputs,
            device=device,
            dtype=dtype,
        )
        if lipschitz_tensor is not None:
            self.register_buffer("lipschitz", lipschitz_tensor)
            pairwise_dists = torch.cdist(
                self.candidate_set,
                self.candidate_set,
                p=2.0,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            self.register_buffer("_pairwise_dists", pairwise_dists)
        else:
            self.lipschitz = None
            self._pairwise_dists = None

        self._scaling_spec = scaling
        scale_tensor = self._resolve_scaling(
            scaling=scaling,
            model=model,
            num_outputs=self.num_outputs,
            device=device,
            dtype=dtype,
        )
        self.register_buffer("scaling", scale_tensor)

        if max_step is not None and max_step <= 0:
            raise ValueError(f"`max_step` must be > 0 when provided, got {max_step}.")
        self.max_step: float | None = float(max_step) if max_step is not None else None
        self.x_current: Tensor | None = (
            self._validate_x_current(x_current) if x_current is not None else None
        )

        # Initialize C_0(x): [h_k, +inf) on S_0 for constrained outputs, (-inf, +inf)
        # elsewhere, unless explicitly overridden by prev_lower / prev_upper.
        c0_lower = torch.full(
            (self._num_candidates, self.num_outputs),
            -torch.inf,
            device=device,
            dtype=dtype,
        )
        c0_lower[
            initial_mask.unsqueeze(-1)
            & self._constrained_output_mask(device=device).unsqueeze(0)
        ] = thresholds.repeat(int(initial_mask.sum().item()))
        c0_upper = torch.full(
            (self._num_candidates, self.num_outputs),
            torch.inf,
            device=device,
            dtype=dtype,
        )
        if prev_lower is not None:
            if prev_lower.shape != (self._num_candidates, self.num_outputs):
                raise ValueError(
                    "`prev_lower` must have shape "
                    f"{(self._num_candidates, self.num_outputs)}, "
                    f"got {tuple(prev_lower.shape)}."
                )
            c0_lower = prev_lower.to(device=device, dtype=dtype)
        if prev_upper is not None:
            if prev_upper.shape != (self._num_candidates, self.num_outputs):
                raise ValueError(
                    "`prev_upper` must have shape "
                    f"{(self._num_candidates, self.num_outputs)}, "
                    f"got {tuple(prev_upper.shape)}."
                )
            c0_upper = prev_upper.to(device=device, dtype=dtype)

        self.register_buffer("_prev_lower", c0_lower)
        self.register_buffer("_prev_upper", c0_upper)
        self._last_sets: SafeOptSets | None = None

    @property
    def step(self) -> int:
        r"""Current 1-based optimization step ``t``."""
        return self._step

    @property
    def constrained_dims(self) -> list[int]:
        r"""Indices of safety-constrained model outputs."""
        return list(self._constrained_dims)

    @property
    def initial_safe_mask(self) -> Tensor:
        r"""Boolean mask of shape ``(M,)`` for the initial safe seed ``S_0``."""
        return self._initial_safe_mask

    @property
    def safe_set(self) -> Tensor:
        r"""Certified safe points ``S_t`` as a ``(|S_t|, d)`` tensor."""
        return self.candidate_set[self.compute_sets().safe_mask]

    @property
    def maximizers(self) -> Tensor:
        r"""Potential maximizers ``M_t`` as a ``(|M_t|, d)`` tensor."""
        return self.candidate_set[self.compute_sets().maximizers_mask]

    @property
    def expanders(self) -> Tensor:
        r"""Potential expanders ``G_t`` as a ``(|G_t|, d)`` tensor."""
        return self.candidate_set[self.compute_sets().expanders_mask]

    def _constrained_output_mask(self, device: torch.device) -> Tensor:
        mask = torch.zeros(self.num_outputs, dtype=torch.bool, device=device)
        mask[self._constrained_dims] = True
        return mask

    @staticmethod
    def _parse_safety_thresholds(
        safety_thresholds: float | Sequence[float] | Tensor,
        num_outputs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, list[int]]:
        if isinstance(safety_thresholds, (int, float)):
            if num_outputs == 1:
                return (
                    torch.tensor(
                        [float(safety_thresholds)], device=device, dtype=dtype
                    ),
                    [0],
                )
            if num_outputs == 2:
                return (
                    torch.tensor(
                        [float(safety_thresholds)], device=device, dtype=dtype
                    ),
                    [1],
                )
            raise ValueError(
                "Scalar `safety_thresholds` is ambiguous when `model.num_outputs > 2`. "
                f"Expected length {num_outputs - 1} or {num_outputs}."
            )

        thresh = torch.as_tensor(safety_thresholds, device=device, dtype=dtype)
        if thresh.ndim == 0:
            return SafeOpt._parse_safety_thresholds(
                safety_thresholds=float(thresh.item()),
                num_outputs=num_outputs,
                device=device,
                dtype=dtype,
            )
        if thresh.ndim != 1:
            raise ValueError(
                "`safety_thresholds` must be a scalar or 1D sequence/tensor, "
                f"got shape {tuple(thresh.shape)}."
            )

        n_thresh = int(thresh.numel())
        if num_outputs == 1:
            if n_thresh != 1:
                raise ValueError(
                    "Single-output models require exactly 1 safety threshold, "
                    f"got {n_thresh}."
                )
            return thresh, [0]

        if n_thresh == num_outputs:
            return thresh, list(range(num_outputs))
        if n_thresh == num_outputs - 1:
            return thresh, list(range(1, num_outputs))
        raise ValueError(
            f"For a model with {num_outputs} outputs, `safety_thresholds` must have "
            f"length {num_outputs} (all outputs constrained) or {num_outputs - 1} "
            f"(outputs 1..{num_outputs - 1} constrained), got {n_thresh}."
        )

    def _parse_initial_safe_set(
        self,
        initial_safe_indices: Sequence[int] | Tensor | None,
        safe_seed: Tensor | None,
    ) -> Tensor:
        if (initial_safe_indices is None) == (safe_seed is None):
            raise ValueError(
                "Exactly one of `initial_safe_indices` or `safe_seed` must be provided."
            )
        device = self.candidate_set.device
        dtype = self.candidate_set.dtype
        mask = torch.zeros(self._num_candidates, dtype=torch.bool, device=device)

        if initial_safe_indices is not None:
            if isinstance(initial_safe_indices, Tensor) and (
                initial_safe_indices.dtype == torch.bool
            ):
                if initial_safe_indices.shape != (self._num_candidates,):
                    raise ValueError(
                        "Boolean `initial_safe_indices` must have shape "
                        f"({self._num_candidates},), got "
                        f"{tuple(initial_safe_indices.shape)}."
                    )
                mask = initial_safe_indices.to(device=device).clone()
            else:
                idx = torch.as_tensor(
                    initial_safe_indices, device=device, dtype=torch.long
                )
                if idx.ndim != 1 or idx.numel() == 0:
                    raise ValueError(
                        "`initial_safe_indices` must be a non-empty 1D index sequence "
                        "or boolean mask."
                    )
                if (idx < 0).any() or (idx >= self._num_candidates).any():
                    raise ValueError(
                        "`initial_safe_indices` contains out-of-bounds indices for "
                        f"`candidate_set` of size {self._num_candidates}."
                    )
                mask[idx] = True
        else:
            assert safe_seed is not None
            seed = safe_seed.to(device=device, dtype=dtype)
            if seed.ndim == 1:
                seed = seed.unsqueeze(0)
            if seed.ndim != 2 or seed.shape[0] == 0 or seed.shape[1] != self._dim:
                raise ValueError(
                    f"`safe_seed` must have shape `(k, {self._dim})` or "
                    f"`({self._dim},)` with `k >= 1`, got {tuple(safe_seed.shape)}."
                )
            dists = torch.cdist(
                seed,
                self.candidate_set,
                p=2.0,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            exact_matches = dists <= self.candidate_match_atol
            has_exact = exact_matches.any(dim=-1)
            mask |= exact_matches.any(dim=0)
            if not has_exact.all():
                nearest_idx = dists[~has_exact].argmin(dim=-1)
                mask[nearest_idx] = True

        if not mask.any():
            raise ValueError("Initial safe set `S_0` must be non-empty.")
        return mask

    @staticmethod
    def _parse_lipschitz(
        lipschitz: float | Sequence[float] | Tensor | None,
        constrained_dims: list[int],
        num_outputs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if lipschitz is None:
            return None
        num_c = len(constrained_dims)
        lip = torch.as_tensor(lipschitz, device=device, dtype=dtype)
        if lip.ndim == 0:
            lip = lip.expand(num_c).clone()
        elif lip.ndim == 1:
            if lip.numel() == 1:
                lip = lip.expand(num_c).clone()
            elif lip.numel() == num_c:
                lip = lip.clone()
            elif lip.numel() == num_outputs:
                lip = lip[constrained_dims].clone()
            else:
                raise ValueError(
                    f"`lipschitz` must be scalar or have length {num_c} or "
                    f"{num_outputs}, got length {lip.numel()}."
                )
        else:
            raise ValueError(
                "`lipschitz` must be a scalar or 1D tensor/sequence, "
                f"got shape {tuple(lip.shape)}."
            )
        if (lip <= 0).any():
            raise ValueError("`lipschitz` constants must be strictly positive.")
        return lip

    @staticmethod
    def _extract_auto_scale_single(
        model: Model,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        m = int(model.num_outputs)
        if hasattr(model, "outcome_transform") and hasattr(
            model.outcome_transform, "stdvs"
        ):
            stdvs = model.outcome_transform.stdvs.detach().to(
                device=device, dtype=dtype
            )
            if stdvs.numel() == m:
                return stdvs.reshape(m).clamp_min(1e-8)
        if hasattr(model, "covar_module") and hasattr(
            model.covar_module, "outputscale"
        ):
            os = model.covar_module.outputscale.detach().to(device=device, dtype=dtype)
            if os.numel() == m:
                return os.reshape(m).clamp_min(1e-16).sqrt()
            if os.numel() == 1:
                return os.reshape(1).clamp_min(1e-16).sqrt().expand(m).clone()
        return torch.ones(m, device=device, dtype=dtype)

    @classmethod
    def _resolve_scaling(
        cls,
        scaling: float | Sequence[float] | Tensor | str | None,
        model: Model,
        num_outputs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if scaling is None:
            return torch.ones(num_outputs, device=device, dtype=dtype)
        if isinstance(scaling, str):
            if scaling != "auto":
                raise ValueError(
                    f"Unsupported string `scaling={scaling!r}`. Expected `'auto'`."
                )
            if isinstance(model, ModelList):
                parts = [
                    cls._extract_auto_scale_single(sub, device=device, dtype=dtype)
                    for sub in model.models
                ]
                return torch.cat(parts, dim=0)
            return cls._extract_auto_scale_single(model, device=device, dtype=dtype)

        sc = torch.as_tensor(scaling, device=device, dtype=dtype)
        if sc.ndim == 0:
            sc = sc.expand(num_outputs).clone()
        elif sc.ndim == 1:
            if sc.numel() == 1:
                sc = sc.expand(num_outputs).clone()
            elif sc.numel() != num_outputs:
                raise ValueError(
                    f"`scaling` must be scalar or have length {num_outputs}, "
                    f"got length {sc.numel()}."
                )
        else:
            raise ValueError(
                "`scaling` must be a scalar, 1D sequence/tensor, `'auto'`, or `None`."
            )
        if (sc <= 0).any():
            raise ValueError("`scaling` values must be strictly positive.")
        return sc

    def _validate_x_current(self, x_current: Tensor) -> Tensor:
        x_curr = x_current.to(
            device=self.candidate_set.device, dtype=self.candidate_set.dtype
        )
        if x_curr.ndim == 1 and x_curr.shape[0] == self._dim:
            return x_curr.unsqueeze(0)
        if x_curr.ndim == 2 and x_curr.shape == (1, self._dim):
            return x_curr
        raise ValueError(
            f"`x_current` must have shape `({self._dim},)` or `(1, {self._dim})`, "
            f"got {tuple(x_current.shape)}."
        )

    def _get_sqrt_beta(self, step: int) -> Tensor:
        raw_beta = self.beta(step) if callable(self.beta) else self.beta
        b = torch.as_tensor(
            raw_beta,
            device=self.candidate_set.device,
            dtype=self.candidate_set.dtype,
        )
        if b.ndim > 1 or b.numel() not in (1, self.num_outputs):
            raise ValueError(
                f"`beta` must be a scalar or length-{self.num_outputs} tensor, "
                f"got shape {tuple(b.shape)}."
            )
        if (b <= 0).any():
            raise ValueError("`beta` must be strictly positive.")
        return b.reshape(1, -1).sqrt()

    def compute_uncertainty(self, lower: Tensor, upper: Tensor) -> Tensor:
        r"""Compute scaled uncertainty widths ``w_t(x)`` across all outputs.

        Args:
            lower: Lower confidence bounds of shape ``(M, m)``.
            upper: Upper confidence bounds of shape ``(M, m)``.

        Returns:
            A ``(M,)`` tensor of normalized widths
            ``\max_{k} (u_{t,k}(x) - l_{t,k}(x)) / scale_k``.
        """
        return ((upper - lower) / self.scaling.unsqueeze(0)).max(dim=-1).values

    def compute_sets(self) -> SafeOptSets:
        r"""Compute ``S_t``, ``M_t``, ``G_t``, and confidence bounds ``(l_t, u_t)``.

        Returns:
            A ``SafeOptSets`` instance containing boolean masks ``safe_mask``,
            ``maximizers_mask``, and ``expanders_mask`` of shape ``(M,)``, along
            with ``lower`` and ``upper`` confidence bounds of shape ``(M, m)``.
        """
        device = self.candidate_set.device
        dtype = self.candidate_set.dtype
        sqrt_beta = self._get_sqrt_beta(self._step)

        with torch.no_grad():
            posterior = self.model.posterior(
                self.candidate_set, observation_noise=False
            )
            mean = posterior.mean.reshape(self._num_candidates, self.num_outputs)
            var = posterior.variance.reshape(
                self._num_candidates, self.num_outputs
            ).clamp_min(0.0)
            std = var.sqrt()

            q_lower = mean - sqrt_beta * std
            q_upper = mean + sqrt_beta * std

            if self.monotonic_bounds:
                lower = torch.maximum(self._prev_lower, q_lower)
                upper = torch.minimum(self._prev_upper, q_upper)
                # Guard against inverted intervals under noisy outlier observations.
                upper = torch.maximum(upper, lower)
            else:
                lower = q_lower
                upper = q_upper

            c_dims = self._constrained_dims
            thresholds = self.safety_thresholds

            # 1. Certified Safe Set S_t
            if self.lipschitz is not None:
                assert self._pairwise_dists is not None
                prev_idx = self._prev_safe_mask.nonzero(as_tuple=True)[0]
                dists_from_prev = self._pairwise_dists[prev_idx]
                l_prev = lower[prev_idx][:, c_dims]
                lip = self.lipschitz
                certifies = (
                    l_prev[:, None, :]
                    - dists_from_prev[:, :, None] * lip[None, None, :]
                ) >= thresholds[None, None, :]
                newly_safe = certifies.all(dim=-1).any(dim=0)
                safe_mask = self._initial_safe_mask | newly_safe
            else:
                gp_safe = (lower[:, c_dims] >= thresholds.unsqueeze(0)).all(dim=-1)
                safe_mask = self._initial_safe_mask | gp_safe

            # 2. Potential Maximizers M_t (objective is always output 0)
            best_safe_lower = lower[safe_mask, 0].max()
            maximizers_mask = safe_mask & (upper[:, 0] >= best_safe_lower)

            # 3. Potential Expanders G_t
            unsafe_mask = ~safe_mask
            expanders_mask = torch.zeros_like(safe_mask)
            if unsafe_mask.any():
                safe_idx = safe_mask.nonzero(as_tuple=True)[0]
                unsafe_idx = unsafe_mask.nonzero(as_tuple=True)[0]

                if self.lipschitz is not None:
                    assert self._pairwise_dists is not None
                    dists_su = self._pairwise_dists[safe_idx][:, unsafe_idx]
                    u_safe = upper[safe_idx][:, c_dims]
                    lip = self.lipschitz
                    opt_cert = (
                        u_safe[:, None, :] - dists_su[:, :, None] * lip[None, None, :]
                    ) >= thresholds[None, None, :]
                    expanders_mask[safe_idx] = opt_cert.all(dim=-1).any(dim=-1)
                else:
                    noisy_posterior = self.model.posterior(
                        self.candidate_set, observation_noise=True
                    )
                    noisy_var = noisy_posterior.variance.reshape(
                        self._num_candidates, self.num_outputs
                    )
                    cov_all = _extract_all_output_covariances(
                        posterior=posterior,
                        num_candidates=self._num_candidates,
                    )
                    cov_c = cov_all[c_dims]
                    cov_su = cov_c[:, safe_idx, :][:, :, unsafe_idx]

                    min_var = 1e-12 if dtype == torch.float32 else 1e-18
                    denom_s = (
                        noisy_var[safe_idx][:, c_dims]
                        .transpose(0, 1)
                        .clamp_min(min_var)
                    )
                    mu_s = mean[safe_idx][:, c_dims].transpose(0, 1)
                    u_s = upper[safe_idx][:, c_dims].transpose(0, 1)
                    mu_u = mean[unsafe_idx][:, c_dims].transpose(0, 1)
                    var_u = var[unsafe_idx][:, c_dims].transpose(0, 1)

                    gain = cov_su / denom_s.unsqueeze(-1)
                    cond_mean = mu_u.unsqueeze(1) + gain * (u_s - mu_s).unsqueeze(-1)
                    cond_var = (var_u.unsqueeze(1) - gain * cov_su).clamp_min(0.0)

                    if sqrt_beta.numel() == 1:
                        sb_c = sqrt_beta.reshape(1, 1, 1)
                    else:
                        sb_c = sqrt_beta[0, c_dims].reshape(len(c_dims), 1, 1)

                    cond_lower = cond_mean - sb_c * cond_var.sqrt()
                    cert = cond_lower >= thresholds.reshape(len(c_dims), 1, 1)
                    expanders_mask[safe_idx] = cert.all(dim=0).any(dim=-1)

        sets = SafeOptSets(
            safe_mask=safe_mask.to(device=device),
            maximizers_mask=maximizers_mask.to(device=device),
            expanders_mask=expanders_mask.to(device=device),
            lower=lower,
            upper=upper,
        )
        self._last_sets = sets
        return sets

    def _eligible_candidate_mask(
        self,
        sets: SafeOptSets,
        x_current: Tensor | None = None,
        max_step: float | None = None,
    ) -> Tensor:
        eff_max_step = self.max_step if max_step is None else float(max_step)
        if eff_max_step is not None and eff_max_step <= 0:
            raise ValueError(f"`max_step` must be > 0, got {eff_max_step}.")

        eff_x_current = (
            self._validate_x_current(x_current)
            if x_current is not None
            else self.x_current
        )
        active_mask = sets.maximizers_mask | sets.expanders_mask

        if eff_max_step is None:
            return active_mask if active_mask.any() else sets.safe_mask

        if eff_x_current is None:
            raise ValueError("`x_current` must be provided when `max_step` is set.")

        step_dists = torch.linalg.vector_norm(
            self.candidate_set - eff_x_current, ord=2, dim=-1
        )
        within_step = step_dists <= eff_max_step
        eligible = active_mask & within_step
        if eligible.any():
            return eligible

        fallback = sets.safe_mask & within_step
        if fallback.any():
            return fallback

        raise RuntimeError(
            "No certified safe candidate lies within `max_step` of `x_current`."
        )

    def next_point(
        self,
        x_current: Tensor | None = None,
        max_step: float | None = None,
    ) -> Tensor:
        r"""Select the next safe query point maximizing scaled uncertainty width.

        Evaluates ``\arg\max_{x \in M_t \cup G_t} w_t(x)``, optionally restricted to
        ``||x - x_current||_2 <= max_step`` (falling back to safe candidates in
        ``S_t`` within ``max_step`` if ``(M_t \cup G_t)`` has no reachable point).

        Args:
            x_current: Optional ``(d,)`` or ``(1, d)`` tensor overriding
                ``self.x_current`` for this step.
            max_step: Optional positive float overriding ``self.max_step`` for this
                step.

        Returns:
            A ``(1, d)`` tensor containing the selected safe candidate.
        """
        sets = self.compute_sets()
        eligible_mask = self._eligible_candidate_mask(
            sets=sets, x_current=x_current, max_step=max_step
        )
        widths = self.compute_uncertainty(lower=sets.lower, upper=sets.upper)
        eligible_idx = eligible_mask.nonzero(as_tuple=True)[0]
        best_idx = eligible_idx[widths[eligible_idx].argmax()]
        return self.candidate_set[best_idx : best_idx + 1]

    def get_best_safe_point(self) -> Tensor:
        r"""Return the pessimistic safe optimum ``\arg\max_{x \in S_t} l_{t,0}(x)``.

        Returns:
            A ``(1, d)`` tensor containing the safe candidate with maximum objective
            lower confidence bound.
        """
        sets = self.compute_sets()
        safe_idx = sets.safe_mask.nonzero(as_tuple=True)[0]
        best_idx = safe_idx[sets.lower[safe_idx, 0].argmax()]
        return self.candidate_set[best_idx : best_idx + 1]

    def update(
        self,
        model: Model,
        x_current: Tensor | None = None,
    ) -> SafeOpt:
        r"""Advance SafeOpt to step ``t + 1`` with an updated GP model.

        Propagates ``S_t`` into ``S_{t-1}`` (and ``C_t`` into ``C_{t-1}`` when
        ``monotonic_bounds=True``), updates ``self.model``, increments ``self.step``,
        and refreshes ``scaling`` when ``scaling="auto"``.

        Args:
            model: Updated BoTorch ``Model`` with ``self.num_outputs`` outputs.
            x_current: Optional ``(d,)`` or ``(1, d)`` tensor updating
                ``self.x_current``.

        Returns:
            ``self`` updated to step ``t + 1``.
        """
        if int(model.num_outputs) != self.num_outputs:
            raise ValueError(
                f"Updated model has {model.num_outputs} outputs, expected "
                f"{self.num_outputs}."
            )
        prev_sets = (
            self._last_sets if self._last_sets is not None else self.compute_sets()
        )
        self._prev_safe_mask.copy_(prev_sets.safe_mask | self._initial_safe_mask)
        if self.monotonic_bounds:
            self._prev_lower.copy_(prev_sets.lower)
            self._prev_upper.copy_(prev_sets.upper)

        self.model = model
        self._step += 1
        self._get_sqrt_beta(self._step)
        if self._scaling_spec == "auto":
            self.scaling.copy_(
                self._resolve_scaling(
                    scaling="auto",
                    model=model,
                    num_outputs=self.num_outputs,
                    device=self.candidate_set.device,
                    dtype=self.candidate_set.dtype,
                )
            )
        if x_current is not None:
            self.x_current = self._validate_x_current(x_current)
        self._last_sets = None
        return self

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate SafeOpt acquisition values on ``X``.

        Points matching an eligible candidate in ``M_t \cup G_t`` (subject to
        ``max_step`` if configured) receive their normalized uncertainty width
        ``w_t(x) >= 0``; all other points receive ``-inf``.

        Args:
            X: A ``batch_shape x 1 x d`` tensor of query points.

        Returns:
            A ``batch_shape`` tensor of SafeOpt acquisition values.
        """
        sets = self.compute_sets()
        eligible_mask = self._eligible_candidate_mask(sets=sets)
        widths = self.compute_uncertainty(lower=sets.lower, upper=sets.upper)
        cand_scores = torch.full_like(widths, -torch.inf)
        cand_scores[eligible_mask] = widths[eligible_mask]

        batch_shape = X.shape[:-2]
        X_flat = X.squeeze(-2).reshape(-1, self._dim)
        dists = torch.cdist(
            X_flat,
            self.candidate_set,
            p=2.0,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        min_dists, nearest_idx = dists.min(dim=-1)
        matched = min_dists <= self.candidate_match_atol

        out = torch.full(
            (X_flat.shape[0],),
            -torch.inf,
            device=X.device,
            dtype=X.dtype,
        )
        out[matched] = cand_scores[nearest_idx[matched]].to(
            device=X.device, dtype=X.dtype
        )
        return out.reshape(batch_shape)
