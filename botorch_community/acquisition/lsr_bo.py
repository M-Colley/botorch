#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Local Search Region Constrained Bayesian Optimization (LSR-BO) and Multi-Step
Lookahead Rollout under Local Search Constraints. See [paulson2022lsrlookahead]_
and [paulson2023lsrbo]_.

In online optimization of physical and control systems (such as chemical
processes or vapor compression cycles), abrupt jumps between consecutive inputs
can violate transient safety constraints or trigger hardware fail-safes. At
iteration ``k`` with current operating point ``x_k``, the next query ``x_{k+1}``
is restricted to a local search region intersected with the global domain:

.. math::

    \mathcal{X}_k = \mathcal{T}(x_k) \cap \Theta, \quad
    \mathcal{T}(x_k) = \left\{
        x \in \mathbb{R}^d : \left\| \frac{x - x_k}{\delta} \right\|_p \le 1
    \right\}.

This module implements:
1. Local search region geometry utilities (``compute_local_box_bounds``,
   ``make_local_region_constraint``, ``project_to_local_region``, and
   ``gen_local_region_initial_conditions``).
2. The LSR-BO global-local exploration policy (``optimize_lsr_bo_step`` and
   ``LSRBO``) from [paulson2023lsrbo]_, which escapes suboptimal local basins by
   comparing local and global acquisition improvements and projecting toward the
   global target whenever local utility falls below a fraction ``gamma`` of
   global utility.
3. Approximate dynamic programming (ADP) multi-step rollout lookahead under
   local search constraints (``LocalSearchRolloutAcquisitionFunction`` and
   ``GaussHermiteNormalSampler``) from [paulson2022lsrlookahead]_.

References

.. [paulson2022lsrlookahead]
    J. A. Paulson, F. Sorourifar, and A. Chakrabarty.
    Efficient Multi-Step Lookahead Bayesian Optimization with Local Search
    Constraints. In Proceedings of the 2022 IEEE 61st Conference on Decision
    and Control (CDC), pages 123-129, December 2022.

.. [paulson2023lsrbo]
    J. A. Paulson, F. Sorourifar, C. R. Laughman, and A. Chakrabarty.
    LSR-BO: Local Search Region Constrained Bayesian Optimization for
    Performance Optimization of Vapor Compression Systems. In Proceedings of
    the 2023 American Control Conference (ACC), pages 576-582, May 2023.

Contributor: M-Colley
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from botorch import settings
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.acquisition.analytic import _ei_helper, _log_ei_helper, _scaled_improvement
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.models.utils import fantasize as fantasize_flag
from botorch.optim.initializers import initialize_q_batch, initialize_q_batch_nonneg
from botorch.optim.optimize import optimize_acqf
from botorch.posteriors import Posterior
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import NormalMCSampler, SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_normal_samples, draw_sobol_samples
from botorch.utils.transforms import (
    average_over_ensemble_models,
    t_batch_mode_transform,
)
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from torch import Tensor


def _validate_lsr_inputs(
    x_current: Tensor,
    bounds: Tensor,
    delta: float | Tensor,
    p: float,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Validate and normalize local search region geometry inputs."""
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError(
            f"`bounds` must have shape `(2, d)`, but got {tuple(bounds.shape)}."
        )
    d = bounds.shape[1]
    if (bounds[0] > bounds[1]).any():
        raise ValueError("Lower bounds cannot exceed upper bounds in `bounds`.")
    if math.isnan(p) or p < 1.0:
        raise ValueError(f"Norm order `p` must satisfy `p >= 1.0`, got {p}.")

    x_curr = x_current.to(device=bounds.device, dtype=bounds.dtype)
    if x_curr.ndim == 2 and x_curr.shape[0] == 1 and x_curr.shape[1] == d:
        x_curr = x_curr.squeeze(0)
    elif x_curr.ndim != 1 or x_curr.shape[0] != d:
        raise ValueError(
            f"`x_current` must have shape `(d,)` or `(1, d)` with d={d}, "
            f"got {tuple(x_current.shape)}."
        )

    delta_t = torch.as_tensor(delta, device=bounds.device, dtype=bounds.dtype)
    if delta_t.ndim == 0:
        delta_vec = delta_t.expand(d).clone()
    elif delta_t.ndim == 1 and delta_t.shape[0] == d:
        delta_vec = delta_t
    else:
        raise ValueError(
            f"`delta` must be a positive scalar or `(d,)` tensor with d={d}, "
            f"got shape {tuple(delta_t.shape)}."
        )
    if not (torch.isfinite(delta_vec).all() and (delta_vec > 0).all()):
        raise ValueError("`delta` must contain strictly positive, finite values.")

    return x_curr, bounds, delta_vec


def _extract_x_current_from_model(
    model: Model,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    r"""Extract the most recent training input ``train_inputs[-1]`` from ``model``."""
    curr: Any = model
    if not hasattr(curr, "train_inputs") and hasattr(curr, "models") and curr.models:
        curr = curr.models[0]
    if getattr(curr, "_has_transformed_inputs", False) and hasattr(
        curr, "_original_train_inputs"
    ):
        train_X = curr._original_train_inputs
    elif hasattr(curr, "train_inputs") and curr.train_inputs:
        train_X = curr.train_inputs[0]
    else:
        raise ValueError(
            "`x_current` was not provided and could not be inferred from `model`."
        )
    if train_X.shape[-2] < 1:
        raise ValueError(
            "`x_current` was not provided and `model` has empty `train_inputs`."
        )
    x_last = train_X[..., -1, :]
    while x_last.ndim > 1:
        x_last = x_last[0]
    return x_last.detach().to(device=device, dtype=dtype)


def _scaled_lp_power(z: Tensor, p: float) -> Tensor:
    r"""Compute ``sum_i |z_i|^p`` with NaN-free gradients at ``z_i == 0``."""
    if p == 2.0:
        return z.square().sum(dim=-1)
    if p == 1.0:
        return z.abs().sum(dim=-1)
    # Double-where avoids 0 * inf = NaN in backward at z == 0 for general p > 1.
    z_safe = torch.where(z == 0, torch.ones_like(z), z.abs())
    return torch.where(z == 0, torch.zeros_like(z), z_safe.pow(p)).sum(dim=-1)


def _scaled_lp_norm(z: Tensor, p: float) -> Tensor:
    r"""Compute ``||z||_p`` along the last dimension."""
    if math.isinf(p):
        return z.abs().amax(dim=-1)
    if p == 2.0:
        return torch.linalg.vector_norm(z, ord=2, dim=-1)
    if p == 1.0:
        return z.abs().sum(dim=-1)
    return _scaled_lp_power(z=z, p=p).pow(1.0 / p)


def compute_local_box_bounds(
    x_current: Tensor,
    bounds: Tensor,
    delta: float | Tensor,
    p: float = float("inf"),
) -> Tensor:
    r"""Compute the axis-aligned local box bounds around ``x_current``.

    Computes ``[max(lb, x_current - delta), min(ub, x_current + delta)]``. For
    ``p = float("inf")``, this box is the exact local search region
    :math:`\mathcal{T}(x_k) \cap \Theta`. For finite ``p >= 1.0``, this box is
    the tight outer bounding box containing :math:`\mathcal{T}(x_k) \cap \Theta`.

    Args:
        x_current: Current state tensor of shape ``(d,)`` or ``(1, d)``.
        bounds: Global domain bounds tensor of shape ``(2, d)``.
        delta: Scalar or ``(d,)`` tensor of per-dimension local step radii.
        p: Order of the :math:`\ell_p` norm defining :math:`\mathcal{T}(x_k)`,
            with ``p >= 1.0`` or ``float("inf")``.

    Returns:
        A ``(2, d)`` tensor of intersected local lower and upper bounds.
    """
    x_curr, bounds, delta_vec = _validate_lsr_inputs(
        x_current=x_current, bounds=bounds, delta=delta, p=p
    )
    local_lb = torch.maximum(bounds[0], x_curr - delta_vec)
    local_ub = torch.minimum(bounds[1], x_curr + delta_vec)
    if (local_lb > local_ub).any():
        raise ValueError(
            "Local search region around `x_current` does not intersect `bounds`."
        )
    return torch.stack([local_lb, local_ub], dim=0)


def make_local_region_constraint(
    x_current: Tensor,
    delta: float | Tensor,
    p: float = 2.0,
) -> tuple[Callable[[Tensor], Tensor], bool]:
    r"""Create a BoTorch intra-point nonlinear inequality constraint for LSR-BO.

    Constructs the differentiable inequality constraint
    :math:`g(x) = 1 - \| (x - x_k) / \delta \|_p^p \ge 0` (or
    :math:`1 - \| (x - x_k) / \delta \|_\infty \ge 0` when ``p = float("inf")``),
    formatted as ``(callable, True)`` for direct use with
    ``optimize_acqf(..., nonlinear_inequality_constraints=[...])``.

    Args:
        x_current: Current state tensor of shape ``(d,)`` or ``(1, d)``.
        delta: Scalar or ``(d,)`` tensor of per-dimension local step radii.
        p: Order of the :math:`\ell_p` norm, with ``p >= 1.0``.

    Returns:
        A 2-tuple ``(constraint_fn, True)`` representing an intra-point
        nonlinear inequality constraint ``constraint_fn(x) >= 0``.
    """
    if x_current.ndim == 2 and x_current.shape[0] == 1:
        x_curr = x_current.squeeze(0).detach().clone()
    elif x_current.ndim == 1:
        x_curr = x_current.detach().clone()
    else:
        raise ValueError(
            f"`x_current` must have shape `(d,)` or `(1, d)`, "
            f"got {tuple(x_current.shape)}."
        )
    d = x_curr.shape[0]
    if math.isnan(p) or p < 1.0:
        raise ValueError(f"Norm order `p` must satisfy `p >= 1.0`, got {p}.")

    delta_t = torch.as_tensor(delta, device=x_curr.device, dtype=x_curr.dtype)
    if delta_t.ndim == 0:
        delta_vec = delta_t.expand(d).detach().clone()
    elif delta_t.ndim == 1 and delta_t.shape[0] == d:
        delta_vec = delta_t.detach().clone()
    else:
        raise ValueError(
            f"`delta` must be a positive scalar or `(d,)` tensor with d={d}, "
            f"got shape {tuple(delta_t.shape)}."
        )
    if not (torch.isfinite(delta_vec).all() and (delta_vec > 0).all()):
        raise ValueError("`delta` must contain strictly positive, finite values.")

    def constraint_fn(X: Tensor) -> Tensor:
        x_ref = x_curr.to(device=X.device, dtype=X.dtype)
        d_ref = delta_vec.to(device=X.device, dtype=X.dtype)
        z = (X - x_ref) / d_ref
        if math.isinf(p):
            return 1.0 - z.abs().amax(dim=-1)
        return 1.0 - _scaled_lp_power(z=z, p=p)

    return constraint_fn, True


def _prox_scaled_lp_1d(
    y: Tensor,
    x_curr: Tensor,
    delta_vec: Tensor,
    lam: Tensor,
    p: float,
    local_lb: Tensor,
    local_ub: Tensor,
) -> Tensor:
    r"""Solve 1D box-constrained scaled l_p proximal subproblem."""
    diff = y - x_curr
    if p == 2.0:
        denom = 1.0 + 2.0 * lam.unsqueeze(-1) / delta_vec.square()
        x_unc = x_curr + diff / denom
    elif p == 1.0:
        shrink = (diff.abs() - lam.unsqueeze(-1) / delta_vec).clamp_min(0.0)
        x_unc = x_curr + diff.sign() * shrink
    else:
        # Solve monotone scalar equation u + (lam * p / delta^p) * u^(p - 1) = |diff|
        # on u in [0, |diff|] via safeguarded Newton-bisection.
        r = diff.abs()
        coeff = lam.unsqueeze(-1) * p / delta_vec.pow(p)
        u_lo = torch.zeros_like(r)
        u_hi = r.clone()
        u = 0.5 * r
        n_inner = 25 if y.dtype == torch.float32 else 45
        for _ in range(n_inner):
            u_pow = torch.where(u > 0, u.pow(p - 1.0), torch.zeros_like(u))
            f_val = u + coeff * u_pow - r
            u_hi = torch.where(f_val >= 0, u, u_hi)
            u_lo = torch.where(f_val < 0, u, u_lo)
            if p >= 2.0:
                df_val = 1.0 + coeff * (p - 1.0) * torch.where(
                    u > 0, u.pow(p - 2.0), torch.zeros_like(u)
                )
                u_newton = u - f_val / df_val
                in_bracket = (u_newton > u_lo) & (u_newton < u_hi)
                u = torch.where(in_bracket, u_newton, 0.5 * (u_lo + u_hi))
            else:
                u = 0.5 * (u_lo + u_hi)
        x_unc = x_curr + diff.sign() * u_lo
    return x_unc.clamp(min=local_lb, max=local_ub)


def project_to_local_region(
    target_x: Tensor,
    x_current: Tensor,
    bounds: Tensor,
    delta: float | Tensor,
    p: float = float("inf"),
) -> Tensor:
    r"""Compute the exact Euclidean projection of ``target_x`` onto ``T(x_k) n Theta``.

    Solves the strictly convex projection problem:

    .. math::

        \Pi_{\mathcal{T}(x_k) \cap \Theta}(y)
        = \arg\min_{x \in \Theta} \frac{1}{2} \|x - y\|_2^2
        \quad \text{s.t.} \quad
        \left\| \frac{x - x_k}{\delta} \right\|_p \le 1.

    For ``p = float("inf")``, this reduces to coordinate-wise clamping onto
    ``compute_local_box_bounds(x_current, bounds, delta, p)``. For finite
    ``p >= 1.0``, this solves the exact KKT conditions via separable 1D dual
    bisection, supporting both scalar and vector ``delta`` as well as arbitrary
    batch shapes on ``target_x``.

    Args:
        target_x: Target point(s) tensor of shape ``(..., d)``.
        x_current: Current state tensor of shape ``(d,)`` or ``(1, d)``.
        bounds: Global domain bounds tensor of shape ``(2, d)``.
        delta: Scalar or ``(d,)`` tensor of per-dimension local step radii.
        p: Order of the :math:`\ell_p` norm, with ``p >= 1.0`` or ``float("inf")``.

    Returns:
        Projected tensor of the same shape ``(..., d)`` as ``target_x``, strictly
        satisfying both global ``bounds`` and local region constraints.
    """
    orig_dtype = target_x.dtype
    orig_device = target_x.device
    bounds_ref = bounds.to(device=orig_device, dtype=orig_dtype)
    x_curr, bounds_ref, delta_vec = _validate_lsr_inputs(
        x_current=x_current, bounds=bounds_ref, delta=delta, p=p
    )
    d = bounds_ref.shape[1]
    if target_x.shape[-1] != d:
        raise ValueError(
            f"`target_x` last dimension must be {d}, got shape {tuple(target_x.shape)}."
        )

    local_box = compute_local_box_bounds(
        x_current=x_curr, bounds=bounds_ref, delta=delta_vec, p=p
    )
    local_lb, local_ub = local_box[0], local_box[1]
    x_box = target_x.clamp(min=local_lb, max=local_ub)

    if math.isinf(p):
        return x_box

    # Check if box-clamped point already lies inside the l_p ball.
    init_power = _scaled_lp_power(z=(x_box - x_curr) / delta_vec, p=p)
    if bool((init_power <= 1.0).all()):
        return x_box

    # Verify non-emptiness of T(x_k) intersect Theta by checking the point in
    # Theta closest to x_k in every coordinate.
    x_closest = x_curr.clamp(min=local_lb, max=local_ub)
    min_power = _scaled_lp_power(z=(x_closest - x_curr) / delta_vec, p=p)
    tol = 1e-6 if orig_dtype == torch.float32 else 1e-12
    if float(min_power.item()) > 1.0 + tol:
        raise ValueError(
            "Local search region around `x_current` does not intersect `bounds`."
        )
    if float(min_power.item()) >= 1.0 - tol:
        return x_closest.expand_as(target_x).clone()

    # Exact separable KKT dual bisection in float64 for high accuracy, followed
    # by dtype-consistent feasibility tightening.
    y64 = target_x.to(dtype=torch.float64)
    xc64 = x_curr.to(dtype=torch.float64)
    dv64 = delta_vec.to(dtype=torch.float64)
    lb64 = local_lb.to(dtype=torch.float64)
    ub64 = local_ub.to(dtype=torch.float64)

    lam_lo = torch.zeros(y64.shape[:-1], device=orig_device, dtype=torch.float64)
    r_max = (y64 - xc64).abs().amax(dim=-1).clamp_min(1.0)
    d_max = dv64.max()
    lam_hi = torch.full_like(lam_lo, float(d_max.item()) * 2.0) * r_max

    for _ in range(128):
        x_try = _prox_scaled_lp_1d(
            y=y64,
            x_curr=xc64,
            delta_vec=dv64,
            lam=lam_hi,
            p=p,
            local_lb=lb64,
            local_ub=ub64,
        )
        phi_hi = _scaled_lp_power(z=(x_try - xc64) / dv64, p=p)
        need_more = phi_hi > 1.0
        if not bool(need_more.any()):
            break
        lam_hi = torch.where(need_more, lam_hi * 2.0, lam_hi)

    for _ in range(64):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        x_mid = _prox_scaled_lp_1d(
            y=y64,
            x_curr=xc64,
            delta_vec=dv64,
            lam=lam_mid,
            p=p,
            local_lb=lb64,
            local_ub=ub64,
        )
        phi_mid = _scaled_lp_power(z=(x_mid - xc64) / dv64, p=p)
        lam_lo = torch.where(phi_mid > 1.0, lam_mid, lam_lo)
        lam_hi = torch.where(phi_mid > 1.0, lam_hi, lam_mid)

    x_proj64 = _prox_scaled_lp_1d(
        y=y64,
        x_curr=xc64,
        delta_vec=dv64,
        lam=lam_hi,
        p=p,
        local_lb=lb64,
        local_ub=ub64,
    )
    x_proj = x_proj64.to(dtype=orig_dtype).clamp(min=local_lb, max=local_ub)

    # Guard against single-precision rounding pushing ||(x - x_k)/delta||_p
    # slightly above 1.0 by pulling infinitesimally toward x_closest.
    post_norm = _scaled_lp_norm(z=(x_proj - x_curr) / delta_vec, p=p)
    over = post_norm > 1.0
    if bool(over.any()):
        eps = torch.finfo(orig_dtype).eps * 4.0
        scale = (1.0 - eps) / post_norm.clamp_min(1.0).unsqueeze(-1)
        x_pulled = (x_closest + scale * (x_proj - x_closest)).clamp(
            min=local_lb, max=local_ub
        )
        x_proj = torch.where(over.unsqueeze(-1), x_pulled, x_proj)

    return torch.where((init_power <= 1.0).unsqueeze(-1), x_box, x_proj)


def _sample_local_region_candidates(
    x_current: Tensor,
    bounds: Tensor,
    delta: float | Tensor,
    p: float,
    n: int,
    q: int = 1,
    seed: int | None = None,
) -> Tensor:
    r"""Draw ``n x q x d`` feasible candidates inside ``T(x_current) n Theta``."""
    x_curr, bounds, delta_vec = _validate_lsr_inputs(
        x_current=x_current, bounds=bounds, delta=delta, p=p
    )
    local_bounds = compute_local_box_bounds(
        x_current=x_curr, bounds=bounds, delta=delta_vec, p=p
    )
    d = bounds.shape[1]
    if math.isinf(p):
        return draw_sobol_samples(bounds=local_bounds, n=n, q=q, seed=seed)

    # Combine projected local-box Sobol samples with radial l_p-ball samples.
    n_half = max(1, n // 2)
    n_rem = max(0, n - n_half)
    box_samples = draw_sobol_samples(bounds=local_bounds, n=n_half, q=q, seed=seed)
    box_feasible = project_to_local_region(
        target_x=box_samples, x_current=x_curr, bounds=bounds, delta=delta_vec, p=p
    )
    if n_rem == 0:
        return box_feasible

    unit_cube = torch.stack(
        [
            -torch.ones(d, device=bounds.device, dtype=bounds.dtype),
            torch.ones(d, device=bounds.device, dtype=bounds.dtype),
        ],
        dim=0,
    )
    raw_dirs = draw_sobol_samples(
        bounds=unit_cube,
        n=n_rem,
        q=q,
        seed=None if seed is None else seed + 17,
    )
    norms = _scaled_lp_norm(z=raw_dirs, p=p).clamp_min(1e-12).unsqueeze(-1)
    unit_dirs = raw_dirs / norms
    rad_bounds = torch.zeros(2, 1, device=bounds.device, dtype=bounds.dtype)
    rad_bounds[1] = 1.0
    radii = draw_sobol_samples(
        bounds=rad_bounds,
        n=n_rem,
        q=q,
        seed=None if seed is None else seed + 31,
    ).pow(1.0 / d)
    ball_raw = x_curr + delta_vec * unit_dirs * radii
    ball_feasible = project_to_local_region(
        target_x=ball_raw, x_current=x_curr, bounds=bounds, delta=delta_vec, p=p
    )
    return torch.cat([box_feasible, ball_feasible], dim=0)


def gen_local_region_initial_conditions(
    x_current: Tensor,
    bounds: Tensor,
    delta: float | Tensor,
    p: float = 2.0,
) -> Callable[..., Tensor]:
    r"""Create an ``ic_generator`` producing feasible restarts in ``T(x_k) n Theta``.

    Required by ``botorch.optim.optimize_acqf`` whenever
    ``nonlinear_inequality_constraints`` are provided (i.e. for finite ``p >= 1``).

    Args:
        x_current: Current state tensor of shape ``(d,)`` or ``(1, d)``.
        bounds: Global domain bounds tensor of shape ``(2, d)``.
        delta: Scalar or ``(d,)`` tensor of per-dimension local step radii.
        p: Order of the :math:`\ell_p` norm, with ``p >= 1.0``.

    Returns:
        A callable compatible with the ``ic_generator`` argument of
        ``botorch.optim.optimize_acqf``.
    """
    x_curr, bounds_clean, delta_vec = _validate_lsr_inputs(
        x_current=x_current, bounds=bounds, delta=delta, p=p
    )

    def ic_generator(
        acq_function: AcquisitionFunction,
        bounds: Tensor,
        q: int,
        num_restarts: int,
        raw_samples: int,
        fixed_features: dict[int, float] | None = None,
        options: dict[str, bool | float | int] | None = None,
        inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
        equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    ) -> Tensor:
        opts = options or {}
        seed = opts.get("seed")
        seed_int = int(seed) if isinstance(seed, int) else None
        n_raw = max(raw_samples, num_restarts)
        # Shrink radius slightly so initial restarts lie strictly inside the l_p ball,
        # avoiding active-boundary degeneracy on step 0 of SciPy SLSQP.
        interior_scale = 1.0 - (1e-2 if bounds.dtype == torch.float32 else 1e-3)
        delta_ic = (
            delta_vec.to(device=bounds.device, dtype=bounds.dtype) * interior_scale
        )
        X_rnd = _sample_local_region_candidates(
            x_current=x_curr.to(device=bounds.device, dtype=bounds.dtype),
            bounds=bounds_clean.to(device=bounds.device, dtype=bounds.dtype),
            delta=delta_ic,
            p=p,
            n=n_raw,
            q=q,
            seed=seed_int,
        )
        if fixed_features:
            for idx, val in fixed_features.items():
                X_rnd[..., idx] = val
            X_rnd = project_to_local_region(
                target_x=X_rnd,
                x_current=x_curr.to(device=bounds.device, dtype=bounds.dtype),
                bounds=bounds_clean.to(device=bounds.device, dtype=bounds.dtype),
                delta=delta_ic,
                p=p,
            )

        with torch.no_grad():
            batch_limit = int(opts.get("init_batch_limit", opts.get("batch_limit", 64)))
            Y_rnd = torch.cat(
                [acq_function(cand) for cand in X_rnd.split(batch_limit, dim=0)],
                dim=0,
            )

        eta = float(opts.get("eta", 1.0))
        is_nonneg = bool(opts.get("nonnegative", False)) and not getattr(
            acq_function, "_log", False
        )
        if is_nonneg:
            X_init, _ = initialize_q_batch_nonneg(
                X=X_rnd, acq_vals=Y_rnd, n=num_restarts, eta=eta
            )
            return X_init
        X_init, _ = initialize_q_batch(X=X_rnd, acq_vals=Y_rnd, n=num_restarts, eta=eta)
        return X_init

    return ic_generator


@dataclass
class LSRBOStepInfo:
    r"""Diagnostic metadata from a single LSR-BO global-local step.

    Args:
        candidate: Selected next point of shape ``(1, d)`` in ``T(x_k) n Theta``.
        acq_value: Scalar acquisition function value at ``candidate``.
        x_current: Current state ``x_k`` of shape ``(d,)``.
        x_local: Local acquisition maximizer of shape ``(1, d)``.
        acq_local: Scalar acquisition value at ``x_local``.
        x_global: Global acquisition maximizer of shape ``(1, d)``.
        acq_global: Scalar acquisition value at ``x_global``.
        used_local_step: ``True`` if ``x_local`` satisfied the ``gamma`` criterion;
            ``False`` if LSR-BO projected toward ``x_global``.
    """

    candidate: Tensor
    acq_value: Tensor
    x_current: Tensor
    x_local: Tensor
    acq_local: Tensor
    x_global: Tensor
    acq_global: Tensor
    used_local_step: bool


def _optimize_over_local_region(
    acq_function: AcquisitionFunction,
    x_current: Tensor,
    bounds: Tensor,
    delta: Tensor,
    p: float,
    num_restarts: int,
    raw_samples: int,
    options: dict[str, bool | float | int | str] | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Maximize ``acq_function`` over ``T(x_current) n Theta`` for ``q = 1``."""
    local_bounds = compute_local_box_bounds(
        x_current=x_current, bounds=bounds, delta=delta, p=p
    )
    if math.isinf(p):
        cand, val = optimize_acqf(
            acq_function=acq_function,
            bounds=local_bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
            retry_on_optimization_warning=False,
        )
        cand = project_to_local_region(
            target_x=cand, x_current=x_current, bounds=bounds, delta=delta, p=p
        )
        with torch.no_grad():
            val = acq_function(cand).squeeze()
        return cand, val

    # Shrink constraint radius by a tiny dtype margin so float64->float32 cast
    # after SciPy SLSQP never violates BoTorch's post-hoc feasibility check.
    margin = 1e-4 if bounds.dtype == torch.float32 else 1e-9
    delta_slsqp = delta * (1.0 - margin)
    nlc = [make_local_region_constraint(x_current=x_current, delta=delta_slsqp, p=p)]
    ic_gen = gen_local_region_initial_conditions(
        x_current=x_current, bounds=bounds, delta=delta_slsqp, p=p
    )
    slsqp_options: dict[str, bool | float | int | str] = {
        "ftol": 1e-5 if bounds.dtype == torch.float32 else 1e-8,
        "maxiter": 25,
    }
    if options is not None:
        slsqp_options.update(options)
    cand, _ = optimize_acqf(
        acq_function=acq_function,
        bounds=local_bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=slsqp_options,
        nonlinear_inequality_constraints=nlc,
        ic_generator=ic_gen,
        retry_on_optimization_warning=False,
    )
    cand = project_to_local_region(
        target_x=cand, x_current=x_current, bounds=bounds, delta=delta, p=p
    )
    with torch.no_grad():
        val = acq_function(cand).squeeze()
    return cand, val


def _should_select_local_candidate(
    acq_function: AcquisitionFunction,
    acq_local: Tensor,
    acq_global: Tensor,
    x_global: Tensor,
    x_current: Tensor,
    delta: Tensor,
    p: float,
    gamma: float,
) -> bool:
    r"""Evaluate LSR-BO's local-vs-global trade-off condition."""
    if gamma <= 0.0:
        return True

    tol = 1e-6 if x_global.dtype == torch.float32 else 1e-10
    dist_global = _scaled_lp_norm(z=(x_global.squeeze(0) - x_current) / delta, p=p)
    if float(dist_global.item()) <= 1.0 + tol:
        return True

    if gamma >= 1.0:
        return False

    is_log = bool(getattr(acq_function, "_log", False))
    local_val = float(acq_local.item())
    global_val = float(acq_global.item())

    if is_log:
        if math.isinf(global_val) and global_val < 0:
            return True
        return (local_val - global_val) >= math.log(gamma)

    if global_val > 0.0 and local_val >= 0.0:
        return local_val >= gamma * global_val

    if global_val == 0.0 and local_val == 0.0:
        return True

    # For acquisition functions that can take negative values (e.g. UCB or
    # PosteriorMean on unstandardized negative objectives), compare nonnegative
    # improvements above min(alpha(x_current), alpha(x_local)).
    with torch.no_grad():
        curr_val = float(acq_function(x_current.view(1, 1, -1)).squeeze().item())
    baseline = min(curr_val, local_val)
    imp_global = global_val - baseline
    imp_local = local_val - baseline
    if imp_global <= 0.0:
        return True
    return imp_local >= gamma * imp_global


def optimize_lsr_bo_step(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    delta: float | Tensor,
    x_current: Tensor | None = None,
    gamma: float = 0.5,
    p: float = float("inf"),
    num_restarts: int = 10,
    raw_samples: int = 256,
    options: dict[str, bool | float | int | str] | None = None,
    local_ball_fraction: float = 0.0,
    return_step_info: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, LSRBOStepInfo]:
    r"""Execute one step of LSR-BO global-local constrained acquisition optimization.

    Implements Algorithm 1 of [paulson2023lsrbo]_:
    1. Maximizes ``acq_function`` locally over :math:`\mathcal{T}(x_k) \cap \Theta`
       to obtain ``(x_local, acq_local)``.
    2. Maximizes ``acq_function`` globally over :math:`\Theta` to obtain
       ``(x_global, acq_global)``.
    3. Compares local and global improvements using ``gamma in [0, 1]`` (in linear
       space for standard acquisitions or log-ratio space ``acq_local - acq_global
       >= log(gamma)`` when ``acq_function._log == True``).
    4. Selects ``x_local`` if the threshold holds; otherwise projects ``x_global``
       onto :math:`\mathcal{T}(x_k) \cap \Theta` (optionally refining locally
       within ``local_ball_fraction * delta`` of the projected point).

    Args:
        acq_function: A BoTorch ``AcquisitionFunction`` (supports both standard
            and ``_log = True`` acquisitions).
        bounds: Global domain bounds tensor of shape ``(2, d)``.
        delta: Scalar or ``(d,)`` tensor of per-dimension local step radii.
        x_current: Current state tensor of shape ``(d,)`` or ``(1, d)``. If
            ``None``, defaults to ``train_inputs[-1]`` of ``acq_function.model``.
        gamma: Global-local trade-off parameter in ``[0, 1]``. ``gamma = 0``
            recovers greedy local BO; ``gamma = 1`` steers toward ``x_global``
            whenever ``x_global`` lies outside :math:`\mathcal{T}(x_k)`.
        p: Order of the :math:`\ell_p` norm defining :math:`\mathcal{T}(x_k)`.
        num_restarts: Number of multistart optimization restarts.
        raw_samples: Number of raw samples for restart initialization.
        options: Optional dictionary of optimizer options for ``optimize_acqf``.
        local_ball_fraction: Optional radius fraction in ``[0, 1)`` around the
            projected global step within which to refine ``acq_function`` while
            staying inside :math:`\mathcal{T}(x_k) \cap \Theta`. Defaults to
            ``0.0`` (exact projected step).
        return_step_info: If ``True``, also return an ``LSRBOStepInfo`` object.

    Returns:
        A 2-tuple ``(candidate, acq_value)`` of shapes ``(1, d)`` and ``()``, or
        a 3-tuple ``(candidate, acq_value, step_info)`` if ``return_step_info``
        is ``True``.
    """
    if not (0.0 <= gamma <= 1.0) or math.isnan(gamma):
        raise ValueError(f"`gamma` must lie in `[0, 1]`, got {gamma}.")
    if not (0.0 <= local_ball_fraction < 1.0) or math.isnan(local_ball_fraction):
        raise ValueError(
            f"`local_ball_fraction` must lie in `[0, 1)`, got {local_ball_fraction}."
        )

    if x_current is None:
        if not hasattr(acq_function, "model") or acq_function.model is None:
            raise ValueError(
                "`x_current` must be specified when `acq_function` has no `model`."
            )
        x_current = _extract_x_current_from_model(
            model=acq_function.model, dtype=bounds.dtype, device=bounds.device
        )

    x_curr, bounds_clean, delta_vec = _validate_lsr_inputs(
        x_current=x_current, bounds=bounds, delta=delta, p=p
    )

    x_local, acq_local = _optimize_over_local_region(
        acq_function=acq_function,
        x_current=x_curr,
        bounds=bounds_clean,
        delta=delta_vec,
        p=p,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
    )

    if gamma == 0.0 and not return_step_info:
        return x_local, acq_local

    x_global, acq_global = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds_clean,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
        retry_on_optimization_warning=False,
    )
    acq_global = acq_global.squeeze()

    # If x_global is already inside T(x_k) and improves upon x_local, upgrade x_local.
    tol = 1e-6 if bounds_clean.dtype == torch.float32 else 1e-10
    dist_global = _scaled_lp_norm(z=(x_global.squeeze(0) - x_curr) / delta_vec, p=p)
    if float(dist_global.item()) <= 1.0 + tol and float(acq_global.item()) > float(
        acq_local.item()
    ):
        x_local = project_to_local_region(
            target_x=x_global,
            x_current=x_curr,
            bounds=bounds_clean,
            delta=delta_vec,
            p=p,
        )
        with torch.no_grad():
            acq_local = acq_function(x_local).squeeze()

    use_local = _should_select_local_candidate(
        acq_function=acq_function,
        acq_local=acq_local,
        acq_global=acq_global,
        x_global=x_global,
        x_current=x_curr,
        delta=delta_vec,
        p=p,
        gamma=gamma,
    )

    if use_local:
        candidate = x_local
        acq_value = acq_local
    else:
        x_proj = project_to_local_region(
            target_x=x_global,
            x_current=x_curr,
            bounds=bounds_clean,
            delta=delta_vec,
            p=p,
        )
        if local_ball_fraction > 0.0:
            sub_bounds = compute_local_box_bounds(
                x_current=x_proj.squeeze(0),
                bounds=compute_local_box_bounds(
                    x_current=x_curr, bounds=bounds_clean, delta=delta_vec, p=p
                ),
                delta=local_ball_fraction * delta_vec,
                p=float("inf"),
            )
            candidate, _ = _optimize_over_local_region(
                acq_function=acq_function,
                x_current=x_curr,
                bounds=sub_bounds,
                delta=delta_vec,
                p=p,
                num_restarts=max(2, num_restarts // 2),
                raw_samples=max(32, raw_samples // 2),
                options=options,
            )
        else:
            candidate = x_proj
        with torch.no_grad():
            acq_value = acq_function(candidate).squeeze()

    step_info = LSRBOStepInfo(
        candidate=candidate,
        acq_value=acq_value,
        x_current=x_curr,
        x_local=x_local,
        acq_local=acq_local,
        x_global=x_global,
        acq_global=acq_global,
        used_local_step=use_local,
    )
    if return_step_info:
        return candidate, acq_value, step_info
    return candidate, acq_value


class LSRBO:
    r"""Local Search Region Constrained BO policy [paulson2023lsrbo]_.

    Balances local exploitation within :math:`\mathcal{T}(x_k) \cap \Theta` and
    directed multi-step traversal toward distant high-value basins in :math:`\Theta`.
    Every recommended query strictly satisfies
    :math:`\|(x_{k+1} - x_k)/\delta\|_p \le 1` and :math:`x_{k+1} \in \Theta`.
    """

    def __init__(
        self,
        bounds: Tensor,
        delta: float | Tensor,
        gamma: float = 0.5,
        p: float = float("inf"),
        num_restarts: int = 10,
        raw_samples: int = 256,
        options: dict[str, bool | float | int | str] | None = None,
        local_ball_fraction: float = 0.0,
    ) -> None:
        r"""Initialize the LSR-BO global-local policy.

        Args:
            bounds: Global domain bounds tensor of shape ``(2, d)``.
            delta: Scalar or ``(d,)`` tensor of per-dimension local step radii.
            gamma: Global-local trade-off threshold in ``[0, 1]``. ``gamma = 0``
                performs pure greedy local BO; ``gamma = 1`` steers toward the
                global acquisition maximizer whenever it lies outside the local
                region.
            p: Order of the :math:`\ell_p` norm defining :math:`\mathcal{T}(x_k)`.
            num_restarts: Number of multistart optimization restarts.
            raw_samples: Number of raw samples for restart initialization.
            options: Optional dictionary of optimizer options for ``optimize_acqf``.
            local_ball_fraction: Optional radius fraction in ``[0, 1)`` around
                the projected global step for local refinement.
        """
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError(
                f"`bounds` must have shape `(2, d)`, got {tuple(bounds.shape)}."
            )
        _, bounds_clean, delta_vec = _validate_lsr_inputs(
            x_current=0.5 * (bounds[0] + bounds[1]),
            bounds=bounds,
            delta=delta,
            p=p,
        )
        if not (0.0 <= gamma <= 1.0) or math.isnan(gamma):
            raise ValueError(f"`gamma` must lie in `[0, 1]`, got {gamma}.")
        if not (0.0 <= local_ball_fraction < 1.0) or math.isnan(local_ball_fraction):
            raise ValueError(
                f"`local_ball_fraction` must lie in `[0, 1)`, got "
                f"{local_ball_fraction}."
            )
        self.bounds = bounds_clean
        self.delta = delta_vec
        self.gamma = gamma
        self.p = p
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.options = options
        self.local_ball_fraction = local_ball_fraction
        self.last_step_info: LSRBOStepInfo | None = None

    def get_local_bounds(self, x_current: Tensor) -> Tensor:
        r"""Compute the axis-aligned local box bounds around ``x_current``.

        Args:
            x_current: Current state tensor of shape ``(d,)`` or ``(1, d)``.

        Returns:
            A ``(2, d)`` tensor of intersected local bounds.
        """
        return compute_local_box_bounds(
            x_current=x_current,
            bounds=self.bounds,
            delta=self.delta,
            p=self.p,
        )

    def project(self, target_x: Tensor, x_current: Tensor) -> Tensor:
        r"""Project ``target_x`` onto :math:`\mathcal{T}(x_k) \cap \Theta`.

        Args:
            target_x: Target point(s) tensor of shape ``(..., d)``.
            x_current: Current state tensor of shape ``(d,)`` or ``(1, d)``.

        Returns:
            Projected tensor of the same shape as ``target_x``.
        """
        return project_to_local_region(
            target_x=target_x,
            x_current=x_current,
            bounds=self.bounds,
            delta=self.delta,
            p=self.p,
        )

    def step(
        self,
        acq_function: AcquisitionFunction,
        x_current: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Select the next feasible query point ``x_{k+1}`` via LSR-BO.

        Args:
            acq_function: Acquisition function to optimize.
            x_current: Current state ``x_k`` of shape ``(d,)`` or ``(1, d)``. If
                ``None``, defaults to ``train_inputs[-1]`` of ``acq_function.model``.

        Returns:
            A 2-tuple ``(candidate, acq_value)`` of shapes ``(1, d)`` and ``()``.
        """
        out = optimize_lsr_bo_step(
            acq_function=acq_function,
            bounds=self.bounds.to(
                device=x_current.device
                if x_current is not None
                else self.bounds.device,
                dtype=x_current.dtype if x_current is not None else self.bounds.dtype,
            ),
            delta=self.delta,
            x_current=x_current,
            gamma=self.gamma,
            p=self.p,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options=self.options,
            local_ball_fraction=self.local_ball_fraction,
            return_step_info=True,
        )
        assert len(out) == 3
        candidate, acq_value, step_info = out
        self.last_step_info = step_info
        return candidate, acq_value

    def __call__(
        self,
        acq_function: AcquisitionFunction,
        x_current: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Alias for ``step`` to evaluate one LSR-BO iteration.

        Args:
            acq_function: Acquisition function to optimize.
            x_current: Current state ``x_k`` of shape ``(d,)`` or ``(1, d)``.

        Returns:
            A 2-tuple ``(candidate, acq_value)`` of shapes ``(1, d)`` and ``()``.
        """
        return self.step(acq_function=acq_function, x_current=x_current)


class GaussHermiteNormalSampler(NormalMCSampler):
    r"""1D Gauss-Hermite quadrature sampler for scalar Gaussian fantasies.

    Produces deterministic standard normal quadrature nodes and normalized
    quadrature weights ``base_weights`` summing to ``1.0``, as used in
    [paulson2022lsrlookahead]_ to integrate single-output Stage-1 fantasies.
    """

    def __init__(self, sample_shape: torch.Size) -> None:
        r"""Initialize the Gauss-Hermite normal quadrature sampler.

        Args:
            sample_shape: Sample shape ``torch.Size([num_quadrature_nodes])``.
        """
        if len(sample_shape) != 1 or sample_shape[0] < 1:
            raise ValueError(
                "`GaussHermiteNormalSampler` requires a 1D positive `sample_shape`, "
                f"got {sample_shape}."
            )
        super().__init__(sample_shape=sample_shape)
        nodes_np, weights_np = np.polynomial.hermite.hermgauss(int(sample_shape[0]))
        nodes = torch.from_numpy(math.sqrt(2.0) * nodes_np).to(dtype=torch.float64)
        weights = torch.from_numpy(weights_np / math.sqrt(math.pi)).to(
            dtype=torch.float64
        )
        self.register_buffer("_gh_nodes", nodes)
        self.register_buffer("base_weights", weights)

    def _construct_base_samples(self, posterior: Posterior) -> None:
        r"""Construct quadrature base samples matching ``posterior``.

        Args:
            posterior: Posterior distribution to sample from.
        """
        target_shape = self._get_collapsed_shape(posterior=posterior)
        output_dim = target_shape[len(self.sample_shape) :].numel()
        if output_dim != 1:
            raise UnsupportedError(
                "`GaussHermiteNormalSampler` supports 1D scalar fantasies "
                f"(`q * m == 1`), but got `q * m = {output_dim}`."
            )
        if self.base_samples is None or self.base_samples.shape != target_shape:
            view_shape = self.sample_shape + torch.Size(
                [1] * (len(target_shape) - len(self.sample_shape))
            )
            base_samples = (
                self._gh_nodes.to(device=posterior.device, dtype=posterior.dtype)
                .view(view_shape)
                .expand(target_shape)
                .contiguous()
            )
            self.register_buffer("base_samples", base_samples)
        self.base_weights = self.base_weights.to(
            device=posterior.device, dtype=posterior.dtype
        )
        self.to(device=posterior.device, dtype=posterior.dtype)


class LocalSearchRolloutAcquisitionFunction(AcquisitionFunction, MCSamplerMixin):
    r"""Multi-step ADP rollout acquisition function under local search constraints.

    Implements the approximate dynamic programming (ADP) rollout lookahead policy
    of [paulson2022lsrlookahead]_ over horizon ``H >= 1``. Given a first-stage
    candidate :math:`x_1 \in \mathcal{T}(x_0) \cap \Theta`:
    1. Evaluates the immediate Stage-1 reward and draws ``num_fantasies``
       reparameterized fantasy observations :math:`y_1^{(s)} \sim p(y \mid x_1,
       \mathcal{D}_0)` via ``model.fantasize(X=x_1, sampler=...)``.
    2. Rolls out stages :math:`h = 2, \dots, H` of a fast local-search-constrained
       base policy :math:`x_h^{(s)} = \arg\max_{x \in \mathcal{T}(x_{h-1}^{(s)})
       \cap \Theta} \alpha_{\text{base}}(x \mid \mathcal{D}_{h-1}^{(s)})` and
       updates the fantasy belief along each trajectory via
       ``condition_on_observations``.
    3. Aggregates either the expected discounted cumulative improvement
       :math:`\text{EI}_1(x_1) + \mathbb{E}\left[\sum_{h=2}^H \beta^{h-1}
       \text{EI}_h(x_h^{(s)})\right]` (``reward_mode="cumulative_ei"``) or the
       expected terminal maximum posterior mean
       :math:`\mathbb{E}[\max_x \mu_H^{(s)}(x)]` (``reward_mode="terminal_mean"``),
       preserving end-to-end autograd differentiability with respect to ``x_1``.
    """

    def __init__(
        self,
        model: Model,
        bounds: Tensor,
        delta: float | Tensor,
        x_current: Tensor | None = None,
        horizon: int = 2,
        num_fantasies: int = 16,
        discount_factor: float = 1.0,
        p: float = float("inf"),
        best_f: float | Tensor | None = None,
        sampler: MCSampler | None = None,
        use_gauss_hermite: bool = False,
        num_local_candidates: int = 32,
        base_policy: str = "ei",
        ucb_beta: float = 2.0,
        lsr_gamma: float = 0.5,
        reward_mode: str = "cumulative_ei",
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
        X_pending: Tensor | None = None,
        seed: int = 0,
    ) -> None:
        r"""Initialize the local-search-constrained multi-step rollout acquisition.

        Args:
            model: A fitted single-outcome model (or multi-outcome model paired
                with a scalarizing ``posterior_transform``).
            bounds: Global domain bounds tensor of shape ``(2, d)``.
            delta: Scalar or ``(d,)`` tensor of per-dimension local step radii.
            x_current: Current operating point ``x_0`` of shape ``(d,)`` or
                ``(1, d)``. If ``None``, defaults to ``train_inputs[-1]``.
            horizon: Lookahead horizon ``H >= 1``. ``horizon = 1`` corresponds to
                myopic single-step lookahead; ``horizon >= 2`` rolls out ``H - 1``
                additional local-search-constrained stages.
            num_fantasies: Number of Stage-1 fantasy trajectories when ``sampler``
                is ``None``.
            discount_factor: Stage reward discount factor in ``(0, 1]``.
            p: Order of the :math:`\ell_p` norm defining :math:`\mathcal{T}(x_k)`.
            best_f: Incumbent best objective value. If ``None``, inferred as the
                best posterior mean across ``model``'s training inputs.
            sampler: Optional ``MCSampler`` for Stage-1 fantasies.
            use_gauss_hermite: If ``True`` and ``sampler is None``, uses
                ``GaussHermiteNormalSampler`` for Stage-1 fantasies.
            num_local_candidates: Number of QMC local candidates evaluated around
                :math:`x_{h-1}^{(s)}` at each rollout stage :math:`h \ge 2`.
            base_policy: Base rollout policy in ``{"ei", "ucb", "lsr_bo"}``.
            ucb_beta: Exploration parameter ``beta > 0`` when ``base_policy="ucb"``.
            lsr_gamma: Global-local threshold in ``[0, 1]`` when
                ``base_policy="lsr_bo"``.
            reward_mode: Rollout objective in ``{"cumulative_ei", "terminal_mean"}``.
            posterior_transform: Optional ``PosteriorTransform`` producing a
                single-outcome posterior.
            maximize: If ``True``, treat the objective as maximization.
            X_pending: Optional ``m x d`` tensor of pending evaluations.
            seed: Seed for deterministic QMC candidate templates and rollout noise.
        """
        super().__init__(model=model)
        model.eval()
        if model.num_outputs > 1 and posterior_transform is None:
            raise UnsupportedError(
                "`LocalSearchRolloutAcquisitionFunction` requires a single-output "
                "`model` or a scalarizing `posterior_transform`."
            )
        if horizon < 1:
            raise ValueError(f"`horizon` must satisfy `horizon >= 1`, got {horizon}.")
        if num_fantasies < 1:
            raise ValueError(
                f"`num_fantasies` must satisfy `num_fantasies >= 1`, got "
                f"{num_fantasies}."
            )
        if not (0.0 < discount_factor <= 1.0) or math.isnan(discount_factor):
            raise ValueError(
                f"`discount_factor` must lie in `(0, 1]`, got {discount_factor}."
            )
        if num_local_candidates < 1:
            raise ValueError(
                f"`num_local_candidates` must be >= 1, got {num_local_candidates}."
            )
        if base_policy not in ("ei", "ucb", "lsr_bo"):
            raise ValueError(
                "`base_policy` must be one of `('ei', 'ucb', 'lsr_bo')`, "
                f"got {base_policy!r}."
            )
        if ucb_beta <= 0.0 or math.isnan(ucb_beta):
            raise ValueError(f"`ucb_beta` must be > 0, got {ucb_beta}.")
        if not (0.0 <= lsr_gamma <= 1.0) or math.isnan(lsr_gamma):
            raise ValueError(f"`lsr_gamma` must lie in `[0, 1]`, got {lsr_gamma}.")
        if reward_mode not in ("cumulative_ei", "terminal_mean"):
            raise ValueError(
                "`reward_mode` must be `'cumulative_ei'` or `'terminal_mean'`, "
                f"got {reward_mode!r}."
            )

        if x_current is None:
            x_current = _extract_x_current_from_model(
                model=model, dtype=bounds.dtype, device=bounds.device
            )
        x_curr, bounds_clean, delta_vec = _validate_lsr_inputs(
            x_current=x_current, bounds=bounds, delta=delta, p=p
        )

        if sampler is None:
            if use_gauss_hermite:
                sampler = GaussHermiteNormalSampler(
                    sample_shape=torch.Size([num_fantasies])
                )
            else:
                sampler = SobolQMCNormalSampler(
                    sample_shape=torch.Size([num_fantasies])
                )
        MCSamplerMixin.__init__(self, sampler=sampler)
        self.num_fantasies = int(sampler.sample_shape[0])

        self.register_buffer("bounds", bounds_clean)
        self.register_buffer("delta", delta_vec)
        self.register_buffer("x_current", x_curr)
        self.horizon = horizon
        self.discount_factor = discount_factor
        self.p = p
        self.num_local_candidates = num_local_candidates
        self.base_policy = base_policy
        self.ucb_beta = ucb_beta
        self.lsr_gamma = lsr_gamma
        self.reward_mode = reward_mode
        self.posterior_transform = posterior_transform
        self.maximize = maximize
        self.seed = seed
        self.set_X_pending(X_pending)

        if best_f is None:
            best_f_val = self._infer_best_f()
        else:
            best_f_val = torch.as_tensor(
                best_f, device=bounds_clean.device, dtype=bounds_clean.dtype
            )
        self.register_buffer("best_f", best_f_val)

        # Precompute deterministic QMC templates for fast, smooth rollout steps.
        local_template = self._build_local_template(
            d=bounds_clean.shape[1],
            n=num_local_candidates,
            p=p,
            dtype=bounds_clean.dtype,
            device=bounds_clean.device,
            seed=seed,
        )
        self.register_buffer("_local_template", local_template)

        n_global = max(32, num_local_candidates)
        global_grid = draw_sobol_samples(
            bounds=bounds_clean, n=n_global, q=1, seed=seed + 101
        ).squeeze(-2)
        self.register_buffer("_global_grid", global_grid)

        # Precompute standard normal QMC draws for continuation stages h >= 2.
        n_cont = max(1, horizon - 1)
        cont_normals = draw_sobol_normal_samples(
            d=n_cont * model.num_outputs,
            n=self.num_fantasies,
            device=bounds_clean.device,
            dtype=bounds_clean.dtype,
            seed=seed + 211,
        ).view(n_cont, self.num_fantasies, model.num_outputs)
        self.register_buffer("_continuation_normals", cont_normals)

    def _infer_best_f(self) -> Tensor:
        r"""Infer incumbent best objective from posterior mean at training inputs."""
        curr: Any = self.model
        if (
            not hasattr(curr, "train_inputs")
            and hasattr(curr, "models")
            and curr.models
        ):
            curr = curr.models[0]
        if getattr(curr, "_has_transformed_inputs", False) and hasattr(
            curr, "_original_train_inputs"
        ):
            train_X = curr._original_train_inputs
        elif hasattr(curr, "train_inputs") and curr.train_inputs:
            train_X = curr.train_inputs[0]
        else:
            return torch.zeros((), device=self.bounds.device, dtype=self.bounds.dtype)
        while train_X.ndim > 2:
            train_X = train_X[0]
        if train_X.shape[-2] == 0:
            return torch.zeros((), device=self.bounds.device, dtype=self.bounds.dtype)
        with torch.no_grad():
            post = self.model.posterior(
                train_X.to(device=self.bounds.device, dtype=self.bounds.dtype),
                posterior_transform=self.posterior_transform,
            )
            mean = post.mean.squeeze(-1)
            return mean.max() if self.maximize else mean.min()

    @staticmethod
    def _build_local_template(
        d: int,
        n: int,
        p: float,
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
    ) -> Tensor:
        r"""Build a deterministic ``n x d`` QMC template for local rollout search."""
        if math.isinf(p):
            # Template in [0, 1]^d interpolating [local_lb, local_ub].
            pts: list[Tensor] = [torch.full((1, d), 0.5, device=device, dtype=dtype)]
            if n >= 1 + 2 * d:
                eye = torch.eye(d, device=device, dtype=dtype)
                pts.append(0.5 + 0.5 * eye)
                pts.append(0.5 - 0.5 * eye)
            used = sum(t.shape[0] for t in pts)
            rem = n - used
            if rem > 0:
                unit_bounds = torch.zeros(2, d, device=device, dtype=dtype)
                unit_bounds[1] = 1.0
                sobol = draw_sobol_samples(
                    bounds=unit_bounds, n=rem, q=1, seed=seed
                ).squeeze(-2)
                pts.append(sobol)
            return torch.cat(pts, dim=0)[:n]

        # Template of offsets xi in B_p(0, 1)^d with ||xi||_p <= 1.
        pts = [torch.zeros((1, d), device=device, dtype=dtype)]
        if n >= 1 + 2 * d:
            eye = torch.eye(d, device=device, dtype=dtype)
            pts.append(eye)
            pts.append(-eye)
        used = sum(t.shape[0] for t in pts)
        rem = n - used
        if rem > 0:
            cube = torch.stack(
                [
                    -torch.ones(d, device=device, dtype=dtype),
                    torch.ones(d, device=device, dtype=dtype),
                ],
                dim=0,
            )
            dirs = draw_sobol_samples(bounds=cube, n=rem, q=1, seed=seed).squeeze(-2)
            norms = _scaled_lp_norm(z=dirs, p=p).clamp_min(1e-12).unsqueeze(-1)
            unit_dirs = dirs / norms
            n_boundary = rem // 2
            n_interior = rem - n_boundary
            radii_list: list[Tensor] = []
            if n_boundary > 0:
                radii_list.append(
                    torch.ones((n_boundary, 1), device=device, dtype=dtype)
                )
            if n_interior > 0:
                r_bounds = torch.zeros(2, 1, device=device, dtype=dtype)
                r_bounds[1] = 1.0
                r_int = draw_sobol_samples(
                    bounds=r_bounds, n=n_interior, q=1, seed=seed + 53
                ).squeeze(-2)
                radii_list.append(r_int.pow(1.0 / d))
            radii = torch.cat(radii_list, dim=0)
            pts.append(unit_dirs * radii)
        return torch.cat(pts, dim=0)[:n]

    def _generate_local_candidates(self, x_prev: Tensor) -> Tensor:
        r"""Generate ``M x (*batch) x d`` feasible local candidates near ``x_prev``."""
        lb = self.bounds[0].to(device=x_prev.device, dtype=x_prev.dtype)
        ub = self.bounds[1].to(device=x_prev.device, dtype=x_prev.dtype)
        delta_vec = self.delta.to(device=x_prev.device, dtype=x_prev.dtype)
        template = self._local_template.to(device=x_prev.device, dtype=x_prev.dtype)
        view_shape = (
            (template.shape[0],) + (1,) * (x_prev.ndim - 1) + (x_prev.shape[-1],)
        )
        tmpl = template.view(view_shape)

        x_in = x_prev.clamp(min=lb, max=ub)
        if math.isinf(self.p):
            local_lb = torch.maximum(lb, x_in - delta_vec)
            local_ub = torch.minimum(ub, x_in + delta_vec)
            return local_lb.unsqueeze(0) + tmpl * (local_ub - local_lb).unsqueeze(0)

        # Since x_in in [lb, ub] and ||tmpl||_p <= 1, clamping x_in + delta * tmpl
        # to [lb, ub] preserves ||(x_cand - x_in) / delta||_p <= ||tmpl||_p <= 1.
        return (x_in.unsqueeze(0) + delta_vec * tmpl).clamp(min=lb, max=ub)

    def _select_base_policy_step(
        self,
        curr_model: Model,
        X_cands: Tensor,
        running_best_obj: Tensor,
    ) -> Tensor:
        r"""Select smooth local rollout step inside ``T(x_{h-1}) n Theta``."""
        sign = 1.0 if self.maximize else -1.0
        tau = 0.05
        with torch.no_grad():
            post_cands = curr_model.posterior(
                X_cands.unsqueeze(-2),
                posterior_transform=self.posterior_transform,
            )
            mean_cands = post_cands.mean.squeeze(-1).squeeze(-1)
            sigma_cands = (
                post_cands.variance.clamp_min(1e-12).sqrt().squeeze(-1).squeeze(-1)
            )

            if self.base_policy == "ucb":
                raw_scores = sign * mean_cands + math.sqrt(self.ucb_beta) * sigma_cands
            else:
                u_local = (
                    sign * mean_cands - running_best_obj.unsqueeze(0)
                ) / sigma_cands
                log_ei_local = _log_ei_helper(u_local) + sigma_cands.log()

                if self.base_policy == "ei" or self.lsr_gamma <= 0.0:
                    raw_scores = log_ei_local
                else:
                    best_local_log_ei = log_ei_local.amax(dim=0)
                    g_grid = self._global_grid.to(
                        device=X_cands.device, dtype=X_cands.dtype
                    )
                    g_shape = (
                        (g_grid.shape[0],) + X_cands.shape[1:-1] + (1, g_grid.shape[-1])
                    )
                    g_eval = g_grid.view(
                        (g_grid.shape[0],)
                        + (1,) * (X_cands.ndim - 2)
                        + (1, g_grid.shape[-1])
                    ).expand(g_shape)
                    post_global = curr_model.posterior(
                        g_eval, posterior_transform=self.posterior_transform
                    )
                    mean_g = post_global.mean.squeeze(-1).squeeze(-1)
                    sigma_g = (
                        post_global.variance.clamp_min(1e-12)
                        .sqrt()
                        .squeeze(-1)
                        .squeeze(-1)
                    )
                    u_global = (sign * mean_g - running_best_obj.unsqueeze(0)) / sigma_g
                    log_ei_global = _log_ei_helper(u_global) + sigma_g.log()
                    best_global_log_ei, j_global = log_ei_global.max(dim=0)
                    x_global_best = g_grid[j_global]

                    if self.lsr_gamma >= 1.0:
                        use_local = best_local_log_ei >= best_global_log_ei
                    else:
                        use_local = (
                            best_local_log_ei - best_global_log_ei
                        ) >= math.log(self.lsr_gamma)

                    neg_dist_sq = -(
                        (X_cands - x_global_best.unsqueeze(0)).square().sum(dim=-1)
                    )
                    raw_scores = torch.where(
                        use_local.unsqueeze(0), log_ei_local, neg_dist_sq
                    )

            centered = raw_scores - raw_scores.amax(dim=0, keepdim=True)
            scale = raw_scores.std(dim=0, keepdim=True).clamp_min(1e-6)
            weights = torch.softmax(centered / (tau * scale), dim=0)

        return (weights.unsqueeze(-1) * X_cands).sum(dim=0)

    def _condition_trajectory_step(
        self,
        curr_model: Model,
        x_h: Tensor,
        stage_idx: int,
    ) -> Model:
        r"""Condition each trajectory in ``curr_model`` on 1 reparameterized fantasy."""
        obs_noise: Tensor | None = None
        if isinstance(curr_model.likelihood, FixedNoiseGaussianLikelihood):
            if curr_model.num_outputs > 1:
                noise_t = curr_model.likelihood.noise.transpose(-1, -2)
            else:
                noise_t = curr_model.likelihood.noise.unsqueeze(-1)
            obs_noise = noise_t.mean(dim=-2, keepdim=True)

        with fantasize_flag():
            post_noisy = curr_model.posterior(
                x_h,
                observation_noise=True if obs_noise is None else obs_noise,
            )
            z_raw = self._continuation_normals[stage_idx].to(
                device=x_h.device, dtype=x_h.dtype
            )
            extra_ones = (1,) * (post_noisy.mean.ndim - 3)
            z_view = z_raw.view(
                self.num_fantasies, *extra_ones, 1, curr_model.num_outputs
            )
            y_h = post_noisy.mean + post_noisy.variance.clamp_min(1e-12).sqrt() * z_view
            cond_kwargs: dict[str, Any] = {}
            if obs_noise is not None:
                cond_kwargs["noise"] = obs_noise.expand(y_h.shape)
            return curr_model.condition_on_observations(X=x_h, Y=y_h, **cond_kwargs)

    def _aggregate_fantasies(self, values: Tensor) -> Tensor:
        r"""Average or quadrature-weight trajectory values across dim 0."""
        weights = getattr(self.sampler, "base_weights", None)
        if weights is None:
            return values.mean(dim=0)
        w = weights.to(device=values.device, dtype=values.dtype).view(
            (-1,) + (1,) * (values.ndim - 1)
        )
        return (w * values).sum(dim=0)

    def _compute_terminal_mean(self, curr_model: Model, x_last: Tensor) -> Tensor:
        r"""Compute differentiable terminal max posterior mean per trajectory."""
        sign = 1.0 if self.maximize else -1.0
        X_local = self._generate_local_candidates(x_prev=x_last)
        g_grid = self._global_grid.to(device=x_last.device, dtype=x_last.dtype)
        g_expanded = g_grid.view(
            (g_grid.shape[0],) + (1,) * (x_last.ndim - 1) + (g_grid.shape[-1],)
        ).expand((g_grid.shape[0],) + x_last.shape)
        X_all = torch.cat([X_local, g_expanded], dim=0)

        with torch.no_grad():
            post_all = curr_model.posterior(
                X_all.unsqueeze(-2), posterior_transform=self.posterior_transform
            )
            means_all = sign * post_all.mean.squeeze(-1).squeeze(-1)
            j_best = means_all.argmax(dim=0)

        gather_idx = j_best.unsqueeze(0).unsqueeze(-1).expand((1,) + x_last.shape)
        x_term = torch.gather(X_all, dim=0, index=gather_idx).squeeze(0).unsqueeze(-2)
        post_term = curr_model.posterior(
            x_term, posterior_transform=self.posterior_transform
        )
        return sign * post_term.mean.squeeze(-1).squeeze(-1)

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the multi-step local-search rollout acquisition on ``X``.

        Args:
            X: Candidate tensor of shape ``(batch_shape) x 1 x d`` representing
                first-stage queries :math:`x_1 \in \mathcal{T}(x_0) \cap \Theta`.

        Returns:
            A ``(batch_shape)``-dim tensor of rollout acquisition values.
        """
        if self.X_pending is not None:
            y_pend = self.model.posterior(self.X_pending).mean.detach()
            pend_kwargs: dict[str, Any] = {}
            if isinstance(self.model.likelihood, FixedNoiseGaussianLikelihood):
                if self.model.num_outputs > 1:
                    noise_p = self.model.likelihood.noise.transpose(-1, -2)
                else:
                    noise_p = self.model.likelihood.noise.unsqueeze(-1)
                pend_kwargs["noise"] = noise_p.mean(dim=-2, keepdim=True).expand(
                    y_pend.shape
                )
            base_model = self.model.condition_on_observations(
                X=self.X_pending, Y=y_pend, **pend_kwargs
            )
        else:
            base_model = self.model

        orig_batch_shape = X.shape[:-2]
        n_batch = math.prod(orig_batch_shape)
        d = X.shape[-1]
        # Flatten t-batch (and omit singleton batch dim when n_batch == 1) to avoid
        # GPyTorch singleton-batch broadcasting issues on multi-output fantasy models.
        X_work = X.reshape(1, d) if n_batch == 1 else X.reshape(n_batch, 1, d)

        sign = 1.0 if self.maximize else -1.0
        best_f = self.best_f.to(device=X.device, dtype=X.dtype)

        # Stage 1 immediate Expected Improvement under D_0.
        post_1 = base_model.posterior(
            X_work, posterior_transform=self.posterior_transform
        )
        mean_1 = post_1.mean.squeeze(-1).squeeze(-1)
        sigma_1 = post_1.variance.clamp_min(1e-12).sqrt().squeeze(-1).squeeze(-1)
        u_1 = _scaled_improvement(
            mean=mean_1, sigma=sigma_1, best_f=best_f, maximize=self.maximize
        )
        ei_1 = _ei_helper(u_1) * sigma_1

        if self.horizon == 1 and self.reward_mode == "cumulative_ei":
            return ei_1.reshape(orig_batch_shape)

        need_grad = torch.is_grad_enabled() and X.requires_grad
        curr_model = base_model.fantasize(
            X=X_work, sampler=self.sampler, propagate_grads=need_grad
        )

        with settings.propagate_grads(need_grad):
            target_batch = curr_model.batch_shape
            x_prev = X_work.squeeze(-2).unsqueeze(0).expand(*target_batch, d)

            if self.horizon == 1 and self.reward_mode == "terminal_mean":
                term_vals = self._compute_terminal_mean(
                    curr_model=curr_model, x_last=x_prev
                )
                return self._aggregate_fantasies(term_vals).reshape(orig_batch_shape)

            post_x1_upd = curr_model.posterior(
                x_prev.unsqueeze(-2), posterior_transform=self.posterior_transform
            )
            mean_x1_upd = post_x1_upd.mean.squeeze(-1).squeeze(-1)
            running_best_obj = torch.maximum(sign * best_f, sign * mean_x1_upd)

            rollout_ei = torch.zeros_like(mean_x1_upd)
            for h in range(2, self.horizon + 1):
                X_cands = self._generate_local_candidates(x_prev=x_prev)
                x_h_vec = self._select_base_policy_step(
                    curr_model=curr_model,
                    X_cands=X_cands,
                    running_best_obj=running_best_obj,
                )
                x_h = x_h_vec.unsqueeze(-2)

                post_h = curr_model.posterior(
                    x_h, posterior_transform=self.posterior_transform
                )
                mean_h = post_h.mean.squeeze(-1).squeeze(-1)
                sigma_h = (
                    post_h.variance.clamp_min(1e-12).sqrt().squeeze(-1).squeeze(-1)
                )
                u_h = (sign * mean_h - running_best_obj) / sigma_h
                ei_h = _ei_helper(u_h) * sigma_h
                rollout_ei = rollout_ei + (self.discount_factor ** (h - 1)) * ei_h

                if h < self.horizon or self.reward_mode == "terminal_mean":
                    curr_model = self._condition_trajectory_step(
                        curr_model=curr_model, x_h=x_h, stage_idx=h - 2
                    )
                    if h < self.horizon:
                        post_upd = curr_model.posterior(
                            x_h, posterior_transform=self.posterior_transform
                        )
                        mean_upd = post_upd.mean.squeeze(-1).squeeze(-1)
                        running_best_obj = torch.maximum(
                            running_best_obj, sign * mean_upd
                        )
                x_prev = x_h_vec

            if self.reward_mode == "terminal_mean":
                term_vals = self._compute_terminal_mean(
                    curr_model=curr_model, x_last=x_prev
                )
                return self._aggregate_fantasies(term_vals).reshape(orig_batch_shape)

            return (ei_1 + self._aggregate_fantasies(rollout_ei)).reshape(
                orig_batch_shape
            )
