#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for run-order optimization of experimental designs (DoE).

When sequential experiments incur physical setup costs, thermal equilibration times,
or slew-rate limits between consecutive configurations, ordering a space-filling
design (such as Owen-scrambled Sobol sequences [Owen1995scrambled]_) can substantially
reduce total transition cost [Tack2001runorder]_, [Tack2004runorder]_,
[Garroi2009runorder]_ or worst-case single-step transition jumps via the Bottleneck
Traveling Salesman Problem (BTSP) [Garfinkel1978btsp]_. Fast non-iterative ordering
is also supported via Skilling's arbitrary-dimensional Hilbert space-filling curve
algorithm [Skilling2004hilbert]_.

References

.. [Garfinkel1978btsp]
    R. S. Garfinkel and K. C. Gilbert. The bottleneck traveling salesman problem:
    Algorithms and probabilistic analysis. Journal of the ACM, 25(3):435--448, 1978.

.. [Garroi2009runorder]
    J.-J. Garroi, P. Goos, and K. Sörensen. A variable-neighbourhood search algorithm
    for finding optimal run orders in the presence of serial correlation. Journal of
    Statistical Planning and Inference, 139(1):30--44, 2009.

.. [Owen1995scrambled]
    A. B. Owen. Randomly permuted (t,m,s)-nets and (t,s)-sequences. In H. Niederreiter
    and P. J.-S. Shiue (Eds.), Monte Carlo and Quasi-Monte Carlo Methods in Scientific
    Computing, pages 299--317. Springer, 1995.

.. [Skilling2004hilbert]
    J. Skilling. Programming the Hilbert curve. AIP Conference Proceedings,
    707:381--387, 2004.

.. [Tack2001runorder]
    L. Tack and M. Vandebroek. (D_t, C)-optimal run orders. Journal of Statistical
    Planning and Inference, 98(1-2):293--310, 2001.

.. [Tack2004runorder]
    L. Tack and M. Vandebroek. Budget constrained run orders in optimum design.
    Journal of Statistical Planning and Inference, 124(2):231--249, 2004.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from botorch.exceptions.errors import InfeasibilityError
from botorch.utils.transforms import normalize, unnormalize
from torch import Tensor
from torch.quasirandom import SobolEngine

__all__ = [
    "compute_transition_distance_matrix",
    "draw_ordered_sobol_samples",
    "hilbert_index",
    "order_samples",
    "order_samples_hilbert",
    "order_samples_tsp",
]


def _extract_2d_points(X: Tensor, name: str = "X") -> tuple[Tensor, bool]:
    r"""Validate ``(N, d)`` or ``(N, 1, d)`` input and return ``(X_2d, has_q_dim)``."""
    if X.ndim == 2:
        has_q_dim = False
        X_2d = X
    elif X.ndim == 3 and X.shape[1] == 1:
        has_q_dim = True
        X_2d = X.squeeze(1)
    else:
        raise ValueError(
            f"`{name}` must have shape `(N, d)` or `(N, 1, d)`, got {tuple(X.shape)}."
        )
    if X_2d.shape[-1] < 1:
        raise ValueError(f"`{name}` must have feature dimension `d >= 1`.")
    return X_2d, has_q_dim


def _validate_x0(x0: Tensor, d: int, ref: Tensor) -> Tensor:
    r"""Validate ``x0`` and return it shaped as ``(1, d)`` matching ``ref``."""
    if x0.ndim == 1 and x0.shape[0] == d:
        x0_2d = x0.unsqueeze(0)
    elif x0.ndim == 2 and x0.shape == (1, d):
        x0_2d = x0
    elif x0.ndim == 3 and x0.shape == (1, 1, d):
        x0_2d = x0.squeeze(1)
    else:
        raise ValueError(
            f"`x0` must have shape `({d},)`, `(1, {d})`, or `(1, 1, {d})`, "
            f"got {tuple(x0.shape)}."
        )
    x0_2d = x0_2d.to(device=ref.device, dtype=ref.dtype)
    if not torch.isfinite(x0_2d).all():
        raise ValueError("`x0` must contain only finite values.")
    return x0_2d


def compute_transition_distance_matrix(
    X: Tensor,
    Y: Tensor | None = None,
    weights: Tensor | None = None,
    metric: Tensor | None = None,
    p: float = 2.0,
) -> Tensor:
    r"""Compute pairwise transition distances between experimental configurations.

    Supports unweighted ``L_p`` norms, 1D non-negative feature weights ``weights``
    computing ``||(x - y) * sqrt(weights)||_p`` (which equals the diagonal Mahalanobis
    distance ``sqrt((x - y)^T diag(weights) (x - y))`` when ``p = 2``), or a 2D
    symmetric positive semi-definite matrix ``metric`` ``W`` computing the generalized
    Mahalanobis distance ``sqrt((x - y)^T W (x - y))`` (with ``p = 2``).

    Args:
        X: A ``(N, d)``-dim tensor of design points.
        Y: An optional ``(M, d)``-dim tensor of target design points. If ``None``,
            defaults to ``X``.
        weights: An optional ``(d,)``-dim tensor of non-negative per-feature weights
            with at least one strictly positive entry. Scales coordinate differences
            by ``sqrt(weights)`` prior to taking the ``L_p`` norm. Cannot be combined
            with ``metric``.
        metric: An optional ``(d, d)``-dim symmetric positive semi-definite matrix
            ``W`` defining distance ``sqrt((x - y)^T W (x - y))``. Requires ``p = 2.0``
            and cannot be combined with ``weights``.
        p: The norm order ``p >= 1.0`` (including ``float("inf")``).

    Returns:
        A ``(N, M)``-dim tensor of pairwise transition distances matching ``X.dtype``
        and ``X.device``.
    """
    if X.ndim != 2 or X.shape[-1] < 1:
        raise ValueError(
            f"`X` must be a 2D tensor of shape `(N, d)` with `d >= 1`, "
            f"got {tuple(X.shape)}."
        )
    d = X.shape[-1]
    same_xy = Y is None
    if same_xy:
        Y_eff_in = X
    else:
        if Y.ndim != 2 or Y.shape[-1] != d:
            raise ValueError(
                f"`Y` must be a 2D tensor with last dimension {d}, "
                f"got {tuple(Y.shape)}."
            )
        Y_eff_in = Y.to(device=X.device, dtype=X.dtype)

    if not (p >= 1.0):
        raise ValueError(f"`p` must satisfy `p >= 1.0`, got {p}.")
    if weights is not None and metric is not None:
        raise ValueError("Cannot specify both `weights` and `metric`.")

    factor: Tensor | None = None
    scale: Tensor | None = None
    if weights is not None:
        if weights.ndim != 1 or weights.shape[0] != d:
            raise ValueError(
                f"`weights` must have shape `({d},)`, got {tuple(weights.shape)}."
            )
        w = weights.to(device=X.device, dtype=X.dtype)
        if not torch.isfinite(w).all():
            raise ValueError("`weights` must contain only finite values.")
        if (w < 0).any() or not (w > 0).any():
            raise ValueError(
                "`weights` must be non-negative with at least one positive entry."
            )
        scale = w.sqrt()
    elif metric is not None:
        if p != 2.0:
            raise ValueError(f"`metric` is only supported with `p=2.0`, got `p={p}`.")
        if metric.ndim != 2 or metric.shape != (d, d):
            raise ValueError(
                f"`metric` must have shape `({d}, {d})`, got {tuple(metric.shape)}."
            )
        W = metric.to(device=X.device, dtype=X.dtype)
        if not torch.isfinite(W).all():
            raise ValueError("`metric` must contain only finite values.")
        base_tol = 1e-5 if W.dtype == torch.float32 else 1e-8
        w_scale = max(1.0, float(W.abs().max().item()))
        if not torch.allclose(W, W.mT, atol=base_tol * w_scale, rtol=base_tol):
            raise ValueError("`metric` must be symmetric.")
        W_sym = 0.5 * (W + W.mT)
        eigvals, eigvecs = torch.linalg.eigh(W_sym)
        eig_scale = max(1.0, float(eigvals.abs().max().item()))
        if float(eigvals.min().item()) < -base_tol * eig_scale:
            raise ValueError(
                "`metric` must be positive semi-definite (all eigenvalues >= 0)."
            )
        factor = eigvecs * eigvals.clamp(min=0.0).sqrt().unsqueeze(0)

    n, m = X.shape[0], Y_eff_in.shape[0]
    if n == 0 or m == 0:
        return torch.zeros((n, m), dtype=X.dtype, device=X.device)

    if scale is not None:
        X_proj = X * scale
        Y_proj = X_proj if same_xy else Y_eff_in * scale
    elif factor is not None:
        X_proj = X @ factor
        Y_proj = X_proj if same_xy else Y_eff_in @ factor
    else:
        X_proj = X
        Y_proj = X_proj if same_xy else Y_eff_in

    dist = torch.cdist(
        X_proj,
        Y_proj,
        p=p,
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    if same_xy:
        dist.fill_diagonal_(0.0)
    return dist


def _held_karp_sum(
    D_np: np.ndarray,
    d0_np: np.ndarray | None,
    popcounts: np.ndarray,
) -> list[int]:
    r"""Solve exact minimum-sum open Hamiltonian path via Held-Karp DP."""
    n = D_np.shape[0]
    num_masks = 1 << n
    dp = np.full((num_masks, n), np.inf, dtype=np.float64)
    parent = np.full((num_masks, n), -1, dtype=np.int32)

    for v in range(n):
        dp[1 << v, v] = 0.0 if d0_np is None else float(d0_np[v])

    for k in range(2, n + 1):
        masks_k = np.flatnonzero(popcounts == k)
        for v in range(n):
            bit_v = 1 << v
            has_v = masks_k[(masks_k & bit_v) != 0]
            prev_masks = has_v ^ bit_v
            cand = dp[prev_masks] + D_np[None, :, v]
            best_u = np.argmin(cand, axis=1)
            dp[has_v, v] = cand[np.arange(has_v.shape[0]), best_u]
            parent[has_v, v] = best_u

    full_mask = num_masks - 1
    cur_v = int(np.argmin(dp[full_mask]))
    cur_mask = full_mask
    rev_path = [cur_v]
    for _ in range(n - 1):
        prev_v = int(parent[cur_mask, cur_v])
        cur_mask ^= 1 << cur_v
        cur_v = prev_v
        rev_path.append(cur_v)
    rev_path.reverse()
    return rev_path


def _held_karp_open_tsp(
    D_np: np.ndarray,
    d0_np: np.ndarray | None,
    bottleneck: bool,
) -> list[int]:
    r"""Exact Held-Karp DP for sum-of-edges or lexicographic bottleneck open TSP."""
    n = D_np.shape[0]
    num_masks = 1 << n
    popcounts = np.zeros(num_masks, dtype=np.int32)
    for b in range(n):
        popcounts[1 << b : 1 << (b + 1)] = popcounts[: 1 << b] + 1

    if not bottleneck:
        return _held_karp_sum(D_np=D_np, d0_np=d0_np, popcounts=popcounts)

    # Pass 1: exact global minimum bottleneck edge weight B*.
    dp_bn = np.full((num_masks, n), np.inf, dtype=np.float64)
    for v in range(n):
        dp_bn[1 << v, v] = 0.0 if d0_np is None else float(d0_np[v])

    for k in range(2, n + 1):
        masks_k = np.flatnonzero(popcounts == k)
        for v in range(n):
            bit_v = 1 << v
            has_v = masks_k[(masks_k & bit_v) != 0]
            prev_masks = has_v ^ bit_v
            cand = np.maximum(dp_bn[prev_masks], D_np[None, :, v])
            dp_bn[has_v, v] = np.min(cand, axis=1)

    opt_bottleneck = float(np.min(dp_bn[num_masks - 1]))

    # Pass 2: among all paths with every edge <= B*, minimize total path length.
    D_restricted = np.where(D_np <= opt_bottleneck, D_np, np.inf)
    d0_restricted = (
        None if d0_np is None else np.where(d0_np <= opt_bottleneck, d0_np, np.inf)
    )
    return _held_karp_sum(D_np=D_restricted, d0_np=d0_restricted, popcounts=popcounts)


def _path_metrics(
    path: list[int],
    D_np: np.ndarray,
    d0_np: np.ndarray | None,
) -> tuple[float, float]:
    r"""Compute ``(max_jump, sum_length)`` for an open Hamiltonian path."""
    arr = np.asarray(path, dtype=np.intp)
    step_dists = D_np[arr[:-1], arr[1:]] if arr.size > 1 else np.empty(0)
    if d0_np is not None and arr.size > 0:
        first = float(d0_np[arr[0]])
        total = first + float(step_dists.sum())
        max_jump = max(first, float(step_dists.max())) if step_dists.size > 0 else first
    else:
        total = float(step_dists.sum())
        max_jump = float(step_dists.max()) if step_dists.size > 0 else 0.0
    return max_jump, total


def _greedy_nn_from_start(start: int, D_np: np.ndarray) -> list[int]:
    r"""Construct a greedy nearest-neighbor tour starting at ``start``."""
    n = D_np.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    cur = start
    for _ in range(n - 1):
        row = np.where(visited, np.inf, D_np[cur])
        nxt = int(np.argmin(row))
        visited[nxt] = True
        tour.append(nxt)
        cur = nxt
    return tour


def _prim_mst(D_np: np.ndarray) -> tuple[list[list[int]], float]:
    r"""Compute an MST in ``O(N^2)`` via Prim's algorithm and its max edge weight."""
    n = D_np.shape[0]
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    min_dist = D_np[0].copy()
    parent = np.zeros(n, dtype=np.int32)
    adj: list[list[int]] = [[] for _ in range(n)]
    max_edge = 0.0
    for _ in range(n - 1):
        cand = np.where(in_mst, np.inf, min_dist)
        v = int(np.argmin(cand))
        u = int(parent[v])
        w = float(cand[v])
        if w > max_edge:
            max_edge = w
        in_mst[v] = True
        adj[u].append(v)
        adj[v].append(u)
        better = (~in_mst) & (D_np[v] < min_dist)
        min_dist[better] = D_np[v, better]
        parent[better] = v
    return adj, max_edge


def _mst_preorder_tour(
    adj: list[list[int]],
    root: int,
    D_np: np.ndarray,
) -> list[int]:
    r"""Depth-first traversal of an MST visiting closer children first."""
    n = len(adj)
    visited = [False] * n
    tour: list[int] = []
    stack = [root]
    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        tour.append(u)
        children = [v for v in adj[u] if not visited[v]]
        children.sort(key=lambda v: D_np[u, v], reverse=True)
        stack.extend(children)
    return tour


def _find_threshold_path(
    D_np: np.ndarray,
    d0_np: np.ndarray | None,
    tau: float,
) -> list[int] | None:
    r"""Find a Hamiltonian path in threshold graph ``G(tau)`` via Posa rotation."""
    n = D_np.shape[0]
    adj = D_np <= tau
    np.fill_diagonal(adj, False)
    degrees = adj.sum(axis=1)
    if n >= 2 and np.any(degrees == 0):
        return None
    if int(np.sum(degrees == 1)) > 2:
        return None

    if d0_np is None:
        valid_starts = np.argsort(degrees)
    else:
        cand_starts = np.flatnonzero(d0_np <= tau)
        if cand_starts.size == 0:
            return None
        valid_starts = cand_starts[np.argsort(degrees[cand_starts])]

    for start_idx in valid_starts[: min(8, valid_starts.size)]:
        start = int(start_idx)
        path = [start]
        visited = np.zeros(n, dtype=bool)
        visited[start] = True

        rotations = 0
        max_rotations = 4 * n
        while True:
            # Phase A: Warnsdorff greedy extension (fewest remaining neighbors first).
            while len(path) < n:
                end = path[-1]
                unvis = np.flatnonzero(adj[end] & (~visited))
                if unvis.size == 0:
                    break
                rem_deg = adj[unvis][:, ~visited].sum(axis=1)
                order = np.lexsort((D_np[end, unvis], rem_deg))
                nxt = int(unvis[order[0]])
                path.append(nxt)
                visited[nxt] = True

            if len(path) == n:
                return path
            if rotations >= max_rotations:
                break

            # If unanchored and head can extend directly, flip whole path.
            if d0_np is None:
                head_unvis = np.flatnonzero(adj[path[0]] & (~visited))
                if head_unvis.size > 0:
                    path.reverse()
                    rotations += 1
                    continue

            # Phase B: Posa suffix rotation to expose an extensible endpoint.
            end = path[-1]
            pos = {node: idx for idx, node in enumerate(path)}
            in_nbrs = [
                pos[int(w)]
                for w in np.flatnonzero(adj[end])
                if int(w) in pos and pos[int(w)] < len(path) - 2
            ]
            if not in_nbrs:
                break

            advanced = False
            for idx in in_nbrs:
                new_end = path[idx + 1]
                if np.any(adj[new_end] & (~visited)):
                    path[idx + 1 :] = reversed(path[idx + 1 :])
                    rotations += 1
                    advanced = True
                    break
            if not advanced:
                idx = in_nbrs[rotations % len(in_nbrs)]
                path[idx + 1 :] = reversed(path[idx + 1 :])
                rotations += 1

    return None


def _two_opt_sum(
    path: list[int],
    D_np: np.ndarray,
    d0_np: np.ndarray | None,
    max_iter: int,
) -> list[int]:
    r"""Refine an open Hamiltonian path with 2-opt to minimize total path length."""
    n = len(path)
    if n <= 1 or max_iter <= 0:
        return list(path)

    p = list(path)
    _, cur_sum = _path_metrics(p, D_np, d0_np)

    for _ in range(max_iter):
        eps = 1e-12 * max(1.0, cur_sum)
        best_delta = -eps
        best_move: tuple[int, int] | None = None

        for i in range(n - 1):
            pi = p[i]
            old_left = (
                (0.0 if d0_np is None else float(d0_np[pi]))
                if i == 0
                else float(D_np[p[i - 1], pi])
            )
            pim1 = -1 if i == 0 else p[i - 1]
            for j in range(i + 1, n):
                pj = p[j]
                old_right = 0.0 if j == n - 1 else float(D_np[pj, p[j + 1]])
                new_left = (
                    (0.0 if d0_np is None else float(d0_np[pj]))
                    if i == 0
                    else float(D_np[pim1, pj])
                )
                new_right = 0.0 if j == n - 1 else float(D_np[pi, p[j + 1]])
                delta = (new_left + new_right) - (old_left + old_right)
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)

        if best_move is None:
            break
        bi, bj = best_move
        p[bi : bj + 1] = reversed(p[bi : bj + 1])
        cur_sum += best_delta

    return p


def _two_opt_bottleneck(
    path: list[int],
    D_np: np.ndarray,
    d0_np: np.ndarray | None,
    max_iter: int,
) -> list[int]:
    r"""Refine an open path with 2-opt minimizing ``(max_jump, sum_length)``."""
    n = len(path)
    if n <= 1 or max_iter <= 0:
        return list(path)

    p = list(path)
    for _ in range(max_iter):
        # Build edge arrays of length n + 1: slot 0 is start, 1..n-1 internal, n is end.
        max_edges = np.full(n + 1, -1.0, dtype=np.float64)
        sum_edges = np.zeros(n + 1, dtype=np.float64)
        if d0_np is not None:
            max_edges[0] = float(d0_np[p[0]])
            sum_edges[0] = max_edges[0]
        for k in range(1, n):
            w = float(D_np[p[k - 1], p[k]])
            max_edges[k] = w
            sum_edges[k] = w

        # Top-3 edge slots allow O(1) unchanged-max queries when removing 2 edges.
        top_order = np.argsort(max_edges)[::-1]
        idx1, idx2, idx3 = int(top_order[0]), int(top_order[1]), int(top_order[2])
        val1, val2, val3 = (
            float(max_edges[idx1]),
            float(max_edges[idx2]),
            float(max_edges[idx3]),
        )
        cur_max = max(0.0, val1)
        cur_sum = float(sum_edges.sum())
        eps = 1e-12 * max(1.0, cur_max, cur_sum)

        best_max = cur_max
        best_sum = cur_sum
        best_move: tuple[int, int] | None = None

        for i in range(n - 1):
            pi = p[i]
            pim1 = -1 if i == 0 else p[i - 1]
            old_sum_i = float(sum_edges[i])
            for j in range(i + 1, n):
                pj = p[j]
                rem_b = j + 1
                if idx1 != i and idx1 != rem_b:
                    unchanged_max = val1
                elif idx2 != i and idx2 != rem_b:
                    unchanged_max = val2
                else:
                    unchanged_max = val3

                if i == 0:
                    if d0_np is None:
                        new_m_left = -1.0
                        new_s_left = 0.0
                    else:
                        new_m_left = float(d0_np[pj])
                        new_s_left = new_m_left
                else:
                    new_m_left = float(D_np[pim1, pj])
                    new_s_left = new_m_left

                if j == n - 1:
                    new_m_right = -1.0
                    new_s_right = 0.0
                else:
                    new_m_right = float(D_np[pi, p[j + 1]])
                    new_s_right = new_m_right

                cand_max = max(0.0, unchanged_max, new_m_left, new_m_right)
                cand_sum = (
                    cur_sum
                    - old_sum_i
                    - float(sum_edges[rem_b])
                    + new_s_left
                    + new_s_right
                )
                if cand_max < best_max - eps or (
                    abs(cand_max - best_max) <= eps and cand_sum < best_sum - eps
                ):
                    best_max = cand_max
                    best_sum = cand_sum
                    best_move = (i, j)

        if best_move is None:
            break
        bi, bj = best_move
        p[bi : bj + 1] = reversed(p[bi : bj + 1])

    return p


def _heuristic_open_tsp(
    D_np: np.ndarray,
    d0_np: np.ndarray | None,
    bottleneck: bool,
    two_opt_max_iter: int,
) -> list[int]:
    r"""Multi-start NN (+ MST & threshold binary search for BTSP) + 2-opt."""
    n = D_np.shape[0]
    max_starts = min(n, 32)
    if d0_np is not None:
        starts = [int(s) for s in np.argsort(d0_np)[:max_starts]]
    elif n <= 64:
        starts = list(range(n))
    else:
        starts = [int(round(i * (n - 1) / (max_starts - 1))) for i in range(max_starts)]

    nn_tours = [list(range(n)), *(_greedy_nn_from_start(s, D_np) for s in starts)]

    if not bottleneck:
        nn_tours.sort(key=lambda t: _path_metrics(t, D_np, d0_np)[1])
        best_tour = nn_tours[0]
        best_cost = _path_metrics(best_tour, D_np, d0_np)[1]
        for cand in nn_tours[: min(4, len(nn_tours))]:
            refined = _two_opt_sum(cand, D_np, d0_np, max_iter=two_opt_max_iter)
            cost = _path_metrics(refined, D_np, d0_np)[1]
            if cost < best_cost:
                best_cost = cost
                best_tour = refined
        return best_tour

    # Bottleneck TSP: combine NN, sum-TSP, MST preorder, and threshold binary search
    sum_tour = _heuristic_open_tsp(
        D_np=D_np,
        d0_np=d0_np,
        bottleneck=False,
        two_opt_max_iter=two_opt_max_iter,
    )
    candidates: list[list[int]] = [list(range(n)), sum_tour]
    nn_tours.sort(key=lambda t: _path_metrics(t, D_np, d0_np))
    candidates.append(nn_tours[0])

    mst_adj, mst_max = _prim_mst(D_np)
    if d0_np is not None:
        mst_roots = [int(np.argmin(d0_np))]
    else:
        leaves = [u for u, nbrs in enumerate(mst_adj) if len(nbrs) <= 1]
        mst_roots = leaves[:4] if leaves else [0]
    mst_tours = [_mst_preorder_tour(mst_adj, r, D_np) for r in mst_roots]
    mst_tours.sort(key=lambda t: _path_metrics(t, D_np, d0_np))
    candidates.append(mst_tours[0])

    # Binary search over unique edge weights in [lower_bound, best_known_max]
    best_init_max = min(_path_metrics(t, D_np, d0_np)[0] for t in candidates)
    lb = mst_max if d0_np is None else max(mst_max, float(np.min(d0_np)))
    upper_tri = D_np[np.triu_indices(n, k=1)]
    all_weights = upper_tri if d0_np is None else np.concatenate([upper_tri, d0_np])
    taus = np.unique(all_weights[(all_weights >= lb) & (all_weights < best_init_max)])
    if taus.size > 0:
        lo_idx, hi_idx = 0, int(taus.size) - 1
        best_thresh_path: list[int] | None = None
        while lo_idx <= hi_idx:
            mid = (lo_idx + hi_idx) // 2
            found = _find_threshold_path(D_np, d0_np, float(taus[mid]))
            if found is not None:
                best_thresh_path = found
                hi_idx = mid - 1
            else:
                lo_idx = mid + 1
        if best_thresh_path is not None:
            candidates.append(best_thresh_path)

    best_tour = candidates[0]
    best_score = _path_metrics(best_tour, D_np, d0_np)
    for cand in candidates:
        refined = _two_opt_bottleneck(cand, D_np, d0_np, max_iter=two_opt_max_iter)
        score = _path_metrics(refined, D_np, d0_np)
        if score < best_score:
            best_score = score
            best_tour = refined
    return best_tour


def order_samples_tsp(
    X: Tensor,
    x0: Tensor | None = None,
    weights: Tensor | None = None,
    metric: Tensor | None = None,
    p: float = 2.0,
    bottleneck: bool = False,
    prepend_x0: bool = False,
    exact_max_n: int = 15,
    two_opt_max_iter: int = 200,
) -> tuple[Tensor, Tensor]:
    r"""Order design points along an open Hamiltonian path (TSP or Bottleneck TSP).

    Minimizes either the total cumulative transition distance (``bottleneck=False``)
    [Tack2001runorder]_, [Garroi2009runorder]_ or the maximum single-step transition
    jump (``bottleneck=True``, with total path length as exact tie-breaker)
    [Garfinkel1978btsp]_. For ``N <= exact_max_n``, solves for a globally optimal
    path via Held-Karp dynamic programming in ``O(N^2 2^N)`` time. For
    ``N > exact_max_n``, uses multi-start nearest-neighbor (augmented with MST and
    binary-search threshold reachability for ``bottleneck=True``) followed by 2-opt
    refinement.

    Args:
        X: A ``(N, d)`` or ``(N, 1, d)``-dim tensor of design points.
        x0: An optional initial state / anchor configuration of shape ``(d,)``,
            ``(1, d)``, or ``(1, 1, d)``. When provided, transition cost includes
            the initial step ``d(x0, X[indices[0]])``.
        weights: Optional ``(d,)``-dim non-negative feature weights passed to
            ``compute_transition_distance_matrix``.
        metric: Optional ``(d, d)``-dim symmetric PSD metric matrix passed to
            ``compute_transition_distance_matrix``.
        p: The ``L_p`` norm order ``p >= 1.0`` (default ``2.0``).
        bottleneck: If ``False``, minimize total path length. If ``True``, minimize
            the maximum single-step transition jump (breaking ties by total length).
        prepend_x0: If ``True`` (requires ``x0 is not None``), prepend ``x0`` at
            index 0 of ``X_ordered`` (giving ``N + 1`` rows) while ``indices`` still
            gives the ``(N,)`` permutation of ``X``.
        exact_max_n: Maximum ``N`` solved exactly via Held-Karp DP (default ``15``).
        two_opt_max_iter: Maximum number of 2-opt improvement passes for heuristic
            refinement when ``N > exact_max_n`` (default ``200``).

    Returns:
        A two-element tuple ``(X_ordered, indices)`` where ``X_ordered`` preserves
        the 2D/3D shape of ``X`` (with ``N + 1`` rows if ``prepend_x0=True``) and
        ``indices`` is a ``(N,)`` ``LongTensor`` on ``X.device`` permuting ``X``.
    """
    X_2d, has_q_dim = _extract_2d_points(X, name="X")
    n, d = X_2d.shape

    if exact_max_n < 0:
        raise ValueError(f"`exact_max_n` must be non-negative, got {exact_max_n}.")
    if two_opt_max_iter < 0:
        raise ValueError(
            f"`two_opt_max_iter` must be non-negative, got {two_opt_max_iter}."
        )
    if prepend_x0 and x0 is None:
        raise ValueError("`prepend_x0=True` requires `x0` to be provided.")

    x0_2d = _validate_x0(x0, d=d, ref=X_2d) if x0 is not None else None
    D = compute_transition_distance_matrix(X=X_2d, weights=weights, metric=metric, p=p)
    d0 = (
        compute_transition_distance_matrix(
            X=x0_2d, Y=X_2d, weights=weights, metric=metric, p=p
        )[0]
        if x0_2d is not None
        else None
    )

    if n <= 1:
        indices = torch.arange(n, dtype=torch.long, device=X.device)
    else:
        D_np = D.detach().cpu().to(dtype=torch.float64).numpy()
        d0_np = (
            None if d0 is None else d0.detach().cpu().to(dtype=torch.float64).numpy()
        )
        if n <= exact_max_n:
            perm_list = _held_karp_open_tsp(
                D_np=D_np, d0_np=d0_np, bottleneck=bottleneck
            )
        else:
            perm_list = _heuristic_open_tsp(
                D_np=D_np,
                d0_np=d0_np,
                bottleneck=bottleneck,
                two_opt_max_iter=two_opt_max_iter,
            )
        indices = torch.tensor(perm_list, dtype=torch.long, device=X.device)

    X_ordered = X[indices]
    if prepend_x0 and x0_2d is not None:
        x0_row = x0_2d.unsqueeze(1) if has_q_dim else x0_2d
        X_ordered = torch.cat([x0_row, X_ordered], dim=0)
    return X_ordered, indices


def _skilling_axes_to_transpose(coords: Tensor, num_bits: int) -> Tensor:
    r"""Transform quantized grid coordinates in-place to Skilling Hilbert transpose."""
    out = coords.clone()
    d = out.shape[-1]
    m_bit = 1 << (num_bits - 1)

    # Phase 1: inverse undo of subcube rotations/reflections
    q_bit = m_bit
    while q_bit > 1:
        p_mask = q_bit - 1
        for i in range(d):
            bit_set = (out[:, i] & q_bit) != 0
            t = torch.where(bit_set, p_mask, (out[:, 0] ^ out[:, i]) & p_mask)
            out[:, 0] ^= t
            out[:, i] ^= torch.where(bit_set, 0, t)
        q_bit >>= 1

    # Phase 2: Gray-encode across dimensions
    for i in range(1, d):
        out[:, i] ^= out[:, i - 1]

    t_glob = torch.zeros(out.shape[0], dtype=torch.int64, device=out.device)
    q_bit = m_bit
    while q_bit > 1:
        t_glob ^= torch.where((out[:, d - 1] & q_bit) != 0, q_bit - 1, 0)
        q_bit >>= 1
    out ^= t_glob.unsqueeze(-1)
    return out


def _compute_hilbert_chunks(
    X_2d: Tensor,
    bounds: Tensor | None,
    num_bits: int,
) -> Tensor:
    r"""Compute MSB-to-LSB 62-bit non-negative Hilbert key chunks ``(N, K)``."""
    if (
        isinstance(num_bits, bool)
        or not isinstance(num_bits, int)
        or not (1 <= num_bits <= 52)
    ):
        raise ValueError(f"`num_bits` must be an integer in `[1, 52]`, got {num_bits}.")

    n, d = X_2d.shape
    num_chunks = max(1, math.ceil((d * num_bits) / 62))
    if n == 0:
        return torch.empty((0, num_chunks), dtype=torch.int64, device=X_2d.device)

    X_f64 = X_2d.to(dtype=torch.float64)
    if bounds is not None:
        if bounds.ndim != 2 or bounds.shape != (2, d):
            raise ValueError(
                f"`bounds` must have shape `(2, {d})`, got {tuple(bounds.shape)}."
            )
        b_f64 = bounds.to(device=X_2d.device, dtype=torch.float64)
        if not torch.isfinite(b_f64).all() or (b_f64[1] < b_f64[0]).any():
            raise ValueError(
                "`bounds` must contain finite values with `bounds[1] >= bounds[0]`."
            )
        U = normalize(X_f64, b_f64, update_constant_bounds=True).clamp(0.0, 1.0)
    elif n == 1:
        U = torch.zeros_like(X_f64)
    else:
        lo = X_f64.min(dim=0).values
        hi = X_f64.max(dim=0).values
        U = normalize(
            X_f64, torch.stack([lo, hi], dim=0), update_constant_bounds=True
        ).clamp(0.0, 1.0)

    grid_max = (1 << num_bits) - 1
    coords = torch.floor(torch.ldexp(U, torch.tensor(num_bits))).to(torch.int64)
    coords.clamp_(0, grid_max)

    transpose = _skilling_axes_to_transpose(coords=coords, num_bits=num_bits)

    chunks: list[Tensor] = []
    acc = torch.zeros(n, dtype=torch.int64, device=X_2d.device)
    bits_in_acc = 0
    for bit in range(num_bits - 1, -1, -1):
        for axis in range(d):
            b_val = (transpose[:, axis] >> bit) & 1
            acc = (acc << 1) | b_val
            bits_in_acc += 1
            if bits_in_acc == 62:
                chunks.append(acc)
                acc = torch.zeros(n, dtype=torch.int64, device=X_2d.device)
                bits_in_acc = 0
    if bits_in_acc > 0 or not chunks:
        chunks.append(acc)
    return torch.stack(chunks, dim=-1)


def _lexsort_chunks(chunks: Tensor) -> Tensor:
    r"""Stable lexicographic sort of ``(N, K)`` non-negative int64 chunks."""
    n, k_chunks = chunks.shape
    perm = torch.arange(n, dtype=torch.long, device=chunks.device)
    if n <= 1:
        return perm
    for k in range(k_chunks - 1, -1, -1):
        step = torch.argsort(chunks[perm, k], dim=0, stable=True)
        perm = perm[step]
    return perm


def hilbert_index(
    X: Tensor,
    bounds: Tensor | None = None,
    num_bits: int = 16,
) -> Tensor:
    r"""Compute 1D Hilbert space-filling curve indices via Skilling's algorithm.

    Applies Skilling's (2004) ``AxesToTranspose`` transformation [Skilling2004hilbert]_
    and bit de-interleaving across ``num_bits`` dyadic grid levels in ``d`` dimensions.
    When ``d * num_bits <= 62``, returns the exact integer Hilbert curve index in
    ``[0, 2^(d * num_bits))``. When ``d * num_bits > 62`` (where a 64-bit signed
    integer would overflow), returns the exact dense lexicographic rank in ``[0, N)``
    across all ``d * num_bits`` Hilbert bits so no precision is lost.

    Args:
        X: A ``(N, d)`` or ``(N, 1, d)``-dim tensor of points.
        bounds: Optional ``(2, d)``-dim tensor of lower and upper box bounds. If
            ``None``, min-max bounds of ``X`` are used.
        num_bits: Number of bits per coordinate dimension (grid resolution ``2^b`` per
            axis, ``1 <= num_bits <= 52``, default ``16``).

    Returns:
        A ``(N,)``-dim ``torch.int64`` tensor of Hilbert indices (when
        ``d * num_bits <= 62``) or exact dense lexicographic Hilbert ranks (when
        ``d * num_bits > 62``) on ``X.device``.
    """
    X_2d, _ = _extract_2d_points(X, name="X")
    chunks = _compute_hilbert_chunks(X_2d=X_2d, bounds=bounds, num_bits=num_bits)
    if chunks.shape[1] == 1:
        return chunks[:, 0]

    n = X_2d.shape[0]
    perm = _lexsort_chunks(chunks)
    sorted_chunks = chunks[perm]
    is_new = torch.zeros(n, dtype=torch.int64, device=X.device)
    if n > 1:
        is_new[1:] = (
            (sorted_chunks[1:] != sorted_chunks[:-1]).any(dim=-1).to(torch.int64)
        )
    sorted_ranks = torch.cumsum(is_new, dim=0)
    ranks = torch.empty(n, dtype=torch.int64, device=X.device)
    ranks[perm] = sorted_ranks
    return ranks


def order_samples_hilbert(
    X: Tensor,
    bounds: Tensor | None = None,
    num_bits: int = 16,
) -> tuple[Tensor, Tensor]:
    r"""Order design points along Skilling's ``d``-dimensional Hilbert curve.

    Sorts points lexicographically across all ``d * num_bits`` Hilbert Gray-code bits
    [Skilling2004hilbert]_ without integer overflow in arbitrary dimension ``d >= 1``.

    Args:
        X: A ``(N, d)`` or ``(N, 1, d)``-dim tensor of design points.
        bounds: Optional ``(2, d)``-dim tensor of lower and upper box bounds. If
            ``None``, min-max bounds of ``X`` are used.
        num_bits: Number of bits per coordinate dimension (``1 <= num_bits <= 52``,
            default ``16``).

    Returns:
        A two-element tuple ``(X_ordered, indices)`` where ``X_ordered`` has the same
        shape as ``X`` and ``indices`` is a ``(N,)`` ``LongTensor`` on ``X.device``.
    """
    X_2d, _ = _extract_2d_points(X, name="X")
    chunks = _compute_hilbert_chunks(X_2d=X_2d, bounds=bounds, num_bits=num_bits)
    indices = _lexsort_chunks(chunks)
    return X[indices], indices


def order_samples(
    X: Tensor,
    method: str = "tsp",
    x0: Tensor | None = None,
    block_size: int | None = None,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    r"""Unified run-order optimization interface with optional nested block chaining.

    Supports TSP (``"tsp"``), Bottleneck TSP (``"bottleneck_tsp"``), and Hilbert
    curve (``"hilbert"``) sequencing. When ``block_size`` is specified, partitions
    ``X`` into consecutive blocks ``X[0:B], X[B:2B], ...`` to preserve QMC/Sobol
    prefix space-filling properties under early stopping while chaining each block
    from the final state of the preceding block.

    Args:
        X: A ``(N, d)`` or ``(N, 1, d)``-dim tensor of design points.
        method: Ordering strategy, one of ``"tsp"``, ``"bottleneck_tsp"``, or
            ``"hilbert"``.
        x0: Optional starting anchor point of shape ``(d,)``, ``(1, d)``, or
            ``(1, 1, d)``.
        block_size: Optional positive integer block size ``B`` for staged/nested
            run-order optimization.
        kwargs: Method-specific keyword arguments forwarded to ``order_samples_tsp``
            (e.g., ``weights``, ``metric``, ``p``, ``prepend_x0``, ``exact_max_n``,
            ``two_opt_max_iter``) or ``order_samples_hilbert`` (e.g., ``bounds``,
            ``num_bits``, plus optional ``prepend_x0``, ``weights``, ``metric``,
            ``p`` for anchor orientation).

    Returns:
        A two-element tuple ``(X_ordered, indices)`` where ``X_ordered`` preserves
        the 2D/3D shape of ``X`` (with ``N + 1`` rows if ``prepend_x0=True``) and
        ``indices`` is a ``(N,)`` ``LongTensor`` on ``X.device`` permuting ``X``.
    """
    valid_methods = {"tsp", "bottleneck_tsp", "hilbert"}
    if method not in valid_methods:
        raise ValueError(
            f"Unknown `method={method!r}`. Expected one of {sorted(valid_methods)}."
        )
    if block_size is not None and (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
    ):
        raise ValueError(
            f"`block_size` must be a positive integer, got {block_size!r}."
        )
    if "bottleneck" in kwargs:
        raise ValueError(
            "Select `method='bottleneck_tsp'` or `method='tsp'` instead of passing "
            "`bottleneck` to `order_samples`."
        )

    X_2d, has_q_dim = _extract_2d_points(X, name="X")
    n, d = X_2d.shape
    opts = dict(kwargs)
    prepend_x0 = bool(opts.pop("prepend_x0", False))
    if prepend_x0 and x0 is None:
        raise ValueError("`prepend_x0=True` requires `x0` to be provided.")
    x0_2d = _validate_x0(x0, d=d, ref=X_2d) if x0 is not None else None

    if block_size is None and method in {"tsp", "bottleneck_tsp"}:
        return order_samples_tsp(
            X=X,
            x0=x0_2d,
            bottleneck=(method == "bottleneck_tsp"),
            prepend_x0=prepend_x0,
            **opts,
        )

    eff_block = max(n, 1) if block_size is None else block_size

    if method == "hilbert":
        bounds = opts.pop("bounds", None)
        num_bits = opts.pop("num_bits", 16)
        weights = opts.pop("weights", None)
        metric = opts.pop("metric", None)
        p = opts.pop("p", 2.0)
        if opts:
            raise ValueError(
                f"Unexpected keyword arguments for `method='hilbert'`: {sorted(opts)}."
            )
        # Validate distance parameters upfront if provided or anchor exists
        if weights is not None or metric is not None or p != 2.0:
            compute_transition_distance_matrix(
                X=X_2d[:1], weights=weights, metric=metric, p=p
            )
        if bounds is None and n > 1:
            bounds = torch.stack(
                [X_2d.min(dim=0).values, X_2d.max(dim=0).values], dim=0
            )
        elif bounds is not None or n <= 1:
            # Validate bounds/num_bits even when n == 0
            _compute_hilbert_chunks(X_2d=X_2d[:1], bounds=bounds, num_bits=num_bits)

        if n == 0:
            indices = torch.empty(0, dtype=torch.long, device=X.device)
        else:
            all_idx: list[Tensor] = []
            anchor = x0_2d
            for start in range(0, n, eff_block):
                end = min(start + eff_block, n)
                X_blk = X_2d[start:end]
                _, blk_idx = order_samples_hilbert(
                    X=X_blk, bounds=bounds, num_bits=num_bits
                )
                if anchor is not None and blk_idx.numel() > 1:
                    endpoints = torch.stack(
                        [X_blk[blk_idx[0]], X_blk[blk_idx[-1]]], dim=0
                    )
                    d_end = compute_transition_distance_matrix(
                        X=anchor, Y=endpoints, weights=weights, metric=metric, p=p
                    )[0]
                    if float(d_end[1].item()) < float(d_end[0].item()):
                        blk_idx = blk_idx.flip(0)
                all_idx.append(blk_idx + start)
                anchor = X_blk[blk_idx[-1:]]
            indices = torch.cat(all_idx, dim=0)
    else:
        is_bn = method == "bottleneck_tsp"
        if n == 0:
            _, indices = order_samples_tsp(
                X=X, x0=x0_2d, bottleneck=is_bn, prepend_x0=False, **opts
            )
        else:
            all_idx = []
            anchor = x0_2d
            for start in range(0, n, eff_block):
                end = min(start + eff_block, n)
                X_blk = X_2d[start:end]
                _, blk_idx = order_samples_tsp(
                    X=X_blk,
                    x0=anchor,
                    bottleneck=is_bn,
                    prepend_x0=False,
                    **opts,
                )
                all_idx.append(blk_idx + start)
                anchor = X_blk[blk_idx[-1:]]
            indices = torch.cat(all_idx, dim=0)

    X_ordered = X[indices]
    if prepend_x0 and x0_2d is not None:
        x0_row = x0_2d.unsqueeze(1) if has_q_dim else x0_2d
        X_ordered = torch.cat([x0_row, X_ordered], dim=0)
    return X_ordered, indices


def _validate_linear_inequality_constraints(
    inequality_constraints: list[tuple[Tensor, Tensor, float]],
    d: int,
    ref: Tensor,
) -> list[tuple[Tensor, Tensor, float]]:
    r"""Validate intra-point linear constraints ``sum X[idx] * coef >= rhs``."""
    validated: list[tuple[Tensor, Tensor, float]] = []
    for i, constraint in enumerate(inequality_constraints):
        if not isinstance(constraint, tuple) or len(constraint) != 3:
            raise ValueError(
                f"Constraint {i} must be a 3-tuple `(indices, coefficients, rhs)`."
            )
        indices, coefficients, rhs = constraint
        if (
            indices.ndim != 1
            or indices.numel() < 1
            or indices.is_floating_point()
            or indices.is_complex()
            or indices.dtype == torch.bool
        ):
            raise ValueError(
                f"Constraint {i} `indices` must be a non-empty 1D integer tensor."
            )
        if coefficients.ndim != 1 or coefficients.shape != indices.shape:
            raise ValueError(
                f"Constraint {i} `coefficients` must be 1D matching `indices` shape "
                f"{tuple(indices.shape)}, got {tuple(coefficients.shape)}."
            )
        idx = indices.to(device=ref.device, dtype=torch.long)
        if int(idx.min().item()) < 0 or int(idx.max().item()) >= d:
            raise ValueError(
                f"Constraint {i} `indices` out of bounds for dimension `d={d}`."
            )
        coef = coefficients.to(device=ref.device, dtype=ref.dtype)
        validated.append((idx, coef, float(rhs)))
    return validated


def draw_ordered_sobol_samples(
    bounds: Tensor,
    n: int,
    q: int = 1,
    x0: Tensor | None = None,
    method: str = "tsp",
    block_size: int | None = None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    constraint_callable: Callable[[Tensor], Tensor] | None = None,
    seed: int | None = None,
    max_draws: int = 100000,
    **kwargs: Any,
) -> Tensor:
    r"""Draw feasible Owen-scrambled Sobol samples and optimize their run order.

    Draws ``n`` feasible candidates within ``bounds`` using an Owen-scrambled Sobol
    sequence [Owen1995scrambled]_, optionally rejection-filtering by BoTorch-style
    linear ``inequality_constraints`` and/or nonlinear ``constraint_callable(X) >= 0``,
    and sequences them via ``order_samples``.

    Args:
        bounds: A ``(2, d)``-dim tensor specifying lower and upper box bounds.
        n: The number of feasible sequential design points to draw.
        q: Batch size per sequential step. Must be ``1`` for run-order sequencing.
        x0: Optional starting anchor point of shape ``(d,)``, ``(1, d)``, or
            ``(1, 1, d)``.
        method: Run-order method passed to ``order_samples`` (``"tsp"``,
            ``"bottleneck_tsp"``, or ``"hilbert"``).
        block_size: Optional staged block size passed to ``order_samples``.
        inequality_constraints: Optional list of ``(indices, coefficients, rhs)``
            tuples encoding ``sum_j X[..., indices[j]] * coefficients[j] >= rhs``.
        constraint_callable: Optional callable accepting a ``(batch, d)`` tensor and
            returning a ``(batch,)`` or ``(batch, m)`` tensor where ``>= 0`` denotes
            feasibility.
        seed: Optional random seed for Owen scrambling of the Sobol sequence.
        max_draws: Maximum total Sobol draws attempted when rejection-sampling under
            constraints before raising ``InfeasibilityError`` (default ``100000``).
        kwargs: Additional keyword arguments forwarded to ``order_samples``.

    Returns:
        A ``(n, 1, d)``-dim tensor (or ``(n + 1, 1, d)`` if ``prepend_x0=True``) of
        ordered feasible Sobol samples matching ``bounds.dtype`` and ``bounds.device``.
    """
    if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] < 1:
        raise ValueError(
            f"`bounds` must have shape `(2, d)` with `d >= 1`, "
            f"got {tuple(bounds.shape)}."
        )
    if not torch.isfinite(bounds).all() or (bounds[1] < bounds[0]).any():
        raise ValueError(
            "`bounds` must contain finite values with `bounds[1] >= bounds[0]`."
        )
    if q != 1:
        raise ValueError(
            f"`draw_ordered_sobol_samples` requires `q=1` for run-order sequencing, "
            f"got `q={q}`."
        )
    if n < 0:
        raise ValueError(f"`n` must be non-negative, got {n}.")
    if max_draws < 1:
        raise ValueError(f"`max_draws` must be >= 1, got {max_draws}.")

    d = bounds.shape[1]
    lin_constraints = (
        _validate_linear_inequality_constraints(inequality_constraints, d=d, ref=bounds)
        if inequality_constraints is not None
        else []
    )

    if n == 0:
        empty = torch.empty((0, 1, d), dtype=bounds.dtype, device=bounds.device)
        ordered, _ = order_samples(
            X=empty, method=method, x0=x0, block_size=block_size, **kwargs
        )
        return ordered

    if n > max_draws:
        raise InfeasibilityError(
            f"Requested `n={n}` samples exceeds `max_draws={max_draws}`."
        )

    sobol_engine = SobolEngine(dimension=d, scramble=True, seed=seed)
    if not lin_constraints and constraint_callable is None:
        raw = sobol_engine.draw(n, dtype=bounds.dtype).to(device=bounds.device)
        X_feasible = unnormalize(raw, bounds, update_constant_bounds=False).unsqueeze(1)
    else:
        accepted: list[Tensor] = []
        num_accepted = 0
        total_drawn = 0
        while num_accepted < n and total_drawn < max_draws:
            batch_draws = min(max(4 * n, 256), max_draws - total_drawn)
            raw = sobol_engine.draw(batch_draws, dtype=bounds.dtype).to(
                device=bounds.device
            )
            X_cand = unnormalize(raw, bounds, update_constant_bounds=False)
            total_drawn += batch_draws

            is_feas = torch.ones(batch_draws, dtype=torch.bool, device=bounds.device)
            for idx, coef, rhs in lin_constraints:
                is_feas &= (X_cand[:, idx] * coef).sum(dim=-1) >= rhs

            if constraint_callable is not None and is_feas.any():
                c_val = constraint_callable(X_cand)
                if c_val.ndim == 0 or c_val.shape[0] != batch_draws:
                    raise ValueError(
                        "`constraint_callable` must return a tensor with leading "
                        f"batch dimension {batch_draws}, got {tuple(c_val.shape)}."
                    )
                c_feas = (
                    c_val >= 0
                    if c_val.ndim == 1
                    else (c_val >= 0).reshape(batch_draws, -1).all(dim=-1)
                )
                is_feas &= c_feas.to(device=bounds.device)

            if is_feas.any():
                kept = X_cand[is_feas]
                accepted.append(kept)
                num_accepted += kept.shape[0]

        if num_accepted < n:
            raise InfeasibilityError(
                f"Could only find {num_accepted} feasible Sobol samples out of {n} "
                f"requested within `max_draws={max_draws}`."
            )
        X_feasible = torch.cat(accepted, dim=0)[:n].unsqueeze(1)

    if method == "hilbert" and "bounds" not in kwargs:
        kwargs = {**kwargs, "bounds": bounds}
    X_ordered, _ = order_samples(
        X=X_feasible, method=method, x0=x0, block_size=block_size, **kwargs
    )
    return X_ordered
