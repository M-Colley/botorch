#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
from typing import Any

import torch
from botorch.exceptions.errors import InfeasibilityError
from botorch.utils.run_order import (
    compute_transition_distance_matrix,
    draw_ordered_sobol_samples,
    hilbert_index,
    order_samples,
    order_samples_hilbert,
    order_samples_tsp,
)
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.testing import BotorchTestCase
from torch import Tensor


def _brute_force_open_tsp(
    X: Tensor,
    x0: Tensor | None = None,
    weights: Tensor | None = None,
    metric: Tensor | None = None,
    p: float = 2.0,
    bottleneck: bool = False,
) -> tuple[float, float, tuple[int, ...]]:
    r"""Brute-force all ``N!`` permutations returning ``(max_jump, sum_len, perm)``."""
    D = compute_transition_distance_matrix(X=X, weights=weights, metric=metric, p=p).to(
        dtype=torch.float64
    )
    d0 = (
        compute_transition_distance_matrix(
            X=x0.reshape(1, -1), Y=X, weights=weights, metric=metric, p=p
        )[0].to(dtype=torch.float64)
        if x0 is not None
        else None
    )
    n = X.shape[0]
    best_key: tuple[float, float] | None = None
    best_perm: tuple[int, ...] = ()
    for perm in itertools.permutations(range(n)):
        steps = [float(D[perm[t], perm[t + 1]].item()) for t in range(n - 1)]
        if d0 is not None:
            steps = [float(d0[perm[0]].item()), *steps]
        max_j = max(steps) if steps else 0.0
        sum_l = sum(steps)
        key = (max_j, sum_l) if bottleneck else (sum_l, max_j)
        if best_key is None or key < best_key:
            best_key = key
            best_perm = perm
    assert best_key is not None
    if bottleneck:
        return best_key[0], best_key[1], best_perm
    return best_key[1], best_key[0], best_perm


def _eval_path(
    X_ordered_2d: Tensor,
    x0: Tensor | None = None,
    weights: Tensor | None = None,
    metric: Tensor | None = None,
    p: float = 2.0,
) -> tuple[float, float]:
    r"""Evaluate ``(max_jump, sum_len)`` of an ordered ``(N, d)`` sequence."""
    pts = X_ordered_2d
    if x0 is not None:
        pts = torch.cat([x0.reshape(1, -1).to(pts), pts], dim=0)
    if pts.shape[0] <= 1:
        return 0.0, 0.0
    D = compute_transition_distance_matrix(
        X=pts[:-1], Y=pts[1:], weights=weights, metric=metric, p=p
    ).diagonal()
    return float(D.max().item()), float(D.sum().item())


class TestRunOrder(BotorchTestCase):
    def test_compute_transition_distance_matrix(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            atol = 1e-5 if dtype == torch.float else 1e-10

            X = torch.tensor(
                [[0.0, 0.0, 1.0], [1.0, 2.0, 1.0], [3.0, -1.0, 0.5]], **tkwargs
            )
            Y = torch.tensor([[0.5, 1.0, 0.0], [-1.0, 0.0, 2.0]], **tkwargs)

            # Self-distance: zero diagonal, symmetric, matches Euclidean norm
            D_self = compute_transition_distance_matrix(X)
            self.assertEqual(D_self.shape, torch.Size([3, 3]))
            self.assertEqual(D_self.dtype, dtype)
            self.assertEqual(D_self.device.type, self.device.type)
            self.assertTrue(torch.equal(D_self.diagonal(), torch.zeros(3, **tkwargs)))
            self.assertAllClose(D_self, D_self.mT, atol=atol)

            # Cross-distance for p = 1, 2, 3, inf
            for p in (1.0, 2.0, 3.0, float("inf")):
                D_xy = compute_transition_distance_matrix(X, Y, p=p)
                expected = torch.linalg.vector_norm(
                    X.unsqueeze(1) - Y.unsqueeze(0), ord=p, dim=-1
                )
                self.assertAllClose(D_xy, expected, atol=atol)

            # 1D weights matches diagonal metric when p = 2, and supports 0 entry
            weights = torch.tensor([4.0, 0.25, 0.0], **tkwargs)
            D_w = compute_transition_distance_matrix(X, Y, weights=weights, p=2.0)
            D_diag = compute_transition_distance_matrix(
                X, Y, metric=torch.diag(weights), p=2.0
            )
            self.assertAllClose(D_w, D_diag, atol=atol)

            # Weighted L_1 and L_inf
            D_w_l1 = compute_transition_distance_matrix(X, Y, weights=weights, p=1.0)
            expected_w_l1 = (
                ((X.unsqueeze(1) - Y.unsqueeze(0)) * weights.sqrt()).abs().sum(dim=-1)
            )
            self.assertAllClose(D_w_l1, expected_w_l1, atol=atol)

            # Full-rank SPD and rank-deficient PSD 2D metric W
            A = torch.tensor(
                [[2.0, -0.5, 0.3], [0.1, 1.5, -0.4], [0.0, 0.2, 1.0]], **tkwargs
            )
            W_spd = A.mT @ A
            D_spd = compute_transition_distance_matrix(X, Y, metric=W_spd)
            diff = X.unsqueeze(1) - Y.unsqueeze(0)
            quad = torch.einsum("nmd,de,nme->nm", diff, W_spd, diff).clamp(min=0.0)
            self.assertAllClose(D_spd, quad.sqrt(), atol=atol)

            # Rank-1 PSD metric (zero eigenvalues present)
            u = torch.tensor([[1.0], [-2.0], [0.5]], **tkwargs)
            W_psd = u @ u.mT
            D_psd = compute_transition_distance_matrix(X, metric=W_psd)
            self.assertTrue(torch.equal(D_psd.diagonal(), torch.zeros(3, **tkwargs)))
            diff_xx = X.unsqueeze(1) - X.unsqueeze(0)
            quad_psd = torch.einsum("nmd,de,nme->nm", diff_xx, W_psd, diff_xx).clamp(
                min=0.0
            )
            self.assertAllClose(D_psd, quad_psd.sqrt(), atol=atol)

            # Empty inputs
            empty = torch.empty((0, 3), **tkwargs)
            self.assertEqual(
                compute_transition_distance_matrix(empty).shape, torch.Size([0, 0])
            )
            self.assertEqual(
                compute_transition_distance_matrix(X, empty).shape, torch.Size([3, 0])
            )

    def test_compute_transition_distance_matrix_validation(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        X = torch.rand(4, 3, **tkwargs)

        with self.assertRaisesRegex(ValueError, "2D tensor"):
            compute_transition_distance_matrix(torch.rand(4, **tkwargs))
        with self.assertRaisesRegex(ValueError, "d >= 1"):
            compute_transition_distance_matrix(torch.empty(4, 0, **tkwargs))
        with self.assertRaisesRegex(ValueError, "`Y` must be a 2D tensor"):
            compute_transition_distance_matrix(X, Y=torch.rand(4, 2, **tkwargs))
        with self.assertRaisesRegex(ValueError, "`p` must satisfy"):
            compute_transition_distance_matrix(X, p=0.5)
        with self.assertRaisesRegex(ValueError, "`p` must satisfy"):
            compute_transition_distance_matrix(X, p=float("nan"))
        with self.assertRaisesRegex(ValueError, "Cannot specify both"):
            compute_transition_distance_matrix(
                X, weights=torch.ones(3, **tkwargs), metric=torch.eye(3, **tkwargs)
            )
        with self.assertRaisesRegex(ValueError, "`weights` must have shape"):
            compute_transition_distance_matrix(X, weights=torch.ones(2, **tkwargs))
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_transition_distance_matrix(
                X, weights=torch.tensor([1.0, float("inf"), 1.0], **tkwargs)
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            compute_transition_distance_matrix(
                X, weights=torch.tensor([1.0, -0.1, 1.0], **tkwargs)
            )
        with self.assertRaisesRegex(ValueError, "at least one positive"):
            compute_transition_distance_matrix(X, weights=torch.zeros(3, **tkwargs))
        with self.assertRaisesRegex(ValueError, "only supported with `p=2.0`"):
            compute_transition_distance_matrix(X, metric=torch.eye(3, **tkwargs), p=1.0)
        with self.assertRaisesRegex(ValueError, "`metric` must have shape"):
            compute_transition_distance_matrix(X, metric=torch.eye(2, **tkwargs))
        with self.assertRaisesRegex(ValueError, "finite"):
            bad_w = torch.eye(3, **tkwargs)
            bad_w[0, 0] = float("nan")
            compute_transition_distance_matrix(X, metric=bad_w)
        with self.assertRaisesRegex(ValueError, "symmetric"):
            asym = torch.tensor(
                [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], **tkwargs
            )
            compute_transition_distance_matrix(X, metric=asym)
        with self.assertRaisesRegex(ValueError, "positive semi-definite"):
            indef = torch.diag(torch.tensor([1.0, -0.2, 1.0], **tkwargs))
            compute_transition_distance_matrix(X, metric=indef)

    def test_hilbert_curve_exact_grid_walks_and_high_dim(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}

            # 1. Exact 2D 4x4 Skilling Hilbert curve (num_bits=2, 16 cells)
            axis_4 = torch.arange(4, **tkwargs)
            grid_2d = torch.cartesian_prod(axis_4, axis_4)
            # Permute input arbitrarily to verify sorting recovers exact walk
            perm_in = torch.tensor(
                [7, 0, 15, 3, 11, 5, 13, 1, 9, 2, 14, 6, 12, 4, 8, 10],
                device=self.device,
                dtype=torch.long,
            )
            shuffled_2d = grid_2d[perm_in]
            h_2d = hilbert_index(shuffled_2d, num_bits=2)
            self.assertEqual(h_2d.dtype, torch.int64)
            self.assertEqual(h_2d.device.type, self.device.type)
            self.assertEqual(sorted(h_2d.tolist()), list(range(16)))

            walk_2d, idx_2d = order_samples_hilbert(shuffled_2d, num_bits=2)
            self.assertEqual(walk_2d.dtype, dtype)
            self.assertEqual(walk_2d.shape, torch.Size([16, 2]))
            self.assertTrue(torch.equal(walk_2d, shuffled_2d[idx_2d]))
            steps_2d = (walk_2d[1:] - walk_2d[:-1]).abs().sum(dim=-1)
            self.assertAllClose(steps_2d, torch.ones(15, **tkwargs), atol=1e-6)

            # 2. Exact 3D 2x2x2 (num_bits=1) and 4x4x4 (num_bits=2) Hamiltonian walks
            for num_bits in (1, 2):
                side = 1 << num_bits
                ax = torch.arange(side, **tkwargs)
                grid_3d = torch.cartesian_prod(ax, ax, ax)
                h_3d = hilbert_index(grid_3d, num_bits=num_bits)
                self.assertEqual(sorted(h_3d.tolist()), list(range(side**3)))
                walk_3d, _ = order_samples_hilbert(grid_3d, num_bits=num_bits)
                steps_3d = (walk_3d[1:] - walk_3d[:-1]).abs().sum(dim=-1)
                self.assertAllClose(
                    steps_3d, torch.ones(side**3 - 1, **tkwargs), atol=1e-6
                )

            # 3. 1D monotonicity check (d=1)
            x_1d = torch.tensor([[0.8], [0.1], [0.5], [0.3]], **tkwargs)
            ordered_1d, _ = order_samples_hilbert(x_1d, num_bits=10)
            self.assertTrue(torch.all(ordered_1d[1:, 0] >= ordered_1d[:-1, 0]).item())

            # 4. High-dimensional overflow-free exact sorting (d=12 & d=20, num_bits=16)
            for d_hi in (12, 20):
                bounds = torch.stack(
                    [torch.zeros(d_hi, **tkwargs), torch.ones(d_hi, **tkwargs)]
                )
                X_hi = draw_sobol_samples(bounds=bounds, n=32, q=1, seed=123)
                # Duplicate row 0 and add tiny low-bit perturbation to row 1
                X_hi_mod = X_hi.clone()
                X_hi_mod[5] = X_hi_mod[0]
                X_hi_mod[6] = (X_hi_mod[0] + 2.0e-5).clamp(0.0, 1.0)

                ranks = hilbert_index(X_hi_mod, bounds=bounds, num_bits=16)
                self.assertEqual(ranks.dtype, torch.int64)
                self.assertEqual(ranks.shape, torch.Size([32]))
                self.assertTrue((ranks >= 0).all().item())
                # Identical points have equal rank; 2e-5 (> 2^-16) differs at bit 15
                self.assertEqual(int(ranks[0].item()), int(ranks[5].item()))
                self.assertNotEqual(int(ranks[0].item()), int(ranks[6].item()))

                X_ord, idx_hi = order_samples_hilbert(
                    X_hi_mod, bounds=bounds, num_bits=16
                )
                self.assertEqual(X_ord.shape, torch.Size([32, 1, d_hi]))
                self.assertTrue(torch.equal(idx_hi, torch.argsort(ranks, stable=True)))

            # 5. Edge cases: N=0, N=1, constant bounds
            empty = torch.empty((0, 4), **tkwargs)
            self.assertEqual(hilbert_index(empty).shape, torch.Size([0]))
            ord_empty, idx_empty = order_samples_hilbert(empty)
            self.assertEqual(ord_empty.shape, torch.Size([0, 4]))
            self.assertEqual(idx_empty.shape, torch.Size([0]))

            single = torch.tensor([[0.75, 0.75]], **tkwargs)
            b2 = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
            self.assertEqual(int(hilbert_index(single, bounds=b2, num_bits=2)[0]), 10)
            self.assertEqual(int(hilbert_index(single, num_bits=2)[0]), 0)

        # Validation errors
        X_ok = torch.rand(5, 2, device=self.device, dtype=torch.double)
        with self.assertRaisesRegex(ValueError, "`num_bits` must be an integer"):
            hilbert_index(X_ok, num_bits=0)
        with self.assertRaisesRegex(ValueError, "`num_bits` must be an integer"):
            hilbert_index(X_ok, num_bits=53)
        with self.assertRaisesRegex(ValueError, "`num_bits` must be an integer"):
            hilbert_index(X_ok, num_bits=True)
        with self.assertRaisesRegex(ValueError, "`bounds` must have shape"):
            hilbert_index(X_ok, bounds=torch.zeros(2, 3, device=self.device))
        with self.assertRaisesRegex(ValueError, "`bounds` must contain finite"):
            bad_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=self.device)
            hilbert_index(X_ok, bounds=bad_b)

    def test_order_samples_tsp_exact_against_brute_force(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            atol = 1e-5 if dtype == torch.float else 1e-10

            for seed in (7, 42):
                gen = torch.Generator(device="cpu").manual_seed(seed)
                X = torch.rand((6, 3), generator=gen).to(**tkwargs)
                x0 = torch.rand((3,), generator=gen).to(**tkwargs)
                weights = torch.tensor([3.0, 0.5, 2.0], **tkwargs)
                A = torch.tensor(
                    [[1.5, 0.4, -0.2], [0.4, 1.2, 0.3], [-0.2, 0.3, 0.8]], **tkwargs
                )
                W = A @ A.mT

                configs = [
                    {"x0": None, "weights": None, "metric": None, "p": 2.0},
                    {"x0": x0, "weights": None, "metric": None, "p": 2.0},
                    {"x0": x0, "weights": weights, "metric": None, "p": 1.0},
                    {"x0": None, "weights": None, "metric": W, "p": 2.0},
                ]
                for cfg in configs:
                    for bottleneck in (False, True):
                        bf_max, bf_sum, _ = _brute_force_open_tsp(
                            X=X, bottleneck=bottleneck, **cfg
                        )
                        X_ord, indices = order_samples_tsp(
                            X=X, bottleneck=bottleneck, **cfg
                        )
                        self.assertEqual(X_ord.shape, X.shape)
                        self.assertEqual(X_ord.dtype, dtype)
                        self.assertEqual(indices.dtype, torch.long)
                        self.assertEqual(sorted(indices.tolist()), list(range(6)))
                        self.assertTrue(torch.equal(X_ord, X[indices]))

                        got_max, got_sum = _eval_path(X_ord, **cfg)
                        if bottleneck:
                            self.assertAlmostEqual(got_max, bf_max, delta=atol)
                            self.assertAlmostEqual(got_sum, bf_sum, delta=atol)
                        else:
                            self.assertAlmostEqual(got_sum, bf_sum, delta=atol)

                # Test 3D `(N, 1, d)` shape preservation and `prepend_x0=True`
                X_3d = X.unsqueeze(1)
                X_prep, idx_prep = order_samples_tsp(
                    X=X_3d, x0=x0.view(1, 1, 3), prepend_x0=True
                )
                self.assertEqual(X_prep.shape, torch.Size([7, 1, 3]))
                self.assertEqual(idx_prep.shape, torch.Size([6]))
                self.assertAllClose(X_prep[0, 0], x0, atol=atol)
                self.assertAllClose(X_prep[1:], X_3d[idx_prep], atol=atol)

        # Explicit bottleneck tie-breaker test:
        # From x0=(0, 0), two paths both incur bottleneck jump = 10.0 to cross from
        # x=0 cluster to x=10 cluster, but ordering within clusters differs in sum!
        tkwargs64 = {"device": self.device, "dtype": torch.double}
        x0_tie = torch.tensor([0.0, 0.0], **tkwargs64)
        X_tie = torch.tensor(
            [[0.0, 2.0], [0.0, 1.0], [10.0, 1.0], [10.0, 3.0]], **tkwargs64
        )
        bf_max, bf_sum, _ = _brute_force_open_tsp(X_tie, x0=x0_tie, bottleneck=True)
        X_tie_ord, _ = order_samples_tsp(X_tie, x0=x0_tie, bottleneck=True)
        got_max, got_sum = _eval_path(X_tie_ord, x0=x0_tie)
        self.assertAlmostEqual(got_max, bf_max, delta=1e-12)
        self.assertAlmostEqual(got_sum, bf_sum, delta=1e-12)

        # Trivial N=0, N=1, N=2 and validation errors
        for n_triv in (0, 1, 2):
            X_triv = torch.rand(n_triv, 2, **tkwargs64)
            Xo, idx = order_samples_tsp(X_triv, x0=x0_tie, prepend_x0=True)
            self.assertEqual(Xo.shape, torch.Size([n_triv + 1, 2]))
            self.assertEqual(idx.shape, torch.Size([n_triv]))

        with self.assertRaisesRegex(ValueError, "`prepend_x0=True` requires `x0`"):
            order_samples_tsp(X_tie, prepend_x0=True)
        with self.assertRaisesRegex(ValueError, "`exact_max_n` must be non-negative"):
            order_samples_tsp(X_tie, exact_max_n=-1)
        with self.assertRaisesRegex(ValueError, "`two_opt_max_iter`"):
            order_samples_tsp(X_tie, two_opt_max_iter=-1)
        with self.assertRaisesRegex(ValueError, "`x0` must have shape"):
            order_samples_tsp(X_tie, x0=torch.zeros(5, **tkwargs64))
        with self.assertRaisesRegex(ValueError, "`x0` must contain only finite"):
            order_samples_tsp(X_tie, x0=torch.tensor([0.0, float("nan")], **tkwargs64))
        with self.assertRaisesRegex(ValueError, "must have shape"):
            order_samples_tsp(torch.zeros(4, 2, 2, **tkwargs64))

    def test_order_samples_tsp_heuristic(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.stack([torch.zeros(3, **tkwargs), torch.ones(3, **tkwargs)])
            X_sobol = draw_sobol_samples(bounds=bounds, n=25, q=1, seed=2025).squeeze(1)
            x0 = torch.tensor([0.5, 0.5, 0.5], **tkwargs)

            for anchor in (None, x0):
                raw_max, raw_sum = _eval_path(X_sobol, x0=anchor)

                X_tsp, idx_tsp = order_samples_tsp(X_sobol, x0=anchor)
                tsp_max, tsp_sum = _eval_path(X_tsp, x0=anchor)
                self.assertEqual(sorted(idx_tsp.tolist()), list(range(25)))
                self.assertLess(tsp_sum, raw_sum * 0.85)

                X_btsp, idx_btsp = order_samples_tsp(
                    X_sobol, x0=anchor, bottleneck=True
                )
                btsp_max, _ = _eval_path(X_btsp, x0=anchor)
                self.assertEqual(sorted(idx_btsp.tolist()), list(range(25)))
                eps = 1e-5 if dtype == torch.float else 1e-10
                self.assertLessEqual(btsp_max, raw_max + eps)
                self.assertLessEqual(btsp_max, tsp_max + eps)

    def test_order_samples_and_nested_blocks(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.stack([torch.zeros(3, **tkwargs), torch.ones(3, **tkwargs)])
            X = draw_sobol_samples(bounds=bounds, n=20, q=1, seed=99)
            x0 = torch.zeros(3, **tkwargs)

            for method in ("tsp", "bottleneck_tsp", "hilbert"):
                X_ord, indices = order_samples(
                    X=X,
                    method=method,
                    x0=x0,
                    block_size=8,
                    prepend_x0=True,
                )
                self.assertEqual(X_ord.shape, torch.Size([21, 1, 3]))
                self.assertEqual(indices.shape, torch.Size([20]))
                # Verify exact Sobol prefix subsets at every block boundary
                self.assertEqual(set(indices[:8].tolist()), set(range(8)))
                self.assertEqual(set(indices[8:16].tolist()), set(range(8, 16)))
                self.assertEqual(set(indices[16:20].tolist()), set(range(16, 20)))
                self.assertAllClose(X_ord[0, 0], x0)
                self.assertAllClose(X_ord[1:], X[indices])

                # Within-block TSP ordering reduces total distance vs raw block order
                if method == "tsp":
                    _, raw_sum = _eval_path(X.squeeze(1), x0=x0)
                    _, blk_sum = _eval_path(X[indices].squeeze(1), x0=x0)
                    self.assertLess(blk_sum, raw_sum)

        # Verify Hilbert block reversal when block end is closer to anchor than start
        tkwargs64 = {"device": self.device, "dtype": torch.double}
        X_chain = torch.tensor(
            [[0.0, 0.0], [0.9, 0.0], [0.1, 0.0], [0.8, 0.0]], **tkwargs64
        )
        # Block 0 ends at [0.9, 0.0]; block 1 has [0.1, 0.0] and [0.8, 0.0], so
        # block 1 should reverse to visit [0.8, 0.0] before [0.1, 0.0]!
        X_ch_ord, idx_ch = order_samples(
            X_chain,
            method="hilbert",
            block_size=2,
            x0=torch.zeros(2, **tkwargs64),
            weights=torch.tensor([2.0, 1.0], **tkwargs64),
        )
        self.assertEqual(idx_ch.tolist(), [0, 1, 3, 2])
        self.assertTrue(torch.equal(X_ch_ord, X_chain[idx_ch]))

        # Empty N=0 with block_size across methods + N=70 strided starts & max_iter=0
        X_empty = torch.empty((0, 2), **tkwargs64)
        for m_empty in ("tsp", "hilbert"):
            eo, ei = order_samples(X_empty, method=m_empty, block_size=4)
            self.assertEqual(eo.shape, torch.Size([0, 2]))
            self.assertEqual(ei.shape, torch.Size([0]))

        b70 = torch.stack([torch.zeros(2, **tkwargs64), torch.ones(2, **tkwargs64)])
        X_70 = draw_sobol_samples(bounds=b70, n=70, q=1, seed=5).squeeze(1)
        X_70_ord, idx_70 = order_samples_tsp(X_70, two_opt_max_iter=0)
        self.assertEqual(X_70_ord.shape, torch.Size([70, 2]))
        self.assertEqual(sorted(idx_70.tolist()), list(range(70)))
        X_70_bn, _ = order_samples_tsp(X_70[:20], bottleneck=True, two_opt_max_iter=0)
        self.assertEqual(X_70_bn.shape, torch.Size([20, 2]))

        with self.assertRaisesRegex(ValueError, "d >= 1"):
            order_samples_tsp(torch.empty((3, 0), **tkwargs64))

        # Validation errors on order_samples
        with self.assertRaisesRegex(ValueError, "Unknown `method="):
            order_samples(X_chain, method="invalid")
        with self.assertRaisesRegex(ValueError, "`block_size` must be a positive"):
            order_samples(X_chain, block_size=0)
        with self.assertRaisesRegex(ValueError, "`block_size` must be a positive"):
            order_samples(X_chain, block_size=True)
        with self.assertRaisesRegex(ValueError, "Select `method='bottleneck_tsp'`"):
            order_samples(X_chain, method="tsp", bottleneck=True)
        with self.assertRaisesRegex(ValueError, "Unexpected keyword arguments"):
            order_samples(X_chain, method="hilbert", unknown_kw=1)
        with self.assertRaisesRegex(ValueError, "`prepend_x0=True` requires `x0`"):
            order_samples(X_chain, method="hilbert", prepend_x0=True)

    def test_draw_ordered_sobol_samples(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.tensor([[0.0, 0.0, -1.0], [1.0, 2.0, 1.0]], **tkwargs)
            x0 = torch.tensor([0.2, 0.3, 0.0], **tkwargs)

            # 1. Unconstrained reproducible Sobol draws across methods
            for method in ("tsp", "bottleneck_tsp", "hilbert"):
                s1 = draw_ordered_sobol_samples(
                    bounds=bounds, n=12, method=method, x0=x0, seed=42
                )
                s2 = draw_ordered_sobol_samples(
                    bounds=bounds, n=12, method=method, x0=x0, seed=42
                )
                self.assertEqual(s1.shape, torch.Size([12, 1, 3]))
                self.assertEqual(s1.dtype, dtype)
                self.assertEqual(s1.device.type, self.device.type)
                self.assertTrue((s1 >= bounds[0]).all().item())
                self.assertTrue((s1 <= bounds[1]).all().item())
                self.assertAllClose(s1, s2)

            # 2. Linear inequality_constraints + nonlinear constraint_callable
            # Constraint 1: x_0 + 0.5 * x_1 >= 0.8
            # Constraint 2 (callable returning 2 cols): x_2 <= 0.5 and x_0^2 + x_2 >= 0
            ineq = [
                (
                    torch.tensor([0, 1], device=self.device, dtype=torch.long),
                    torch.tensor([1.0, 0.5], **tkwargs),
                    0.8,
                )
            ]

            def nl_con(X_c: Tensor) -> Tensor:
                c1 = 0.5 - X_c[:, 2]
                c2 = X_c[:, 0].square() + X_c[:, 2]
                return torch.stack([c1, c2], dim=-1)

            s_con = draw_ordered_sobol_samples(
                bounds=bounds,
                n=14,
                x0=x0,
                method="tsp",
                block_size=7,
                inequality_constraints=ineq,
                constraint_callable=nl_con,
                seed=77,
                prepend_x0=True,
            )
            self.assertEqual(s_con.shape, torch.Size([15, 1, 3]))
            s_feas = s_con[1:, 0, :]
            self.assertTrue(
                ((s_feas[:, [0, 1]] * ineq[0][1]).sum(dim=-1) >= 0.8).all().item()
            )
            self.assertTrue((nl_con(s_feas) >= 0.0).all().item())

            # 1D constraint_callable + n=0 edge case
            s_1d_con = draw_ordered_sobol_samples(
                bounds=bounds,
                n=6,
                constraint_callable=lambda X_c: X_c[:, 0] - 0.3,
                seed=11,
            )
            self.assertTrue((s_1d_con[:, 0, 0] >= 0.3).all().item())

            s_zero = draw_ordered_sobol_samples(bounds=bounds, n=0)
            self.assertEqual(s_zero.shape, torch.Size([0, 1, 3]))

        # 3. InfeasibilityError & validation branches
        bounds64 = torch.stack(
            [
                torch.zeros(2, device=self.device, dtype=torch.double),
                torch.ones(2, device=self.device, dtype=torch.double),
            ]
        )
        impossible_ineq = [
            (
                torch.tensor([0], device=self.device, dtype=torch.long),
                torch.tensor([1.0], device=self.device, dtype=torch.double),
                5.0,
            )
        ]
        with self.assertRaises(InfeasibilityError):
            draw_ordered_sobol_samples(
                bounds=bounds64,
                n=4,
                inequality_constraints=impossible_ineq,
                max_draws=512,
                seed=1,
            )
        with self.assertRaises(InfeasibilityError):
            draw_ordered_sobol_samples(bounds=bounds64, n=10, max_draws=5)

        with self.assertRaisesRegex(ValueError, "`bounds` must have shape"):
            draw_ordered_sobol_samples(
                bounds=torch.zeros(3, 2, device=self.device), n=4
            )
        with self.assertRaisesRegex(ValueError, "`bounds` must contain finite"):
            draw_ordered_sobol_samples(
                bounds=torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=self.device), n=4
            )
        with self.assertRaisesRegex(ValueError, "requires `q=1`"):
            draw_ordered_sobol_samples(bounds=bounds64, n=4, q=2)
        with self.assertRaisesRegex(ValueError, "`n` must be non-negative"):
            draw_ordered_sobol_samples(bounds=bounds64, n=-1)
        with self.assertRaisesRegex(ValueError, "`max_draws` must be >= 1"):
            draw_ordered_sobol_samples(bounds=bounds64, n=1, max_draws=0)
        with self.assertRaisesRegex(ValueError, "must be a 3-tuple"):
            bad_con: Any = [(torch.tensor([0]),)]
            draw_ordered_sobol_samples(
                bounds=bounds64, n=2, inequality_constraints=bad_con
            )
        with self.assertRaisesRegex(ValueError, "non-empty 1D integer"):
            draw_ordered_sobol_samples(
                bounds=bounds64,
                n=2,
                inequality_constraints=[
                    (torch.tensor([0.0]), torch.tensor([1.0]), 0.1)
                ],
            )
        with self.assertRaisesRegex(ValueError, "`coefficients` must be 1D"):
            draw_ordered_sobol_samples(
                bounds=bounds64,
                n=2,
                inequality_constraints=[
                    (torch.tensor([0]), torch.tensor([1.0, 2.0]), 0.1)
                ],
            )
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            draw_ordered_sobol_samples(
                bounds=bounds64,
                n=2,
                inequality_constraints=[(torch.tensor([5]), torch.tensor([1.0]), 0.1)],
            )
        with self.assertRaisesRegex(ValueError, "leading batch dimension"):
            draw_ordered_sobol_samples(
                bounds=bounds64,
                n=2,
                constraint_callable=lambda X_c: torch.tensor(1.0),
            )
