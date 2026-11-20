#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
from unittest import mock

import torch
from botorch.acquisition.objective import (
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.input import ChainedInputTransform, Normalize
from botorch.posteriors.posterior import Posterior
from botorch.posteriors.torch import TorchPosterior
from botorch.utils.testing import BotorchTestCase
from botorch_community.acquisition import snake as snake_module
from botorch_community.acquisition.snake import (
    apply_epsilon_point_deletion,
    sample_thompson_optima,
    SnAKe,
    solve_open_tsp_path,
)
from gpytorch.kernels import LinearKernel, RBFKernel, ScaleKernel
from torch import Tensor
from torch.distributions import Normal


class _UnsupportedPathwiseModel(Model):
    r"""Custom ``Model`` without pathwise support for discrete fallback tests."""

    def __init__(self, center: Tensor, num_outputs: int = 1) -> None:
        super().__init__()
        self.center = center
        self._num_outputs = num_outputs

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: PosteriorTransform | None = None,
    ) -> Posterior:
        if posterior_transform is not None:
            raise UnsupportedError("Custom posterior does not support transform.")
        # Peak cleanly near self.center across all outputs.
        diff_sq = (X - self.center.to(X)).pow(2).sum(dim=-1, keepdim=True)
        mean = -diff_sq.expand(*X.shape[:-1], self._num_outputs)
        std = torch.full_like(mean, 0.02)
        return TorchPosterior(distribution=Normal(loc=mean, scale=std))


def _path_cost(
    x_start: Tensor,
    path: Tensor,
    transition_weights: Tensor | None = None,
    transition_metric: Tensor | None = None,
) -> float:
    r"""Compute exact cumulative Open-TSP transition cost from ``x_start``."""
    pts = torch.cat([x_start.reshape(1, -1), path], dim=0).to(dtype=torch.float64)
    diffs = pts[1:] - pts[:-1]
    if transition_weights is not None:
        diffs = diffs * transition_weights.to(dtype=torch.float64)
        return float(diffs.norm(p=2, dim=-1).sum().item())
    if transition_metric is not None:
        W = transition_metric.to(dtype=torch.float64)
        quad = torch.einsum("id,dk,ik->i", diffs, W, diffs).clamp_min(0.0)
        return float(quad.sqrt().sum().item())
    return float(diffs.norm(p=2, dim=-1).sum().item())


class TestSnAKe(BotorchTestCase):
    def _make_single_task_gp(
        self,
        dtype: torch.dtype,
        d: int = 2,
        m: int = 1,
        bounds: Tensor | None = None,
        use_normalize: bool = False,
    ) -> tuple[SingleTaskGP, Tensor, Tensor, Tensor]:
        tkwargs = {"device": self.device, "dtype": dtype}
        if bounds is None:
            bounds = torch.zeros(2, d, **tkwargs)
            bounds[1] = 1.0
        torch.manual_seed(1234)
        u = torch.rand(8, d, **tkwargs)
        train_X = bounds[0] + (bounds[1] - bounds[0]) * u
        target_center = bounds[0] + 0.75 * (bounds[1] - bounds[0])
        base = -((train_X - target_center) ** 2).sum(dim=-1, keepdim=True)
        if m == 1:
            train_Y = base
        else:
            alt_center = bounds[0] + 0.25 * (bounds[1] - bounds[0])
            second = -((train_X - alt_center) ** 2).sum(dim=-1, keepdim=True)
            cols = [base, second] + [base * (k + 1) for k in range(m - 2)]
            train_Y = torch.cat(cols[:m], dim=-1)

        input_tf = Normalize(d=d, bounds=bounds) if use_normalize else None
        model = SingleTaskGP(train_X, train_Y, input_transform=input_tf)
        model.eval()
        return model, train_X, train_Y, bounds

    def test_sample_thompson_optima_pathwise_and_fallback(self) -> None:
        for dtype in (torch.float32, torch.float64):
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.tensor([[-1.0, 0.0], [2.0, 3.0]], **tkwargs)
            model, train_X, train_Y, _ = self._make_single_task_gp(
                dtype=dtype, d=2, m=1, bounds=bounds, use_normalize=True
            )

            # 1. Primary mode: continuous Matheron pathwise optimization.
            X_opt = sample_thompson_optima(
                model=model,
                bounds=bounds,
                budget=5,
                num_candidates=128,
                num_restarts=4,
                num_opt_steps=20,
                seed=7,
            )
            self.assertEqual(X_opt.shape, torch.Size([5, 2]))
            self.assertEqual(X_opt.dtype, dtype)
            self.assertEqual(X_opt.device.type, self.device.type)
            self.assertTrue(torch.all(X_opt >= bounds[0] - 1e-6))
            self.assertTrue(torch.all(X_opt <= bounds[1] + 1e-6))

            # Verify zero-step pathwise selection and warm-start support.
            warm = torch.tensor([[0.5, 1.5], [10.0, -5.0]], **tkwargs)
            X_zero_step = sample_thompson_optima(
                model=model,
                bounds=bounds,
                budget=3,
                num_candidates=64,
                num_opt_steps=0,
                warm_start_points=warm,
                seed=11,
            )
            self.assertEqual(X_zero_step.shape, torch.Size([3, 2]))
            self.assertTrue(torch.all(X_zero_step >= bounds[0]))
            self.assertTrue(torch.all(X_zero_step <= bounds[1]))

            # 2. Explicit discrete_candidates=True mode.
            X_disc = sample_thompson_optima(
                model=model,
                bounds=bounds,
                budget=4,
                discrete_candidates=True,
                num_candidates=128,
                seed=7,
            )
            self.assertEqual(X_disc.shape, torch.Size([4, 2]))
            self.assertTrue(torch.all(X_disc >= bounds[0]))
            self.assertTrue(torch.all(X_disc <= bounds[1]))

            # 3. Fallback when ModelListGP has an unsupported LinearKernel.
            gp_lin_1 = SingleTaskGP(train_X, train_Y, covar_module=LinearKernel())
            gp_lin_2 = SingleTaskGP(train_X, -train_Y, covar_module=LinearKernel())
            model_list_lin = ModelListGP(gp_lin_1, gp_lin_2)
            X_ml_fallback = sample_thompson_optima(
                model=model_list_lin,
                bounds=bounds,
                budget=4,
                num_candidates=128,
                seed=13,
            )
            self.assertEqual(X_ml_fallback.shape, torch.Size([4, 2]))
            self.assertTrue(torch.all(X_ml_fallback >= bounds[0]))
            self.assertTrue(torch.all(X_ml_fallback <= bounds[1]))

            # Also verify ModelListGP with RBFKernel (PathList pathwise mode).
            gp_rbf_1 = SingleTaskGP(train_X, train_Y)
            gp_rbf_2 = SingleTaskGP(train_X, -train_Y)
            model_list_rbf = ModelListGP(gp_rbf_1, gp_rbf_2)
            X_ml_path = sample_thompson_optima(
                model=model_list_rbf,
                bounds=bounds,
                budget=4,
                num_candidates=64,
                num_restarts=3,
                num_opt_steps=10,
                seed=19,
            )
            self.assertEqual(X_ml_path.shape, torch.Size([4, 2]))

            # 4. Fallback on custom Model unregistered with draw_matheron_paths
            # (also testing PosteriorTransform evaluation fallback on custom Posterior).
            target_pt = torch.tensor([1.2, 2.1], **tkwargs)
            custom_model = _UnsupportedPathwiseModel(center=target_pt, num_outputs=2)
            ptf = ScalarizedPosteriorTransform(
                weights=torch.tensor([0.7, 0.3], **tkwargs)
            )
            X_custom = sample_thompson_optima(
                model=custom_model,
                bounds=bounds,
                budget=4,
                posterior_transform=ptf,
                num_candidates=512,
                seed=21,
            )
            self.assertEqual(X_custom.shape, torch.Size([4, 2]))
            # Samples should concentrate near target_pt = [1.2, 2.1].
            mean_dist = (X_custom - target_pt).norm(dim=-1).mean().item()
            self.assertLess(mean_dist, 0.35)

            # 5. Mock UnsupportedError inside draw_matheron_paths on SingleTaskGP.
            with mock.patch(
                "botorch_community.acquisition.snake.draw_matheron_paths",
                side_effect=UnsupportedError("Forced unsupported pathwise test"),
            ):
                X_mock_fb = sample_thompson_optima(
                    model=model,
                    bounds=bounds,
                    budget=3,
                    posterior_transform=ScalarizedPosteriorTransform(
                        weights=torch.ones(1, **tkwargs)
                    ),
                    num_candidates=64,
                    seed=5,
                )
            self.assertEqual(X_mock_fb.shape, torch.Size([3, 2]))

    def test_epsilon_point_deletion(self) -> None:
        for dtype in (torch.float32, torch.float64):
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.tensor([[0.0, 0.0], [10.0, 10.0]], **tkwargs)
            X_obs = torch.tensor([[5.0, 5.0], [2.0, 2.0]], **tkwargs)
            # First two candidates are within normalized dist <= 0.05 of X_obs,
            # third is far ([8.5, 8.5]), fourth duplicates the third!
            X_cand = torch.tensor(
                [
                    [5.1, 5.0],
                    [2.05, 1.95],
                    [8.5, 8.5],
                    [8.52, 8.49],
                ],
                **tkwargs,
            )

            # 1. epsilon = 0.0 and epsilon = None -> pure unperturbed SnAKe.
            X_eps0 = apply_epsilon_point_deletion(
                X_candidates=X_cand,
                bounds=bounds,
                X_observed=X_obs,
                epsilon=0.0,
            )
            self.assertAllClose(X_eps0, X_cand)
            X_eps_none = apply_epsilon_point_deletion(
                X_candidates=X_cand,
                bounds=bounds,
                X_observed=X_obs,
                epsilon=None,
            )
            self.assertAllClose(X_eps_none, X_cand)

            # 2. Fixed epsilon = 0.25 replaces near-observed and near-duplicate points,
            # while preserving the well-separated point [8.5, 8.5].
            X_eps_fixed = apply_epsilon_point_deletion(
                X_candidates=X_cand,
                bounds=bounds,
                X_observed=X_obs,
                epsilon=0.25,
                seed=42,
            )
            self.assertEqual(X_eps_fixed.shape, X_cand.shape)
            self.assertTrue(torch.all(X_eps_fixed >= bounds[0]))
            self.assertTrue(torch.all(X_eps_fixed <= bounds[1]))
            # Point 2 ([8.5, 8.5]) is far from X_obs and earlier accepted points,
            # so unless an earlier random replacement landed right next to [8.5, 8.5],
            # all returned points must stay > 0.25 away from X_obs in [0, 1]^2!
            Z_fixed = (X_eps_fixed - bounds[0]) / (bounds[1] - bounds[0])
            Z_obs = (X_obs - bounds[0]) / (bounds[1] - bounds[0])
            min_dists_to_obs = torch.cdist(Z_fixed, Z_obs, p=2).min(dim=-1).values
            self.assertTrue(torch.all(min_dists_to_obs > 0.25))
            # Pairwise distances within the accepted batch must also exceed 0.25.
            pair_dists = torch.cdist(Z_fixed, Z_fixed, p=2) + torch.eye(4, **tkwargs)
            self.assertTrue(torch.all(pair_dists > 0.25))

            # 3. Adaptive / bo_rescaling epsilon scales with GP kernel lengthscales!
            train_Y = torch.randn(2, 1, **tkwargs)
            gp_short = SingleTaskGP(
                X_obs,
                train_Y,
                covar_module=ScaleKernel(RBFKernel(ard_num_dims=2)),
            )
            gp_short.covar_module.base_kernel.lengthscale = torch.tensor(
                [[1.0, 1.0]], **tkwargs
            )  # normalized ls = 0.1 -> eps = 0.05

            gp_long = SingleTaskGP(
                X_obs,
                train_Y,
                covar_module=ScaleKernel(RBFKernel(ard_num_dims=2)),
            )
            gp_long.covar_module.base_kernel.lengthscale = torch.tensor(
                [[6.0, 6.0]], **tkwargs
            )  # normalized ls = 0.6 -> eps = 0.30

            eps_short = snake_module._compute_adaptive_epsilon(
                model=gp_short, bounds=bounds, X_observed=X_obs
            )
            eps_long = snake_module._compute_adaptive_epsilon(
                model=gp_long, bounds=bounds, X_observed=X_obs
            )
            self.assertAlmostEqual(eps_short, 0.05, places=4)
            self.assertAlmostEqual(eps_long, 0.30, places=4)

            # Candidate at [6.5, 5.0] has normalized dist 0.15 from [5.0, 5.0]:
            # kept under gp_short (eps=0.05), replaced under gp_long (eps=0.30)!
            probe = torch.tensor([[6.5, 5.0]], **tkwargs)
            kept = apply_epsilon_point_deletion(
                X_candidates=probe,
                bounds=bounds,
                epsilon="adaptive",
                model=gp_short,
                seed=9,
            )
            self.assertAllClose(kept, probe)

            replaced = apply_epsilon_point_deletion(
                X_candidates=probe,
                bounds=bounds,
                epsilon="bo_rescaling",
                model=gp_long,
                seed=9,
            )
            self.assertFalse(torch.allclose(replaced, probe))
            z_rep = (replaced - bounds[0]) / (bounds[1] - bounds[0])
            self.assertGreater(torch.cdist(z_rep, Z_obs, p=2).min().item(), 0.30)

            # Verify ChainedInputTransform containing Normalize + isotropic kernel.
            chained_tf = ChainedInputTransform(tf1=Normalize(d=2, bounds=bounds))
            gp_chained = SingleTaskGP(
                X_obs,
                train_Y,
                covar_module=RBFKernel(),
                input_transform=chained_tf,
            )
            gp_chained.covar_module.lengthscale = torch.tensor([[0.4]], **tkwargs)
            eps_chained = snake_module._compute_adaptive_epsilon(
                model=gp_chained, bounds=bounds, X_observed=X_obs
            )
            self.assertAlmostEqual(eps_chained, 0.20, places=4)

            # Verify N^(-1/d) fallback when model has no lengthscale (or model=None).
            X_16 = torch.rand(16, 2, **tkwargs) * 10.0
            eps_fallback = snake_module._compute_adaptive_epsilon(
                model=None, bounds=bounds, X_observed=X_16
            )
            self.assertAlmostEqual(eps_fallback, 0.25, places=5)
            eps_empty = snake_module._compute_adaptive_epsilon(
                model=None, bounds=bounds, X_observed=None
            )
            self.assertAlmostEqual(eps_empty, 0.1, places=5)

            # Verify empty candidate batch and saturated epsilon fallback.
            empty_cand = torch.empty((0, 2), **tkwargs)
            self.assertEqual(
                apply_epsilon_point_deletion(empty_cand, bounds).shape,
                torch.Size([0, 2]),
            )
            sat = apply_epsilon_point_deletion(
                X_candidates=X_cand,
                bounds=bounds,
                X_observed=X_obs,
                epsilon=5.0,
                seed=3,
            )
            self.assertEqual(sat.shape, X_cand.shape)
            self.assertTrue(torch.all(sat >= bounds[0]))
            self.assertTrue(torch.all(sat <= bounds[1]))

    def test_solve_open_tsp_path_exact_and_2opt(self) -> None:
        for dtype in (torch.float32, torch.float64):
            tkwargs = {"device": self.device, "dtype": dtype}

            # 1. Trivial B = 1 case.
            x0 = torch.tensor([0.0, 0.0], **tkwargs)
            single_pt = torch.tensor([[0.7, 0.3]], **tkwargs)
            p1, idx1 = solve_open_tsp_path(single_pt, x_current=x0)
            self.assertAllClose(p1, single_pt)
            self.assertEqual(idx1.tolist(), [0])

            # 2. Exact Held-Karp DP (B = 6 <= 13) matches brute-force optimum
            # and strictly improves over deliberately zig-zag unordered input!
            X_zigzag = torch.tensor(
                [
                    [0.95, 0.90],
                    [0.10, 0.15],
                    [0.80, 0.85],
                    [0.25, 0.20],
                    [0.60, 0.55],
                    [0.40, 0.45],
                ],
                **tkwargs,
            )
            ordered_exact, idx_exact = solve_open_tsp_path(
                X_candidates=X_zigzag, x_current=x0
            )
            self.assertEqual(ordered_exact.shape, X_zigzag.shape)
            self.assertEqual(sorted(idx_exact.tolist()), list(range(6)))
            self.assertAllClose(ordered_exact, X_zigzag[idx_exact])

            unordered_cost = _path_cost(x0, X_zigzag)
            exact_cost = _path_cost(x0, ordered_exact)
            self.assertLess(exact_cost, unordered_cost)

            brute_best = min(
                _path_cost(x0, X_zigzag[list(perm)])
                for perm in itertools.permutations(range(6))
            )
            self.assertAlmostEqual(exact_cost, brute_best, places=5)

            # 3. Larger B = 18 > 13 triggers Nearest-Neighbor + 2-opt refinement
            # and strictly reduces cumulative path length vs unordered samples.
            torch.manual_seed(99)
            angles = torch.linspace(0.1, 5.8, 18, **tkwargs)
            perm_scramble = torch.tensor(
                [17, 0, 15, 2, 13, 4, 11, 6, 9, 8, 7, 10, 5, 12, 3, 14, 1, 16],
                device=self.device,
            )
            X_circle = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)[
                perm_scramble
            ]
            x_start_circle = torch.tensor([[1.0, 0.0]], **tkwargs)
            ordered_2opt, idx_2opt = solve_open_tsp_path(
                X_candidates=X_circle, x_current=x_start_circle
            )
            self.assertEqual(sorted(idx_2opt.tolist()), list(range(18)))
            self.assertLess(
                _path_cost(x_start_circle, ordered_2opt),
                _path_cost(x_start_circle, X_circle),
            )

            # Also verify 2-opt improves a case where greedy NN makes a suboptimal
            # suffix move (using exact_threshold=2 to force 2-opt on B=4).
            X_trap = torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.5],
                    [3.0, 0.0],
                    [2.0, 0.0],
                ],
                **tkwargs,
            )
            ordered_trap, _ = solve_open_tsp_path(
                X_candidates=X_trap, x_current=x0, exact_threshold=2
            )
            self.assertLessEqual(
                _path_cost(x0, ordered_trap),
                _path_cost(x0, X_trap),
            )

            # 4. Custom transition_weights and PSD transition_metric W.
            w = torch.tensor([10.0, 0.5], **tkwargs)
            ordered_w, _ = solve_open_tsp_path(
                X_candidates=X_zigzag,
                x_current=x0,
                transition_weights=w,
            )
            cost_w_opt = _path_cost(x0, ordered_w, transition_weights=w)
            cost_w_raw = _path_cost(x0, X_zigzag, transition_weights=w)
            self.assertLess(cost_w_opt, cost_w_raw)
            brute_w = min(
                _path_cost(x0, X_zigzag[list(perm)], transition_weights=w)
                for perm in itertools.permutations(range(6))
            )
            self.assertAlmostEqual(cost_w_opt, brute_w, places=5)

            A = torch.tensor([[2.0, 0.8], [0.8, 1.5]], **tkwargs)
            W_psd = A @ A.transpose(-1, -2)
            ordered_m, _ = solve_open_tsp_path(
                X_candidates=X_zigzag,
                x_current=x0,
                transition_metric=W_psd,
            )
            cost_m_opt = _path_cost(x0, ordered_m, transition_metric=W_psd)
            cost_m_raw = _path_cost(x0, X_zigzag, transition_metric=W_psd)
            self.assertLess(cost_m_opt, cost_m_raw)

            # 5. Default x_current inferred from model.train_inputs[-1].
            bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
            model, train_X, _, _ = self._make_single_task_gp(
                dtype=dtype, d=2, m=1, bounds=bounds, use_normalize=True
            )
            ordered_from_model, _ = solve_open_tsp_path(
                X_candidates=X_zigzag, x_current=None, model=model
            )
            ordered_explicit, _ = solve_open_tsp_path(
                X_candidates=X_zigzag, x_current=train_X[-1]
            )
            self.assertAllClose(ordered_from_model, ordered_explicit)

    def test_snake_policy_closed_loop_and_mobo(self) -> None:
        for dtype in (torch.float32, torch.float64):
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)

            # 1. Receding-horizon replan_interval=1 with warm-starting.
            model, train_X, train_Y, _ = self._make_single_task_gp(
                dtype=dtype, d=2, m=1, bounds=bounds, use_normalize=True
            )
            policy = SnAKe(
                budget=5,
                epsilon="adaptive",
                replan_interval=1,
                num_candidates=64,
                num_restarts=3,
                num_opt_steps=10,
                warm_start=True,
                seed=101,
            )
            self.assertEqual(policy.step_count, 0)
            self.assertIsNone(policy.cached_path)

            X_hist = train_X.clone()
            Y_hist = train_Y.clone()
            for step in range(3):
                expected_horizon = 5 - step
                next_x, full_path = policy.suggest(model=model, bounds=bounds)
                self.assertEqual(next_x.shape, torch.Size([1, 2]))
                self.assertEqual(full_path.shape, torch.Size([expected_horizon, 2]))
                self.assertAllClose(next_x, full_path[:1])
                self.assertTrue(torch.all(full_path >= bounds[0]))
                self.assertTrue(torch.all(full_path <= bounds[1]))
                self.assertEqual(policy.step_count, step + 1)
                cached = policy.cached_path
                self.assertIsNotNone(cached)
                assert cached is not None
                self.assertEqual(cached.shape, torch.Size([expected_horizon - 1, 2]))

                new_y = -((next_x - 0.75) ** 2).sum(dim=-1, keepdim=True)
                X_hist = torch.cat([X_hist, next_x], dim=0)
                Y_hist = torch.cat([Y_hist, new_y], dim=0)
                model = SingleTaskGP(
                    X_hist, Y_hist, input_transform=Normalize(d=2, bounds=bounds)
                )
                model.eval()

            policy.reset()
            self.assertEqual(policy.step_count, 0)
            self.assertIsNone(policy.cached_path)

            # 2. Periodic replanning (replan_interval=3) pops cached path between
            # model updates and replans on step 3!
            periodic_policy = SnAKe(
                budget=6,
                epsilon=0.1,
                replan_interval=3,
                num_candidates=64,
                num_restarts=2,
                num_opt_steps=8,
                seed=202,
            )
            with mock.patch(
                "botorch_community.acquisition.snake.sample_thompson_optima",
                wraps=sample_thompson_optima,
            ) as spy_ts:
                x_s0, path_s0 = periodic_policy.suggest(model=model, bounds=bounds)
                self.assertEqual(spy_ts.call_count, 1)
                self.assertEqual(path_s0.shape, torch.Size([6, 2]))

                # Step 1 & Step 2 must pop directly from path_s0 without calling TS!
                x_s1, path_s1 = periodic_policy.suggest(model=model, bounds=bounds)
                self.assertEqual(spy_ts.call_count, 1)
                self.assertAllClose(x_s1, path_s0[1:2])
                self.assertAllClose(path_s1, path_s0[1:])

                x_s2, path_s2 = periodic_policy.suggest(
                    model=model, bounds=bounds, remaining_budget=3
                )
                self.assertEqual(spy_ts.call_count, 1)
                self.assertAllClose(x_s2, path_s0[2:3])
                self.assertEqual(path_s2.shape, torch.Size([3, 2]))

                # Step 3 reaches replan_interval=3 -> triggers fresh TS replan!
                x_s3, path_s3 = periodic_policy.suggest(model=model, bounds=bounds)
                self.assertEqual(spy_ts.call_count, 2)
                self.assertEqual(x_s3.shape, torch.Size([1, 2]))
                self.assertEqual(path_s3.shape, torch.Size([3, 2]))

            # 3. Multi-objective Chebyshev scalarization (MOBO) + custom x_current.
            mo_model, _, _, _ = self._make_single_task_gp(
                dtype=dtype, d=2, m=2, bounds=bounds, use_normalize=True
            )
            mo_policy = SnAKe(
                budget=4,
                epsilon="bo_rescaling",
                replan_interval=1,
                num_candidates=64,
                num_restarts=2,
                num_opt_steps=10,
                seed=303,
            )
            next_mo, path_mo = mo_policy.suggest(
                model=mo_model,
                bounds=bounds,
                x_current=torch.tensor([0.5, 0.5], **tkwargs),
            )
            self.assertEqual(next_mo.shape, torch.Size([1, 2]))
            self.assertEqual(path_mo.shape, torch.Size([4, 2]))

            # 4. Suggest on a custom model without train_inputs uses midpoint
            # on first step and _last_x_suggested on second step.
            custom_m = _UnsupportedPathwiseModel(
                center=torch.tensor([0.3, 0.7], **tkwargs), num_outputs=1
            )
            custom_policy = SnAKe(
                budget=3,
                epsilon="adaptive",
                discrete_candidates=True,
                num_candidates=64,
                seed=404,
            )
            nx1, _ = custom_policy.suggest(model=custom_m, bounds=bounds)
            nx2, _ = custom_policy.suggest(
                model=custom_m,
                bounds=bounds,
                x_current=nx1,
                remaining_budget=2,
            )
            nx3, _ = custom_policy.suggest(model=custom_m, bounds=bounds)
            self.assertEqual(nx2.shape, torch.Size([1, 2]))
            self.assertEqual(nx3.shape, torch.Size([1, 2]))

    def test_validation_errors(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
        model, X_obs, _, _ = self._make_single_task_gp(
            dtype=torch.double, d=2, m=1, bounds=bounds
        )

        # Bounds errors.
        with self.assertRaises(ValueError):
            sample_thompson_optima(model, torch.ones(3, 2, **tkwargs), budget=2)
        with self.assertRaises(ValueError):
            sample_thompson_optima(
                model,
                torch.tensor([[0.0, float("inf")], [1.0, 1.0]], **tkwargs),
                budget=2,
            )
        with self.assertRaises(ValueError):
            sample_thompson_optima(
                model,
                torch.tensor([[1.0, 0.0], [0.5, 1.0]], **tkwargs),
                budget=2,
            )

        # sample_thompson_optima argument errors.
        with self.assertRaises(ValueError):
            sample_thompson_optima(model, bounds, budget=0)
        with self.assertRaises(ValueError):
            sample_thompson_optima(model, bounds, budget=2, num_candidates=0)
        with self.assertRaises(ValueError):
            sample_thompson_optima(model, bounds, budget=2, num_restarts=0)
        with self.assertRaises(ValueError):
            sample_thompson_optima(model, bounds, budget=2, num_opt_steps=-1)
        with self.assertRaises(ValueError):
            sample_thompson_optima(model, bounds, budget=2, lr=0.0)
        with self.assertRaises(ValueError):
            sample_thompson_optima(
                model,
                bounds,
                budget=2,
                warm_start_points=torch.ones(2, 3, **tkwargs),
            )

        # apply_epsilon_point_deletion argument errors.
        X_cand = torch.rand(3, 2, **tkwargs)
        with self.assertRaises(ValueError):
            apply_epsilon_point_deletion(X_cand, bounds, epsilon=-0.1)
        with self.assertRaises(ValueError):
            apply_epsilon_point_deletion(X_cand, bounds, epsilon=True)
        with self.assertRaises(ValueError):
            apply_epsilon_point_deletion(X_cand, bounds, epsilon="unknown_mode")
        with self.assertRaises(ValueError):
            apply_epsilon_point_deletion(
                X_cand,
                bounds,
                epsilon=[0.1],  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            apply_epsilon_point_deletion(
                torch.rand(3, 3, **tkwargs), bounds, epsilon=0.1
            )
        with self.assertRaises(ValueError):
            apply_epsilon_point_deletion(
                X_cand,
                bounds,
                X_observed=torch.rand(2, 3, **tkwargs),
                epsilon=0.1,
            )

        # solve_open_tsp_path argument errors.
        x0 = X_obs[-1]
        with self.assertRaises(ValueError):
            solve_open_tsp_path(torch.empty(0, 2, **tkwargs), x_current=x0)
        with self.assertRaises(ValueError):
            solve_open_tsp_path(X_cand, x_current=x0, exact_threshold=0)
        with self.assertRaises(ValueError):
            solve_open_tsp_path(X_cand, x_current=None, model=None)
        with self.assertRaises(ValueError):
            solve_open_tsp_path(X_cand, x_current=torch.ones(2, 2, **tkwargs))
        with self.assertRaises(ValueError):
            solve_open_tsp_path(
                X_cand,
                x_current=x0,
                transition_weights=torch.ones(2, **tkwargs),
                transition_metric=torch.eye(2, **tkwargs),
            )
        with self.assertRaises(ValueError):
            solve_open_tsp_path(
                X_cand,
                x_current=x0,
                transition_weights=torch.ones(3, **tkwargs),
            )
        with self.assertRaises(ValueError):
            solve_open_tsp_path(
                X_cand,
                x_current=x0,
                transition_weights=torch.tensor([-1.0, 1.0], **tkwargs),
            )
        with self.assertRaises(ValueError):
            solve_open_tsp_path(
                X_cand,
                x_current=x0,
                transition_weights=torch.zeros(2, **tkwargs),
            )
        with self.assertRaises(ValueError):
            solve_open_tsp_path(
                X_cand,
                x_current=x0,
                transition_metric=torch.eye(3, **tkwargs),
            )
        with self.assertRaises(ValueError):
            solve_open_tsp_path(
                X_cand,
                x_current=x0,
                transition_metric=torch.tensor([[1.0, 2.0], [0.0, 1.0]], **tkwargs),
            )
        with self.assertRaises(ValueError):
            solve_open_tsp_path(
                X_cand,
                x_current=x0,
                transition_metric=torch.tensor([[1.0, 0.0], [0.0, -0.5]], **tkwargs),
            )

        # SnAKe constructor & suggest validation errors.
        with self.assertRaises(ValueError):
            SnAKe(budget=0)
        with self.assertRaises(ValueError):
            SnAKe(replan_interval=0)
        with self.assertRaises(ValueError):
            SnAKe(num_candidates=0)
        with self.assertRaises(ValueError):
            SnAKe(num_restarts=0)
        with self.assertRaises(ValueError):
            SnAKe(num_opt_steps=-1)
        with self.assertRaises(ValueError):
            SnAKe(lr=0.0)
        with self.assertRaises(ValueError):
            SnAKe(exact_tsp_threshold=0)
        with self.assertRaises(ValueError):
            SnAKe(
                transition_weights=torch.ones(2, **tkwargs),
                transition_metric=torch.eye(2, **tkwargs),
            )

        policy = SnAKe(budget=3)
        with self.assertRaises(ValueError):
            policy.suggest(model=model, bounds=bounds, remaining_budget=0)
        with self.assertRaises(ValueError):
            policy.suggest(
                model=model,
                bounds=bounds,
                x_current=torch.ones(2, 2, **tkwargs),
            )
