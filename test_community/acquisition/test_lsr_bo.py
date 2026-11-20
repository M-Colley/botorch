#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.optim.optimize import optimize_acqf
from botorch.utils.testing import BotorchTestCase
from botorch_community.acquisition.lsr_bo import (
    compute_local_box_bounds,
    GaussHermiteNormalSampler,
    gen_local_region_initial_conditions,
    LocalSearchRolloutAcquisitionFunction,
    LSRBO,
    LSRBOStepInfo,
    make_local_region_constraint,
    optimize_lsr_bo_step,
    project_to_local_region,
)
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from torch import Tensor


def _bimodal_objective_2d(X: Tensor) -> Tensor:
    r"""Bimodal 2D objective: local mode (0.15, 0.15), global mode (0.82, 0.82)."""
    c_local = torch.tensor([0.15, 0.15], device=X.device, dtype=X.dtype)
    c_global = torch.tensor([0.82, 0.82], device=X.device, dtype=X.dtype)
    local_peak = 1.0 * torch.exp(-((X - c_local) ** 2).sum(dim=-1) / (2.0 * 0.08**2))
    global_peak = 3.0 * torch.exp(-((X - c_global) ** 2).sum(dim=-1) / (2.0 * 0.12**2))
    return (local_peak + global_peak).unsqueeze(-1)


def _build_bimodal_gp(
    train_X: Tensor, train_Y: Tensor, prior_mean: float = 0.0
) -> SingleTaskGP:
    r"""Construct a calibrated GP for fast deterministic BO tests."""
    covar_module = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=train_X.shape[-1],
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )
    )
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=torch.full_like(train_Y, 1e-4),
        covar_module=covar_module,
        outcome_transform=None,
    )
    with torch.no_grad():
        model.mean_module.constant.fill_(prior_mean)
        model.covar_module.outputscale = 1.5
        model.covar_module.base_kernel.lengthscale = 0.12
    model.eval()
    return model


class TestLocalSearchRegionGeometry(BotorchTestCase):
    def test_compute_local_box_bounds(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.tensor([[0.0, -1.0, 2.0], [1.0, 1.0, 5.0]], **tkwargs)
            x_curr = torch.tensor([0.1, 0.9, 3.5], **tkwargs)

            # Scalar delta, p = inf and p = 2
            for p in (float("inf"), 2.0):
                lb_ub = compute_local_box_bounds(
                    x_current=x_curr, bounds=bounds, delta=0.25, p=p
                )
                expected = torch.tensor(
                    [[0.0, 0.65, 3.25], [0.35, 1.0, 3.75]], **tkwargs
                )
                self.assertAllClose(lb_ub, expected)

            # Vector delta and (1, d) x_current
            delta_vec = torch.tensor([0.2, 0.5, 1.0], **tkwargs)
            lb_ub_vec = compute_local_box_bounds(
                x_current=x_curr.unsqueeze(0),
                bounds=bounds,
                delta=delta_vec,
                p=float("inf"),
            )
            expected_vec = torch.tensor([[0.0, 0.4, 2.5], [0.3, 1.0, 4.5]], **tkwargs)
            self.assertAllClose(lb_ub_vec, expected_vec)

    def test_project_to_local_region(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            tol = 1e-5 if dtype == torch.float else 1e-10
            bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
            x_curr = torch.tensor([0.2, 0.1], **tkwargs)

            # 1. p = inf with vector delta
            delta_vec = torch.tensor([0.15, 0.25], **tkwargs)
            target = torch.tensor([0.9, -0.5], **tkwargs)
            proj_inf = project_to_local_region(
                target_x=target,
                x_current=x_curr,
                bounds=bounds,
                delta=delta_vec,
                p=float("inf"),
            )
            self.assertAllClose(
                proj_inf, torch.tensor([0.35, 0.0], **tkwargs), atol=tol
            )

            # 2. p = 2 with scalar delta, interior direction: exact radial step
            delta_scalar = 0.2
            target_interior = torch.tensor([0.8, 0.9], **tkwargs)
            proj_l2 = project_to_local_region(
                target_x=target_interior,
                x_current=x_curr,
                bounds=bounds,
                delta=delta_scalar,
                p=2.0,
            )
            direction = target_interior - x_curr
            expected_l2 = x_curr + delta_scalar * direction / direction.norm(p=2)
            self.assertAllClose(proj_l2, expected_l2, atol=tol)
            self.assertLessEqual(
                float(((proj_l2 - x_curr) / delta_scalar).norm(p=2).item()),
                1.0 + 1e-7,
            )

            # 3. p = 2 with active global boundary: target = (0.8, -0.5) hits x_2 = 0
            # Exact projection onto B(x_curr, 0.2) n [0, 1]^2 has x_2 = 0 and
            # x_1 = 0.2 + sqrt(0.2^2 - 0.1^2) = 0.2 + sqrt(0.03).
            target_bdry = torch.tensor([0.8, -0.5], **tkwargs)
            proj_bdry = project_to_local_region(
                target_x=target_bdry,
                x_current=x_curr,
                bounds=bounds,
                delta=delta_scalar,
                p=2.0,
            )
            expected_bdry = torch.tensor([0.2 + math.sqrt(0.03), 0.0], **tkwargs)
            self.assertAllClose(proj_bdry, expected_bdry, atol=tol)

            # 4. p = 2 with vector delta (ellipsoid) & batched targets: check KKT
            # variational inequality <y - x_proj, z - x_proj> <= 0 for all feasible z.
            batch_targets = torch.tensor(
                [[[0.9, 0.8], [0.22, 0.12]], [[-0.4, 0.9], [0.7, -0.3]]],
                **tkwargs,
            )
            proj_ellip = project_to_local_region(
                target_x=batch_targets,
                x_current=x_curr,
                bounds=bounds,
                delta=delta_vec,
                p=2.0,
            )
            self.assertEqual(proj_ellip.shape, batch_targets.shape)
            # Feasible point [0.22, 0.12] must remain untouched
            self.assertAllClose(proj_ellip[0, 1], batch_targets[0, 1], atol=tol)
            scaled_norm = ((proj_ellip - x_curr) / delta_vec).norm(p=2, dim=-1)
            self.assertTrue((scaled_norm <= 1.0 + 1e-7).all())
            self.assertTrue((proj_ellip >= bounds[0]).all())
            self.assertTrue((proj_ellip <= bounds[1]).all())

            # Sample feasible test points z in T(x_curr) n Theta and check KKT
            theta_angles = torch.linspace(0, 2 * math.pi, 64, **tkwargs)
            z_feas = torch.stack(
                [
                    x_curr[0] + delta_vec[0] * torch.cos(theta_angles),
                    x_curr[1] + delta_vec[1] * torch.sin(theta_angles),
                ],
                dim=-1,
            ).clamp(min=bounds[0], max=bounds[1])
            for idx in ((0, 0), (1, 0), (1, 1)):
                y_pt = batch_targets[idx]
                p_pt = proj_ellip[idx]
                inner = ((y_pt - p_pt) * (z_feas - p_pt)).sum(dim=-1)
                self.assertLessEqual(
                    float(inner.max().item()), 5e-4 if dtype == torch.float else 1e-7
                )

            # 5. Also test p = 1.0 and p = 1.5
            for p_val in (1.0, 1.5):
                proj_lp = project_to_local_region(
                    target_x=target_interior,
                    x_current=x_curr,
                    bounds=bounds,
                    delta=delta_vec,
                    p=p_val,
                )
                lp_val = (((proj_lp - x_curr).abs() / delta_vec) ** p_val).sum()
                self.assertLessEqual(float(lp_val.item()), 1.0 + 1e-6)

    def test_nonlinear_constraint_and_optimize_acqf(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
            x_curr = torch.tensor([0.3, 0.3], **tkwargs)
            delta = torch.tensor([0.15, 0.2], **tkwargs)

            for p_val in (2.0, 1.5):
                nlc_fn, is_intra = make_local_region_constraint(
                    x_current=x_curr, delta=delta, p=p_val
                )
                self.assertTrue(is_intra)
                # Gradient at exact center x == x_curr must be finite and zero!
                x_center = x_curr.clone().requires_grad_(True)
                c_center = nlc_fn(x_center)
                self.assertEqual(c_center.ndim, 0)
                self.assertAlmostEqual(float(c_center.item()), 1.0, places=6)
                c_center.backward()
                self.assertIsNotNone(x_center.grad)
                self.assertFalse(torch.isnan(x_center.grad).any())
                self.assertAllClose(x_center.grad, torch.zeros_like(x_center))

            # Optimize EI directly with nonlinear_inequality_constraints
            train_X = torch.tensor([[0.25, 0.25], [0.35, 0.35], [0.8, 0.8]], **tkwargs)
            train_Y = torch.tensor([[0.2], [0.4], [2.0]], **tkwargs)
            model = _build_bimodal_gp(train_X=train_X, train_Y=train_Y)
            ei = ExpectedImprovement(model=model, best_f=2.0)

            local_bounds = compute_local_box_bounds(
                x_current=x_curr, bounds=bounds, delta=delta, p=2.0
            )
            nlc = [make_local_region_constraint(x_current=x_curr, delta=delta, p=2.0)]
            ic_gen = gen_local_region_initial_conditions(
                x_current=x_curr, bounds=bounds, delta=delta, p=2.0
            )
            cand, _ = optimize_acqf(
                acq_function=ei,
                bounds=local_bounds,
                q=1,
                num_restarts=3,
                raw_samples=32,
                options={"maxiter": 20, "ftol": 1e-5},
                nonlinear_inequality_constraints=nlc,
                ic_generator=ic_gen,
                retry_on_optimization_warning=False,
            )
            scaled_dist = ((cand.squeeze(0) - x_curr) / delta).norm(p=2)
            self.assertLessEqual(float(scaled_dist.item()), 1.0 + 1e-5)

    def test_geometry_input_validation(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
        x_curr = torch.tensor([0.5, 0.5], **tkwargs)

        with self.assertRaises(ValueError):
            compute_local_box_bounds(x_curr, bounds[0], delta=0.1)
        with self.assertRaises(ValueError):
            compute_local_box_bounds(
                x_curr, torch.tensor([[1.0, 0.0], [0.0, 1.0]], **tkwargs), delta=0.1
            )
        with self.assertRaises(ValueError):
            compute_local_box_bounds(x_curr, bounds, delta=-0.1)
        with self.assertRaises(ValueError):
            compute_local_box_bounds(x_curr, bounds, delta=0.1, p=0.5)
        with self.assertRaises(ValueError):
            compute_local_box_bounds(torch.ones(3, **tkwargs), bounds, delta=0.1)
        with self.assertRaises(ValueError):
            compute_local_box_bounds(
                torch.tensor([2.0, 2.0], **tkwargs), bounds, delta=0.1
            )
        with self.assertRaises(ValueError):
            project_to_local_region(torch.ones(3, **tkwargs), x_curr, bounds, delta=0.1)


class TestLSRBO(BotorchTestCase):
    def test_lsr_bo_escapes_local_trap_on_bimodal_function(self) -> None:
        r"""Verify gamma=0 is trapped while LSR-BO (gamma=0.5) finds global mode."""
        for dtype in (torch.float, torch.double):
            torch.manual_seed(0)
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
            delta = 0.15

            # Start trapped at local mode (0.15, 0.15) with local neighborhood mapped
            init_X = torch.tensor(
                [
                    [0.05, 0.05],
                    [0.05, 0.25],
                    [0.25, 0.05],
                    [0.25, 0.25],
                    [0.15, 0.23],
                    [0.23, 0.15],
                    [0.15, 0.15],
                ],
                **tkwargs,
            )
            init_Y = _bimodal_objective_2d(init_X)

            # 1. Greedy local BO (gamma = 0.0): stays trapped in suboptimal mode (~1.0)
            X_greedy = init_X.clone()
            Y_greedy = init_Y.clone()
            greedy_policy = LSRBO(
                bounds=bounds,
                delta=delta,
                gamma=0.0,
                p=float("inf"),
                num_restarts=4,
                raw_samples=64,
            )
            for _ in range(8):
                model_g = _build_bimodal_gp(X_greedy, Y_greedy, prior_mean=0.8)
                acqf_g = ExpectedImprovement(model=model_g, best_f=Y_greedy.max())
                x_prev_g = X_greedy[-1]
                x_next_g, _ = greedy_policy(acqf_g)
                step_inf_g = (x_next_g.squeeze(0) - x_prev_g).abs().max()
                self.assertLessEqual(float(step_inf_g.item()), delta + 1e-6)
                X_greedy = torch.cat([X_greedy, x_next_g], dim=0)
                Y_greedy = torch.cat([Y_greedy, _bimodal_objective_2d(x_next_g)], dim=0)

            self.assertLess(float(Y_greedy.max().item()), 1.2)

            # 2. LSR-BO (gamma = 0.5): strictly respects ||x_{t+1} - x_t||_inf <= delta
            # at every step AND traverses the valley to reach the global mode (~3.0)!
            X_lsr = init_X.clone()
            Y_lsr = init_Y.clone()
            lsr_policy = LSRBO(
                bounds=bounds,
                delta=delta,
                gamma=0.5,
                p=float("inf"),
                num_restarts=6,
                raw_samples=128,
            )
            saw_global_projection = False
            for _ in range(10):
                model_l = _build_bimodal_gp(X_lsr, Y_lsr, prior_mean=0.8)
                acqf_l = LogExpectedImprovement(model=model_l, best_f=Y_lsr.max())
                x_prev_l = X_lsr[-1]
                x_next_l, _ = lsr_policy.step(acqf_l)
                self.assertIsInstance(lsr_policy.last_step_info, LSRBOStepInfo)
                if not lsr_policy.last_step_info.used_local_step:
                    saw_global_projection = True
                step_inf_l = (x_next_l.squeeze(0) - x_prev_l).abs().max()
                self.assertLessEqual(float(step_inf_l.item()), delta + 1e-6)
                self.assertTrue((x_next_l >= bounds[0]).all())
                self.assertTrue((x_next_l <= bounds[1]).all())
                X_lsr = torch.cat([X_lsr, x_next_l], dim=0)
                Y_lsr = torch.cat([Y_lsr, _bimodal_objective_2d(x_next_l)], dim=0)

            self.assertTrue(saw_global_projection)
            self.assertGreater(float(Y_lsr.max().item()), 2.7)

    def test_lsr_bo_euclidean_p2_and_ucb_and_gamma_extremes(self) -> None:
        for dtype in (torch.float, torch.double):
            torch.manual_seed(2)
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
            delta = 0.2
            train_X = torch.tensor([[0.1, 0.1], [0.15, 0.15], [0.85, 0.85]], **tkwargs)
            # Negative objective values to also test negative UCB handling
            train_Y = torch.tensor([[-5.0], [-4.8], [-1.0]], **tkwargs)
            model = _build_bimodal_gp(train_X=train_X, train_Y=train_Y)
            ucb = UpperConfidenceBound(model=model, beta=2.0)
            x_curr = torch.tensor([0.15, 0.15], **tkwargs)

            # gamma = 1.0 steers toward global target whenever x_global not in T(x_k)
            cand_g1, val_g1, info_g1 = optimize_lsr_bo_step(
                acq_function=ucb,
                bounds=bounds,
                delta=delta,
                x_current=x_curr,
                gamma=1.0,
                p=2.0,
                num_restarts=4,
                raw_samples=64,
                return_step_info=True,
            )
            self.assertEqual(cand_g1.shape, torch.Size([1, 2]))
            self.assertEqual(val_g1.ndim, 0)
            self.assertFalse(info_g1.used_local_step)
            l2_step = (cand_g1.squeeze(0) - x_curr).norm(p=2)
            self.assertAlmostEqual(
                float(l2_step.item()),
                delta,
                places=4 if dtype == torch.float else 6,
            )

            # Test local_ball_fraction > 0 refinement around projected step
            cand_ball, _ = optimize_lsr_bo_step(
                acq_function=ucb,
                bounds=bounds,
                delta=delta,
                x_current=x_curr,
                gamma=0.8,
                p=2.0,
                num_restarts=4,
                raw_samples=64,
                local_ball_fraction=0.25,
            )
            self.assertLessEqual(
                float((cand_ball.squeeze(0) - x_curr).norm(p=2).item()),
                delta + 1e-5,
            )

    def test_lsr_bo_with_q_log_nehvi(self) -> None:
        r"""Test LSR-BO with qLogNoisyExpectedHypervolumeImprovement (_log=True)."""
        for dtype in (torch.float, torch.double):
            torch.manual_seed(3)
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
            train_X = torch.tensor(
                [
                    [0.10, 0.12],
                    [0.18, 0.14],
                    [0.82, 0.85],
                    [0.14, 0.15],
                ],
                **tkwargs,
            )
            y1 = _bimodal_objective_2d(train_X)
            y2 = _bimodal_objective_2d(1.0 - train_X * 0.2)
            train_Y = torch.cat([y1, y2], dim=-1)
            mo_model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
            mo_model.eval()

            ref_point = torch.tensor([-0.5, -0.5], **tkwargs)
            acqf = qLogNoisyExpectedHypervolumeImprovement(
                model=mo_model,
                ref_point=ref_point,
                X_baseline=train_X,
                prune_baseline=True,
            )
            self.assertTrue(acqf._log)

            for p_val in (float("inf"), 2.0):
                policy = LSRBO(
                    bounds=bounds,
                    delta=0.2,
                    gamma=0.5,
                    p=p_val,
                    num_restarts=2,
                    raw_samples=16,
                    options={"maxiter": 10},
                )
                # Also verifies x_current = train_inputs[-1] on multi-output GP
                cand, val = policy.step(acqf)
                self.assertEqual(cand.shape, torch.Size([1, 2]))
                self.assertEqual(val.ndim, 0)
                x_ref = train_X[-1]
                if math.isinf(p_val):
                    dist = (cand.squeeze(0) - x_ref).abs().max()
                else:
                    dist = (cand.squeeze(0) - x_ref).norm(p=2)
                self.assertLessEqual(float(dist.item()), 0.2 + 1e-5)
                self.assertAllClose(
                    policy.get_local_bounds(x_ref),
                    compute_local_box_bounds(x_ref, bounds, 0.2, p=p_val),
                )
                self.assertAllClose(
                    policy.project(cand, x_ref),
                    cand,
                    atol=1e-5 if dtype == torch.float else 1e-8,
                )

    def test_lsr_bo_validation(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
        with self.assertRaises(ValueError):
            LSRBO(bounds=bounds, delta=0.1, gamma=1.5)
        with self.assertRaises(ValueError):
            LSRBO(bounds=bounds, delta=0.1, local_ball_fraction=1.0)


class TestLocalSearchRolloutAcquisitionFunction(BotorchTestCase):
    def test_rollout_horizons_shapes_and_gradients(self) -> None:
        for dtype in (torch.float, torch.double):
            torch.manual_seed(5)
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
            train_X = torch.tensor(
                [[0.1, 0.1], [0.2, 0.25], [0.6, 0.7], [0.15, 0.2]], **tkwargs
            )
            train_Y = _bimodal_objective_2d(train_X)
            model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
            model.eval()
            best_f = float(train_Y.max().item())

            # Compare horizon=1 with analytic ExpectedImprovement
            rollout_h1 = LocalSearchRolloutAcquisitionFunction(
                model=model,
                bounds=bounds,
                delta=0.2,
                horizon=1,
                num_fantasies=8,
                best_f=best_f,
            )
            analytic_ei = ExpectedImprovement(model=model, best_f=best_f)
            X_test_batch = torch.tensor(
                [
                    [[0.18, 0.22]],
                    [[0.25, 0.30]],
                    [[0.12, 0.15]],
                ],
                **tkwargs,
            )
            self.assertAllClose(
                rollout_h1(X_test_batch),
                analytic_ei(X_test_batch),
                atol=1e-5 if dtype == torch.float else 1e-10,
            )

            # Test horizons 1, 2, 3, t-batch shapes, non-negativity, and autograd wrt X
            prev_vals: Tensor | None = None
            for horizon in (1, 2, 3):
                acqf = LocalSearchRolloutAcquisitionFunction(
                    model=model,
                    bounds=bounds,
                    delta=0.2,
                    horizon=horizon,
                    num_fantasies=8,
                    discount_factor=0.9,
                    p=float("inf"),
                    best_f=best_f,
                    num_local_candidates=16,
                )

                # Unbatched (1, d), 1D t-batch (3, 1, d), and 2D t-batch (2, 3, 1, d)
                val_single = acqf(X_test_batch[0])
                self.assertEqual(val_single.shape, torch.Size([1]))

                X_grad = X_test_batch.clone().requires_grad_(True)
                vals = acqf(X_grad)
                self.assertEqual(vals.shape, torch.Size([3]))
                self.assertTrue((vals >= 0.0).all())
                if prev_vals is not None:
                    # Adding discounted non-negative stage EI cannot decrease value
                    self.assertTrue((vals >= prev_vals - 1e-6).all())
                prev_vals = vals.detach()

                vals.sum().backward()
                self.assertIsNotNone(X_grad.grad)
                self.assertFalse(torch.isnan(X_grad.grad).any())
                self.assertGreater(float(X_grad.grad.abs().sum().item()), 0.0)

                X_2d = X_test_batch.unsqueeze(0).expand(2, -1, -1, -1).clone()
                vals_2d = acqf(X_2d)
                self.assertEqual(vals_2d.shape, torch.Size([2, 3]))

    def test_rollout_gauss_hermite_p2_terminal_mean_and_optimization(self) -> None:
        for dtype in (torch.float, torch.double):
            torch.manual_seed(7)
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
            x_curr = torch.tensor([0.2, 0.2], **tkwargs)
            delta = torch.tensor([0.15, 0.2], **tkwargs)

            train_X = torch.tensor(
                [[0.15, 0.15], [0.25, 0.2], [0.75, 0.8], [0.2, 0.2]], **tkwargs
            )
            train_Y = _bimodal_objective_2d(train_X)
            # FixedNoise GP to exercise fixed-noise conditioning across rollout stages
            model = _build_bimodal_gp(train_X=train_X, train_Y=train_Y)

            # Test Gauss-Hermite + p = 2 + base_policy = "lsr_bo" + optimize_acqf
            rollout_gh = LocalSearchRolloutAcquisitionFunction(
                model=model,
                bounds=bounds,
                delta=delta,
                x_current=x_curr,
                horizon=2,
                num_fantasies=4,
                p=2.0,
                use_gauss_hermite=True,
                num_local_candidates=12,
                base_policy="lsr_bo",
                lsr_gamma=0.5,
            )
            self.assertIsInstance(rollout_gh.sampler, GaussHermiteNormalSampler)

            # Optimize within local_bounds (L-BFGS-B) and with p = 2 SLSQP constraint
            local_bounds_inf = compute_local_box_bounds(
                x_current=x_curr, bounds=bounds, delta=delta, p=float("inf")
            )
            cand_inf, val_inf = optimize_acqf(
                acq_function=rollout_gh,
                bounds=local_bounds_inf,
                q=1,
                num_restarts=2,
                raw_samples=8,
                options={"maxiter": 6},
                retry_on_optimization_warning=False,
            )
            self.assertEqual(cand_inf.shape, torch.Size([1, 2]))
            self.assertGreaterEqual(float(val_inf.item()), 0.0)
            self.assertTrue((cand_inf >= local_bounds_inf[0] - 1e-6).all())
            self.assertTrue((cand_inf <= local_bounds_inf[1] + 1e-6).all())

            delta_slsqp = delta * (1.0 - (1e-4 if dtype == torch.float else 1e-9))
            local_bounds = compute_local_box_bounds(
                x_current=x_curr, bounds=bounds, delta=delta, p=2.0
            )
            nlc = [
                make_local_region_constraint(x_current=x_curr, delta=delta_slsqp, p=2.0)
            ]
            ic_gen = gen_local_region_initial_conditions(
                x_current=x_curr, bounds=bounds, delta=delta_slsqp, p=2.0
            )
            cand, val = optimize_acqf(
                acq_function=rollout_gh,
                bounds=local_bounds,
                q=1,
                num_restarts=1,
                raw_samples=8,
                options={"maxiter": 8, "ftol": 1e-5},
                nonlinear_inequality_constraints=nlc,
                ic_generator=ic_gen,
                retry_on_optimization_warning=False,
            )
            self.assertEqual(cand.shape, torch.Size([1, 2]))
            self.assertGreaterEqual(float(val.item()), 0.0)
            l2_dist = ((cand.squeeze(0) - x_curr) / delta).norm(p=2)
            self.assertLessEqual(float(l2_dist.item()), 1.0 + 1e-5)

            # Test terminal_mean mode, base_policy = "ucb", X_pending, maximize=False
            rollout_term = LocalSearchRolloutAcquisitionFunction(
                model=model,
                bounds=bounds,
                delta=delta,
                x_current=x_curr,
                horizon=2,
                num_fantasies=4,
                p=float("inf"),
                num_local_candidates=12,
                base_policy="ucb",
                ucb_beta=1.5,
                reward_mode="terminal_mean",
                maximize=False,
                X_pending=torch.tensor([[0.22, 0.18]], **tkwargs),
            )
            X_probe = torch.tensor(
                [[[0.25, 0.25]], [[0.18, 0.22]]], **tkwargs
            ).requires_grad_(True)
            term_out = rollout_term(X_probe)
            self.assertEqual(term_out.shape, torch.Size([2]))
            term_out.sum().backward()
            self.assertIsNotNone(X_probe.grad)
            self.assertFalse(torch.isnan(X_probe.grad).any())

            # Also test horizon=1 with terminal_mean + scalarized multi-output GP
            mo_Y = torch.cat([train_Y, 0.5 * train_Y], dim=-1)
            mo_model = SingleTaskGP(train_X=train_X, train_Y=mo_Y)
            mo_model.eval()
            pt = ScalarizedPosteriorTransform(
                weights=torch.tensor([0.7, 0.3], **tkwargs)
            )
            rollout_mo_h1 = LocalSearchRolloutAcquisitionFunction(
                model=mo_model,
                bounds=bounds,
                delta=0.2,
                horizon=1,
                num_fantasies=4,
                reward_mode="terminal_mean",
                posterior_transform=pt,
            )
            X_mo = torch.tensor([[[0.24, 0.22]]], **tkwargs).requires_grad_(True)
            val_mo = rollout_mo_h1(X_mo)
            self.assertEqual(val_mo.shape, torch.Size([1]))
            val_mo.backward()
            self.assertFalse(torch.isnan(X_mo.grad).any())

    def test_rollout_validation(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
        train_X = torch.rand(4, 2, **tkwargs)
        train_Y = torch.rand(4, 1, **tkwargs)
        model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
        model.eval()

        with self.assertRaises(ValueError):
            LocalSearchRolloutAcquisitionFunction(
                model=model, bounds=bounds, delta=0.2, horizon=0
            )
        with self.assertRaises(ValueError):
            LocalSearchRolloutAcquisitionFunction(
                model=model, bounds=bounds, delta=0.2, num_fantasies=0
            )
        with self.assertRaises(ValueError):
            LocalSearchRolloutAcquisitionFunction(
                model=model, bounds=bounds, delta=0.2, discount_factor=1.5
            )
        with self.assertRaises(ValueError):
            LocalSearchRolloutAcquisitionFunction(
                model=model, bounds=bounds, delta=0.2, base_policy="invalid"
            )
        with self.assertRaises(ValueError):
            LocalSearchRolloutAcquisitionFunction(
                model=model, bounds=bounds, delta=0.2, reward_mode="invalid"
            )
        with self.assertRaises(ValueError):
            GaussHermiteNormalSampler(sample_shape=torch.Size([0]))

        mo_model = SingleTaskGP(train_X=train_X, train_Y=torch.rand(4, 2, **tkwargs))
        mo_model.eval()
        with self.assertRaises(UnsupportedError):
            LocalSearchRolloutAcquisitionFunction(
                model=mo_model, bounds=bounds, delta=0.2
            )
        gh = GaussHermiteNormalSampler(sample_shape=torch.Size([4]))
        with self.assertRaises(UnsupportedError):
            gh(mo_model.posterior(train_X[:1]))
