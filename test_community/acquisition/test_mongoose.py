#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy

import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.testing import BotorchTestCase
from botorch_community.acquisition.mongoose import (
    compute_movement_cost,
    meta_train_mongoose,
    MongooseOptimizer,
    MongoosePolicy,
    sample_gp_prior_paths,
)
from gpytorch.kernels import LinearKernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.models import ExactGP


class _ToyExactGP(ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        """Initialize minimal ExactGP for testing GP model prior path sampling.

        Args:
            train_x: Training inputs of shape ``(n, d)``.
            train_y: Training targets of shape ``(n,)``.
        """
        super().__init__(train_x, train_y, GaussianLikelihood())
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())


class _ToyBotorchGP(_ToyExactGP, GPyTorchModel):
    _num_outputs = 1


class TestMongoose(BotorchTestCase):
    def test_mongoose_policy_shapes_bounds_and_conditioning(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.tensor([[-3.0, 1.0, -0.5], [2.0, 5.0, 4.5]], **tkwargs)
            policy = MongoosePolicy(
                input_dim=3,
                bounds=bounds,
                hidden_size=24,
                num_layers=2,
                encoder_hidden_sizes=(16,),
                decoder_hidden_sizes=(16,),
            ).to(**tkwargs)

            # Unbatched single-step forward and bounds adherence
            x0 = torch.tensor([-1.0, 2.5, 0.5], **tkwargs)
            y0 = torch.tensor(0.75, **tkwargs)
            x1, state1 = policy(x=x0, y=y0, tau=0.0)
            self.assertEqual(x1.shape, (3,))
            self.assertEqual(x1.dtype, dtype)
            self.assertEqual(len(state1), 2)
            self.assertEqual(state1[0][0].shape, (1, 24))
            self.assertTrue(torch.all(x1 >= bounds[0]))
            self.assertTrue(torch.all(x1 <= bounds[1]))

            # Hidden state propagation changes next prediction
            y1_obs = torch.tensor([1.2], **tkwargs)
            x2, state2 = policy(x=x1, y=y1_obs, tau=0.25, state=state1)
            x2_fresh, _ = policy(x=x1, y=y1_obs, tau=0.25, state=None)
            self.assertFalse(torch.allclose(x2, x2_fresh))
            self.assertFalse(torch.allclose(state1[1][0], state2[1][0]))

            # Batched rollout with default [0, 1]^d bounds and zero-hidden-layer MLPs
            batch_policy = MongoosePolicy(
                input_dim=2,
                hidden_size=16,
                num_layers=1,
                encoder_hidden_sizes=(),
                decoder_hidden_sizes=(),
                condition_on_alpha=True,
                num_objectives=2,
            ).to(**tkwargs)

            xb = torch.rand(5, 2, **tkwargs)
            yb = torch.randn(5, 2, **tkwargs)
            tau_b = torch.linspace(0.0, 0.8, 5, **tkwargs)
            w_unnorm = torch.tensor([2.0, 6.0], **tkwargs)
            w_norm = torch.tensor([0.25, 0.75], **tkwargs)

            out_low_a, _ = batch_policy(
                x=xb, y=yb, tau=tau_b, alpha=0.0, weights=w_unnorm
            )
            out_high_a, _ = batch_policy(
                x=xb, y=yb, tau=tau_b, alpha=10.0, weights=w_norm
            )
            out_same_w, _ = batch_policy(
                x=xb, y=yb, tau=tau_b, alpha=0.0, weights=w_norm
            )
            self.assertEqual(out_low_a.shape, (5, 2))
            self.assertTrue(torch.all((out_low_a >= 0.0) & (out_low_a <= 1.0)))
            self.assertTrue(torch.allclose(out_low_a, out_same_w, atol=1e-6))
            self.assertFalse(torch.allclose(out_low_a, out_high_a))

            # Default uniform weights when weights=None in multi-objective policy
            out_default_w, _ = batch_policy(
                x=xb,
                y=yb,
                tau=0.5,
                alpha=torch.ones(5, 1, **tkwargs),
                weights=None,
            )
            out_uniform_w, _ = batch_policy(
                x=xb,
                y=yb,
                tau=torch.tensor([0.5], **tkwargs),
                alpha=1.0,
                weights=torch.full((5, 2), 0.5, **tkwargs),
            )
            self.assertTrue(torch.allclose(out_default_w, out_uniform_w, atol=1e-6))

    def test_mongoose_policy_validation_errors(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        with self.assertRaises(ValueError):
            MongoosePolicy(input_dim=0)
        with self.assertRaises(ValueError):
            MongoosePolicy(input_dim=2, hidden_size=0)
        with self.assertRaises(ValueError):
            MongoosePolicy(input_dim=2, num_layers=0)
        with self.assertRaises(ValueError):
            MongoosePolicy(input_dim=2, num_objectives=0)
        with self.assertRaises(ValueError):
            MongoosePolicy(input_dim=2, encoder_hidden_sizes=(0,))
        with self.assertRaises(ValueError):
            MongoosePolicy(input_dim=2, bounds=torch.ones(3, 2, **tkwargs))
        with self.assertRaises(ValueError):
            MongoosePolicy(
                input_dim=2,
                bounds=torch.tensor([[1.0, 0.0], [1.0, 2.0]], **tkwargs),
            )

        policy = MongoosePolicy(
            input_dim=2,
            hidden_size=8,
            num_layers=2,
            num_objectives=2,
            condition_on_alpha=True,
        ).to(**tkwargs)

        z42 = torch.zeros(4, 2, **tkwargs)
        with self.assertRaises(ValueError):
            policy.init_hidden(batch_size=0)
        with self.assertRaises(ValueError):
            policy(
                x=torch.zeros(2, 2, 2, **tkwargs),
                y=torch.zeros(2, 2, **tkwargs),
                tau=0.0,
                alpha=0.1,
            )
        with self.assertRaises(ValueError):
            policy(x=torch.zeros(4, 3, **tkwargs), y=z42, tau=0.0, alpha=0.1)
        with self.assertRaises(ValueError):
            policy(
                x=torch.zeros(2, **tkwargs),
                y=torch.zeros(3, **tkwargs),
                tau=0.0,
                alpha=0.1,
            )
        with self.assertRaises(ValueError):
            policy(x=z42, y=torch.zeros(4, 1, **tkwargs), tau=0.0, alpha=0.1)
        with self.assertRaises(ValueError):
            policy(
                x=z42,
                y=z42,
                tau=torch.zeros(3, **tkwargs),
                alpha=0.1,
            )
        with self.assertRaises(ValueError):
            # Missing alpha when condition_on_alpha=True
            policy(x=z42, y=z42, tau=0.0)
        with self.assertRaises(ValueError):
            # Negative alpha
            policy(x=z42, y=z42, tau=0.0, alpha=-0.5)
        with self.assertRaises(ValueError):
            # Wrong state length
            bad_state = policy.init_hidden(batch_size=4)[:1]
            policy(x=z42, y=z42, tau=0.0, state=bad_state, alpha=0.1)
        with self.assertRaises(ValueError):
            # Wrong state batch size
            bad_state = policy.init_hidden(batch_size=2)
            policy(x=z42, y=z42, tau=0.0, state=bad_state, alpha=0.1)
        with self.assertRaises(ValueError):
            # Negative preference weight
            policy(
                x=z42,
                y=z42,
                tau=0.0,
                alpha=0.1,
                weights=torch.tensor([-0.2, 1.2], **tkwargs),
            )
        with self.assertRaises(ValueError):
            # Zero-sum preference weights
            policy(
                x=z42,
                y=z42,
                tau=0.0,
                alpha=0.1,
                weights=torch.zeros(2, **tkwargs),
            )
        with self.assertRaises(ValueError):
            policy(
                x=z42,
                y=z42,
                tau=0.0,
                alpha=0.1,
                weights=torch.ones(3, **tkwargs),
            )
        with self.assertRaises(ValueError):
            policy(
                x=z42,
                y=z42,
                tau=0.0,
                alpha=0.1,
                weights=torch.ones(2, 2, **tkwargs),
            )
        with self.assertRaises(ValueError):
            policy(
                x=z42,
                y=z42,
                tau=0.0,
                alpha=0.1,
                weights=torch.ones(1, 1, 2, **tkwargs),
            )

    def test_sample_gp_prior_paths_differentiability_and_covariance(self) -> None:
        torch.manual_seed(7)
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.tensor([[-1.0, 0.0], [1.0, 2.0]], **tkwargs)

            for kernel_name in ("rbf", "matern32", "matern52"):
                paths = sample_gp_prior_paths(
                    input_dim=2,
                    batch_size=4,
                    num_objectives=2,
                    kernel=kernel_name,
                    num_rff_features=64,
                    bounds=bounds,
                    **tkwargs,
                )
                x2d = torch.rand(4, 2, **tkwargs).requires_grad_(True)
                y2d = paths(x2d)
                self.assertEqual(y2d.shape, (4, 2))
                y2d.sum().backward()
                self.assertIsNotNone(x2d.grad)
                self.assertTrue(torch.all(torch.isfinite(x2d.grad)))
                self.assertGreater(x2d.grad.abs().sum().item(), 0.0)

                x3d = torch.rand(1, 5, 2, **tkwargs).requires_grad_(True)
                y3d = paths(x3d)
                self.assertEqual(y3d.shape, (4, 5, 2))
                y3d.sum().backward()
                self.assertTrue(torch.all(torch.isfinite(x3d.grad)))

            # Custom GPyTorch Kernel via draw_kernel_feature_paths
            custom_covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2)).to(
                **tkwargs
            )
            k_paths = sample_gp_prior_paths(
                input_dim=2,
                batch_size=3,
                num_objectives=1,
                kernel=custom_covar,
                num_rff_features=64,
                **tkwargs,
            )
            self.assertIsNotNone(k_paths.pathwise_sampler)
            xk = torch.rand(3, 2, **tkwargs).requires_grad_(True)
            yk = k_paths(xk)
            self.assertEqual(yk.shape, (3, 1))
            yk.sum().backward()
            self.assertTrue(torch.all(torch.isfinite(xk.grad)))

            # Custom non-RFF GPyTorch Kernel exercising Nystrom path fallback
            lin_paths = sample_gp_prior_paths(
                input_dim=2,
                batch_size=3,
                num_objectives=1,
                kernel=LinearKernel().to(**tkwargs),
                num_rff_features=32,
                **tkwargs,
            )
            self.assertIsNotNone(lin_paths.nystrom_alpha)
            xl = torch.rand(3, 2, **tkwargs).requires_grad_(True)
            yl = lin_paths(xl)
            self.assertEqual(yl.shape, (3, 1))
            yl.sum().backward()
            self.assertTrue(torch.all(torch.isfinite(xl.grad)))

            # GPyTorch GP and BoTorch GPyTorchModel instances
            for gp_cls in (_ToyExactGP, _ToyBotorchGP):
                gp_model = gp_cls(
                    train_x=torch.rand(3, 2, **tkwargs),
                    train_y=torch.randn(3, **tkwargs),
                ).to(**tkwargs)
                gp_paths = sample_gp_prior_paths(
                    input_dim=2,
                    batch_size=2,
                    num_objectives=1,
                    kernel=gp_model,
                    num_rff_features=32,
                    **tkwargs,
                )
                xg = torch.rand(2, 2, **tkwargs).requires_grad_(True)
                yg = gp_paths(xg)
                self.assertEqual(yg.shape, (2, 1))
                yg.sum().backward()
                self.assertTrue(torch.all(torch.isfinite(xg.grad)))

        # Stationary empirical covariance check against exact GPyTorch kernels
        torch.manual_seed(42)
        tkwargs64 = {"device": self.device, "dtype": torch.double}
        x_grid = torch.tensor(
            [[0.1, 0.2], [0.25, 0.3], [0.6, 0.55], [0.85, 0.9]], **tkwargs64
        )
        ls = 0.4
        os = 1.2
        ref_kernels = {
            "rbf": RBFKernel(),
            "matern32": MaternKernel(nu=1.5),
            "matern52": MaternKernel(nu=2.5),
        }
        for name, base_k in ref_kernels.items():
            ref_k = ScaleKernel(base_k).to(**tkwargs64)
            ref_k.outputscale = os
            ref_k.base_kernel.lengthscale = ls
            exact_cov = ref_k(x_grid, x_grid).to_dense().detach()

            paths = sample_gp_prior_paths(
                input_dim=2,
                batch_size=2000,
                num_objectives=1,
                kernel=name,
                num_rff_features=128,
                lengthscale=ls,
                outputscale=os,
                **tkwargs64,
            )
            vals = paths(x_grid.unsqueeze(0)).squeeze(-1)  # (B, 4)
            emp_cov = (vals.T @ vals) / vals.shape[0]
            self.assertTrue(torch.allclose(emp_cov, exact_cov, atol=0.14))

            # Stationarity: shifted grid has matching empirical covariance
            vals_shifted = paths((x_grid + 0.1).unsqueeze(0)).squeeze(-1)
            emp_cov_shifted = (vals_shifted.T @ vals_shifted) / vals_shifted.shape[0]
            self.assertTrue(torch.allclose(emp_cov, emp_cov_shifted, atol=0.14))

    def test_gp_prior_and_movement_cost_validation(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        with self.assertRaises(ValueError):
            sample_gp_prior_paths(input_dim=0)
        with self.assertRaises(ValueError):
            sample_gp_prior_paths(input_dim=2, batch_size=0)
        with self.assertRaises(ValueError):
            sample_gp_prior_paths(input_dim=2, num_objectives=0)
        with self.assertRaises(ValueError):
            sample_gp_prior_paths(input_dim=2, num_rff_features=3)
        with self.assertRaises(ValueError):
            sample_gp_prior_paths(input_dim=2, kernel="unknown_kernel")
        with self.assertRaises(TypeError):
            sample_gp_prior_paths(input_dim=2, kernel=123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            sample_gp_prior_paths(input_dim=2, lengthscale=-0.1)
        with self.assertRaises(ValueError):
            sample_gp_prior_paths(input_dim=2, lengthscale=torch.tensor([-0.1, 0.2]))
        with self.assertRaises(ValueError):
            sample_gp_prior_paths(input_dim=2, lengthscale=torch.ones(5))
        with self.assertRaises(ValueError):
            sample_gp_prior_paths(input_dim=2, lengthscale_range=(0.5, 0.2))

        p = sample_gp_prior_paths(input_dim=2, batch_size=3, **tkwargs)
        with self.assertRaises(ValueError):
            p(torch.zeros(2, 2, **tkwargs))
        with self.assertRaises(ValueError):
            p(torch.zeros(2, 4, 2, **tkwargs))
        with self.assertRaises(ValueError):
            p(torch.zeros(2, **tkwargs))

        # Movement cost metrics, zero-displacement gradient, and validations
        x_a = torch.tensor([[1.0, 2.0], [0.5, 0.5]], **tkwargs, requires_grad=True)
        x_b = x_a.detach().clone()
        c_zero = compute_movement_cost(x_next=x_a, x_curr=x_b, cost_exponent=1.0)
        c_zero.sum().backward()
        self.assertTrue(torch.all(torch.isfinite(x_a.grad)))
        self.assertTrue(torch.allclose(x_a.grad, torch.zeros_like(x_a)))

        w_diag = torch.tensor([4.0, 9.0], **tkwargs)
        w_mat = torch.diag(w_diag)
        x_next = torch.tensor([[2.0, 4.0]], **tkwargs)
        x_curr = torch.tensor([[1.0, 2.0]], **tkwargs)
        c_diag = compute_movement_cost(
            x_next, x_curr, movement_weights=w_diag, cost_exponent=2.0
        )
        c_mat = compute_movement_cost(
            x_next, x_curr, movement_metric=w_mat, cost_exponent=2.0
        )
        # 4 * (1)^2 + 9 * (2)^2 = 40
        self.assertAlmostEqual(c_diag.item(), 40.0, places=6)
        self.assertAlmostEqual(c_mat.item(), 40.0, places=6)

        with self.assertRaises(ValueError):
            compute_movement_cost(x_next, torch.zeros(2, 2, **tkwargs))
        with self.assertRaises(ValueError):
            compute_movement_cost(
                x_next, x_curr, movement_weights=w_diag, movement_metric=w_mat
            )
        with self.assertRaises(ValueError):
            compute_movement_cost(x_next, x_curr, cost_exponent=0.0)
        with self.assertRaises(ValueError):
            compute_movement_cost(x_next, x_curr, smooth_eps=0.0)
        with self.assertRaises(ValueError):
            compute_movement_cost(
                x_next, x_curr, movement_weights=torch.tensor([-1.0, 1.0], **tkwargs)
            )
        with self.assertRaises(ValueError):
            compute_movement_cost(
                x_next, x_curr, movement_weights=torch.ones(3, **tkwargs)
            )
        with self.assertRaises(ValueError):
            # Non-symmetric matrix
            compute_movement_cost(
                x_next,
                x_curr,
                movement_metric=torch.tensor([[1.0, 2.0], [0.0, 1.0]], **tkwargs),
            )
        with self.assertRaises(ValueError):
            # Symmetric but non-SPD matrix
            compute_movement_cost(
                x_next,
                x_curr,
                movement_metric=torch.tensor([[1.0, 2.0], [2.0, 1.0]], **tkwargs),
            )
        with self.assertRaises(ValueError):
            compute_movement_cost(
                x_next, x_curr, movement_metric=torch.eye(3, **tkwargs)
            )

    def test_meta_train_mongoose_loss_and_smoothness(self) -> None:
        for dtype in (torch.float, torch.double):
            torch.manual_seed(123)
            tkwargs = {"device": self.device, "dtype": dtype}
            base_policy = MongoosePolicy(
                input_dim=2,
                hidden_size=32,
                num_layers=1,
            ).to(**tkwargs)
            policy_free = copy.deepcopy(base_policy)
            policy_smooth = copy.deepcopy(base_policy)

            anchor_x0 = torch.tensor([0.15, 0.15], **tkwargs)

            # Held-out evaluation tasks for clean before/after & alpha comparison
            eval_paths = sample_gp_prior_paths(
                input_dim=2,
                batch_size=32,
                kernel="rbf",
                num_rff_features=128,
                lengthscale=0.25,
                **tkwargs,
            )
            eval_x0 = anchor_x0.unsqueeze(0).expand(32, 2)

            def _eval_rollout(
                pol: MongoosePolicy, alpha_pen: float
            ) -> tuple[float, float]:
                with torch.no_grad():
                    xt = eval_x0.clone()
                    st = pol.init_hidden(batch_size=32, **tkwargs)
                    vals = [eval_paths(xt).squeeze(-1)]
                    dists = []
                    for t in range(5):
                        xn, st = pol(xt, eval_paths(xt), tau=t / 5, state=st)
                        dists.append(compute_movement_cost(xn, xt))
                        xt = xn
                        vals.append(eval_paths(xt).squeeze(-1))
                    best_f = torch.stack(vals, dim=-1).amax(dim=-1)
                    path_len = torch.stack(dists, dim=-1).sum(dim=-1)
                    meta_obj = (-best_f + alpha_pen * path_len).mean().item()
                    return meta_obj, path_len.mean().item()

            init_loss_smooth, _ = _eval_rollout(policy_smooth, alpha_pen=5.0)
            init_loss_free, _ = _eval_rollout(policy_free, alpha_pen=0.0)

            res_free = meta_train_mongoose(
                policy=policy_free,
                num_steps=15,
                batch_size=16,
                horizon=5,
                alpha=0.0,
                lr=1e-2,
                tau_max=0.1,
                num_rff_features=64,
                kernel="rbf",
                lengthscale=0.25,
                x0=anchor_x0,
            )
            res_smooth = meta_train_mongoose(
                policy=policy_smooth,
                num_steps=15,
                batch_size=16,
                horizon=5,
                alpha=5.0,
                lr=1e-2,
                tau_max=None,  # exercises exact torch.amax branch
                num_rff_features=64,
                kernel="rbf",
                lengthscale=0.25,
                x0=anchor_x0,
            )

            self.assertEqual(len(res_free.losses), 15)
            self.assertEqual(len(res_smooth.losses), 15)
            self.assertLess(res_smooth.losses[-1], res_smooth.losses[0])

            final_loss_smooth, path_len_smooth = _eval_rollout(
                policy_smooth, alpha_pen=5.0
            )
            final_loss_free, path_len_free = _eval_rollout(policy_free, alpha_pen=0.0)

            self.assertLess(final_loss_smooth, init_loss_smooth)
            self.assertLess(final_loss_free, init_loss_free)
            # High movement penalty alpha=5.0 yields strictly shorter paths
            self.assertLess(path_len_smooth, path_len_free)
            self.assertLess(res_smooth.path_lengths[-1], res_free.path_lengths[-1])

        # Test alpha_range + uniform random x0 + multi-objective Chebyshev
        tkwargs64 = {"device": self.device, "dtype": torch.double}
        mo_policy = MongoosePolicy(
            input_dim=2,
            hidden_size=16,
            num_objectives=2,
            condition_on_alpha=True,
        ).to(**tkwargs64)
        res_mo = meta_train_mongoose(
            policy=mo_policy,
            num_steps=4,
            batch_size=8,
            horizon=3,
            alpha_range=(0.0, 2.0),
            lr=5e-3,
            tau_max=0.0,
            noise_std=0.0,
            num_rff_features=32,
            x0=None,
            scalarization="chebyshev",
            cost_exponent=2.0,
        )
        self.assertEqual(len(res_mo.losses), 4)

        # Validation branches for meta_train_mongoose
        with self.assertRaises(ValueError):
            meta_train_mongoose(mo_policy, num_steps=0)
        with self.assertRaises(ValueError):
            meta_train_mongoose(mo_policy, batch_size=0)
        with self.assertRaises(ValueError):
            meta_train_mongoose(mo_policy, horizon=0)
        with self.assertRaises(ValueError):
            meta_train_mongoose(mo_policy, alpha=-1.0)
        with self.assertRaises(ValueError):
            meta_train_mongoose(mo_policy, alpha_range=(2.0, 1.0))
        with self.assertRaises(ValueError):
            meta_train_mongoose(mo_policy, lr=0.0)
        with self.assertRaises(ValueError):
            meta_train_mongoose(mo_policy, max_grad_norm=0.0)
        with self.assertRaises(ValueError):
            meta_train_mongoose(mo_policy, tau_max=-0.1)
        with self.assertRaises(ValueError):
            meta_train_mongoose(mo_policy, noise_std=-0.1)
        with self.assertRaises(ValueError):
            meta_train_mongoose(mo_policy, scalarization="invalid")
        with self.assertRaises(ValueError):
            meta_train_mongoose(mo_policy, x0=torch.ones(5, **tkwargs64))
        with self.assertRaises(ValueError):
            # x0 outside bounds
            meta_train_mongoose(mo_policy, x0=torch.tensor([-0.5, 0.5], **tkwargs64))

    def test_mongoose_optimizer_stateful_matches_stateless_replay(self) -> None:
        for dtype in (torch.float, torch.double):
            torch.manual_seed(99)
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.tensor([[-2.0, 0.0], [2.0, 4.0]], **tkwargs)
            policy = MongoosePolicy(
                input_dim=2,
                bounds=bounds,
                hidden_size=24,
                num_layers=2,
                num_objectives=2,
                condition_on_alpha=True,
            ).to(**tkwargs)

            weights = torch.tensor([0.3, 0.7], **tkwargs)
            opt = MongooseOptimizer(
                policy=policy,
                horizon=4,
                alpha=1.5,
                weights=weights,
                maximize=False,
                y_shift=1.0,
                y_scale=2.0,
            )
            self.assertIsNone(opt.X_observed)
            self.assertIsNone(opt.Y_observed)
            self.assertIsNone(opt.pending_x)
            self.assertEqual(opt.step_count, 0)

            # Unbatched loop vs stateless suggest() across 6 steps (> horizon)
            x0 = torch.tensor([0.0, 1.5], **tkwargs)
            y0 = torch.tensor([1.2, -0.4], **tkwargs)
            x_curr = opt.reset(x0, y0)

            x_hist = [x0]
            y_hist = [y0]
            replay_x = opt.suggest(
                X_observed=torch.stack(x_hist, dim=0),
                Y_observed=torch.stack(y_hist, dim=0),
            )
            self.assertTrue(torch.allclose(x_curr, replay_x, atol=1e-6))

            for step_idx in range(5):
                y_obs = torch.stack([x_curr.sum(), (x_curr**2).sum()])
                # Exercise x_new actuator override on one step
                x_actual = (
                    torch.clamp(x_curr + 0.05, min=bounds[0], max=bounds[1])
                    if step_idx == 2
                    else None
                )
                x_recorded = x_curr if x_actual is None else x_actual
                x_next = opt.step(y_new=y_obs, x_new=x_actual)

                x_hist.append(x_recorded)
                y_hist.append(y_obs)
                replay_next = opt.suggest(
                    X_observed=torch.stack(x_hist, dim=0),
                    Y_observed=torch.stack(y_hist, dim=0),
                )
                self.assertTrue(torch.allclose(x_next, replay_next, atol=1e-6))
                x_curr = x_next

            self.assertEqual(opt.step_count, 6)
            self.assertEqual(opt.X_observed.shape, (6, 2))
            self.assertEqual(opt.Y_observed.shape, (6, 2))

            # Batched single-objective loop vs stateless suggest()
            so_policy = MongoosePolicy(input_dim=2, hidden_size=16).to(**tkwargs)
            so_opt = MongooseOptimizer(policy=so_policy, horizon=5)
            xb0 = torch.rand(3, 2, **tkwargs)
            yb0 = torch.randn(3, **tkwargs)
            xb1 = so_opt.reset(xb0, yb0)
            self.assertTrue(
                torch.allclose(
                    xb1,
                    so_opt.suggest(xb0.unsqueeze(1), yb0.unsqueeze(1)),
                    atol=1e-6,
                )
            )
            yb1 = torch.randn(3, **tkwargs)
            xb2 = so_opt.step(yb1)
            xb_seq = torch.stack([xb0, xb1], dim=1)
            yb_seq = torch.stack([yb0, yb1], dim=1)
            self.assertTrue(
                torch.allclose(xb2, so_opt.suggest(xb_seq, yb_seq), atol=1e-6)
            )
            self.assertEqual(so_opt.X_observed.shape, (3, 2, 2))
            self.assertEqual(so_opt.Y_observed.shape, (3, 2))

            # Unbatched 1D Y_observed for single-objective suggest
            self.assertEqual(
                so_opt.suggest(xb_seq[0], yb_seq[0]).shape,
                (2,),
            )

    def test_mongoose_optimizer_validation_errors(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        policy = MongoosePolicy(input_dim=2, hidden_size=8).to(**tkwargs)

        with self.assertRaises(ValueError):
            MongooseOptimizer(policy=policy, horizon=0)
        with self.assertRaises(ValueError):
            MongooseOptimizer(policy=policy, y_scale=0.0)
        with self.assertRaises(ValueError):
            MongooseOptimizer(
                policy=policy, y_scale=torch.tensor([1.0, -1.0], **tkwargs)
            )

        opt = MongooseOptimizer(policy=policy, horizon=5)
        with self.assertRaises(RuntimeError):
            opt.step(torch.tensor(0.5, **tkwargs))
        with self.assertRaises(ValueError):
            opt.reset(torch.zeros(2, 2, 2, **tkwargs), torch.zeros(2, **tkwargs))

        opt.reset(torch.zeros(2, **tkwargs), torch.tensor(0.1, **tkwargs))
        with self.assertRaises(ValueError):
            opt.step(
                torch.tensor(0.2, **tkwargs),
                x_new=torch.zeros(3, **tkwargs),
            )

        with self.assertRaises(ValueError):
            opt.suggest(
                torch.zeros(2, 2, **tkwargs),
                torch.zeros(2, 1, **tkwargs),
                horizon=0,
            )
        with self.assertRaises(ValueError):
            opt.suggest(torch.zeros(2, **tkwargs), torch.zeros(2, **tkwargs))
        with self.assertRaises(ValueError):
            opt.suggest(torch.zeros(0, 2, **tkwargs), torch.zeros(0, 1, **tkwargs))
        with self.assertRaises(ValueError):
            opt.suggest(torch.zeros(2, 3, **tkwargs), torch.zeros(2, 1, **tkwargs))
        with self.assertRaises(ValueError):
            opt.suggest(torch.zeros(2, 2, **tkwargs), torch.zeros(3, 1, **tkwargs))
        with self.assertRaises(ValueError):
            opt.suggest(
                torch.zeros(2, 3, 2, **tkwargs),
                torch.zeros(2, 4, 1, **tkwargs),
            )
