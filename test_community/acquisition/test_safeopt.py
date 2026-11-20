#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf_discrete
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior_list import PosteriorList
from botorch.posteriors.torch import TorchPosterior
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from botorch_community.acquisition.safeopt import (
    _extract_all_output_covariances,
    SafeOpt,
    SafeOptSets,
)
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from torch import Tensor


def _build_fixed_gp(
    train_X: Tensor,
    train_Y: Tensor,
    lengthscale: float = 0.25,
    outputscale: float = 1.0,
    noise: float = 1e-4,
    standardize: bool = False,
) -> SingleTaskGP:
    r"""Create a deterministic SingleTaskGP with fixed hyperparameters for testing."""
    m = train_Y.shape[-1]
    batch_shape = torch.Size([m]) if m > 1 else torch.Size()
    covar_module = ScaleKernel(
        RBFKernel(batch_shape=batch_shape, ard_num_dims=train_X.shape[-1]),
        batch_shape=batch_shape,
    )
    covar_module.base_kernel.lengthscale = lengthscale
    covar_module.outputscale = outputscale
    likelihood = GaussianLikelihood(batch_shape=batch_shape)
    likelihood.noise = noise
    outcome_transform = Standardize(m=m) if standardize else None
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        likelihood=likelihood,
        mean_module=ConstantMean(batch_shape=batch_shape),
        covar_module=covar_module,
        outcome_transform=outcome_transform,
    )
    model.eval()
    return model


class TestSafeOpt(BotorchTestCase):
    def test_lipschitz_safeopt_closed_loop_1d_and_2d(self) -> None:
        r"""Test Sui et al. (2015) Lipschitz SafeOpt in 1D and 2D closed loops."""
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}

            # 1D concave quadratic: f(x) = 1 - 4 * (x - 0.6)^2, safe for f(x) >= 0.0
            # True safe interval is [0.1, 1.0]; global max is at x = 0.6 (f = 1.0).
            # Lipschitz bound on [0, 1] is max |f'(x)| = 8 * 0.6 = 4.8; use L = 3.0
            # locally with beta = 1.5 on a 41-point grid.
            grid_1d = torch.linspace(0.0, 1.0, 41, **tkwargs).unsqueeze(-1)

            def f_1d(x: Tensor) -> Tensor:
                return 1.0 - 4.0 * (x - 0.6).pow(2)

            threshold_1d = 0.0
            # Start from known safe seed x = 0.35 (f(0.35) = 0.75 > 0)
            seed_idx = 14
            for monotonic in (False, True):
                with self.subTest(dim=1, dtype=dtype, monotonic=monotonic):
                    train_X = grid_1d[[seed_idx]]
                    train_Y = f_1d(train_X)
                    model = _build_fixed_gp(
                        train_X, train_Y, lengthscale=0.3, outputscale=1.0, noise=1e-4
                    )
                    opt = SafeOpt(
                        model=model,
                        candidate_set=grid_1d,
                        safety_thresholds=threshold_1d,
                        initial_safe_indices=[seed_idx],
                        beta=1.5,
                        lipschitz=3.0,
                        monotonic_bounds=monotonic,
                    )

                    initial_safe_count = int(opt.compute_sets().safe_mask.sum().item())
                    prev_safe_mask = opt.compute_sets().safe_mask.clone()
                    prev_lower = opt.compute_sets().lower.clone()
                    prev_upper = opt.compute_sets().upper.clone()

                    for _ in range(10):
                        x_next = opt.next_point()
                        self.assertEqual(x_next.shape, torch.Size([1, 1]))
                        y_next = f_1d(x_next)
                        # Strict safety: never query an unsafe point!
                        self.assertGreaterEqual(float(y_next.item()), threshold_1d)

                        train_X = torch.cat([train_X, x_next], dim=0)
                        train_Y = torch.cat([train_Y, y_next], dim=0)
                        model = _build_fixed_gp(
                            train_X,
                            train_Y,
                            lengthscale=0.3,
                            outputscale=1.0,
                            noise=1e-4,
                        )
                        opt.update(model, x_current=x_next)
                        sets = opt.compute_sets()
                        if monotonic:
                            # S_{t-1} \subseteq S_t and C_t \subseteq C_{t-1}
                            self.assertTrue((sets.safe_mask | ~prev_safe_mask).all())
                            self.assertTrue((sets.lower >= prev_lower - 1e-6).all())
                            self.assertTrue((sets.upper <= prev_upper + 1e-6).all())
                        prev_safe_mask = sets.safe_mask.clone()
                        prev_lower = sets.lower.clone()
                        prev_upper = sets.upper.clone()

                    final_sets = opt.compute_sets()
                    self.assertIsInstance(final_sets, SafeOptSets)
                    self.assertGreater(
                        int(final_sets.safe_mask.sum().item()), initial_safe_count
                    )
                    best_x = opt.get_best_safe_point()
                    self.assertEqual(best_x.shape, torch.Size([1, 1]))
                    self.assertGreater(
                        float(f_1d(best_x).item()),
                        float(f_1d(grid_1d[[seed_idx]]).item()),
                    )

            # 2D closed-loop Lipschitz SafeOpt test
            coords = torch.linspace(0.0, 1.0, 9, **tkwargs)
            g0, g1 = torch.meshgrid(coords, coords, indexing="ij")
            grid_2d = torch.stack([g0.reshape(-1), g1.reshape(-1)], dim=-1)

            def f_2d(x: Tensor) -> Tensor:
                center = torch.tensor([[0.65, 0.65]], **tkwargs)
                return 1.0 - 2.5 * (x - center).pow(2).sum(dim=-1, keepdim=True)

            seed_pt_2d = torch.tensor([[0.5, 0.5]], **tkwargs)
            train_X2 = seed_pt_2d.clone()
            train_Y2 = f_2d(train_X2)
            model2 = _build_fixed_gp(
                train_X2, train_Y2, lengthscale=0.4, outputscale=1.0, noise=1e-4
            )
            opt2 = SafeOpt(
                model=model2,
                candidate_set=grid_2d,
                safety_thresholds=0.0,
                safe_seed=seed_pt_2d,
                beta=1.5,
                lipschitz=2.2,
                monotonic_bounds=True,
            )
            init_safe_2d = int(opt2.compute_sets().safe_mask.sum().item())
            for _ in range(8):
                x_next = opt2.next_point()
                self.assertEqual(x_next.shape, torch.Size([1, 2]))
                y_next = f_2d(x_next)
                self.assertGreaterEqual(float(y_next.item()), 0.0)
                train_X2 = torch.cat([train_X2, x_next], dim=0)
                train_Y2 = torch.cat([train_Y2, y_next], dim=0)
                model2 = _build_fixed_gp(
                    train_X2, train_Y2, lengthscale=0.4, outputscale=1.0, noise=1e-4
                )
                opt2.update(model2)

            self.assertGreater(
                int(opt2.compute_sets().safe_mask.sum().item()), init_safe_2d
            )

    def test_gp_fantasy_expanders_match_condition_on_observations(self) -> None:
        r"""Verify rank-1 GP fantasy expanders match condition_on_observations."""
        torch.manual_seed(42)
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            grid = torch.linspace(0.0, 1.0, 21, **tkwargs).unsqueeze(-1)
            train_X = grid[[4, 10, 14]]
            beta = 2.25

            # 1) Single-output SingleTaskGP (with Standardize outcome transform)
            train_X = grid[[2, 10, 18]]
            train_Y_1 = torch.tensor([[-0.4], [1.0], [-0.3]], **tkwargs)
            model_1 = _build_fixed_gp(
                train_X,
                train_Y_1,
                lengthscale=0.22,
                outputscale=0.8,
                noise=1e-3,
                standardize=True,
            )
            opt_1 = SafeOpt(
                model=model_1,
                candidate_set=grid,
                safety_thresholds=0.1,
                initial_safe_indices=[10],
                beta=beta,
                lipschitz=None,
            )
            self._assert_expanders_match_fantasy_conditioning(
                opt_1, is_model_list=False
            )

            # 2) Multi-output SingleTaskGP (m = 2) with len(thresholds) == m and m - 1
            train_Y_2 = torch.tensor(
                [
                    [-0.3, -0.4],
                    [1.1, 0.9],
                    [-0.2, -0.3],
                ],
                **tkwargs,
            )
            for standardize in (False, True):
                model_st2 = _build_fixed_gp(
                    train_X,
                    train_Y_2,
                    lengthscale=0.24,
                    outputscale=0.8,
                    noise=2e-3,
                    standardize=standardize,
                )
                for thresholds in ([0.1, 0.1], [0.1], 0.1):
                    with self.subTest(
                        model="SingleTaskGP_2out",
                        dtype=dtype,
                        standardize=standardize,
                        thresholds=thresholds,
                    ):
                        opt_st2 = SafeOpt(
                            model=model_st2,
                            candidate_set=grid,
                            safety_thresholds=thresholds,
                            initial_safe_indices=[10],
                            beta=beta,
                            lipschitz=None,
                        )
                        self._assert_expanders_match_fantasy_conditioning(
                            opt_st2, is_model_list=False
                        )

            # 3) ModelListGP (m = 2) with len(thresholds) in {m, m - 1}
            for standardize in (False, True):
                gp_obj = _build_fixed_gp(
                    train_X,
                    train_Y_2[:, :1],
                    lengthscale=0.26,
                    outputscale=0.9,
                    noise=1e-3,
                    standardize=standardize,
                )
                gp_con = _build_fixed_gp(
                    train_X,
                    train_Y_2[:, 1:2],
                    lengthscale=0.22,
                    outputscale=0.7,
                    noise=3e-3,
                    standardize=standardize,
                )
                model_list = ModelListGP(gp_obj, gp_con)
                model_list.eval()

                for thresholds in ([0.1, 0.1], [0.1]):
                    with self.subTest(
                        model="ModelListGP",
                        dtype=dtype,
                        standardize=standardize,
                        thresholds=thresholds,
                    ):
                        opt_ml = SafeOpt(
                            model=model_list,
                            candidate_set=grid,
                            safety_thresholds=thresholds,
                            initial_safe_indices=[10],
                            beta=beta,
                            lipschitz=None,
                            scaling="auto",
                        )
                        self._assert_expanders_match_fantasy_conditioning(
                            opt_ml, is_model_list=True
                        )

    def _assert_expanders_match_fantasy_conditioning(
        self, opt: SafeOpt, is_model_list: bool
    ) -> None:
        sets = opt.compute_sets()
        safe_idx = sets.safe_mask.nonzero(as_tuple=True)[0]
        unsafe_idx = (~sets.safe_mask).nonzero(as_tuple=True)[0]
        self.assertGreater(safe_idx.numel(), 0)
        self.assertGreater(unsafe_idx.numel(), 0)

        X_unsafe = opt.candidate_set[unsafe_idx]
        sqrt_beta = float(opt._get_sqrt_beta(opt.step).item())
        c_dims = opt.constrained_dims
        thresholds = opt.safety_thresholds

        expected_expanders = torch.zeros_like(sets.safe_mask)
        for i in safe_idx.tolist():
            x_i = opt.candidate_set[i : i + 1]
            y_i = sets.upper[i : i + 1, :]
            if is_model_list:
                fantasy_model = opt.model.condition_on_observations(
                    X=[x_i] * opt.num_outputs, Y=y_i
                )
            else:
                fantasy_model = opt.model.condition_on_observations(X=x_i, Y=y_i)
            f_post = fantasy_model.posterior(X_unsafe, observation_noise=False)
            f_mean = f_post.mean.reshape(X_unsafe.shape[0], opt.num_outputs)
            f_var = f_post.variance.reshape(
                X_unsafe.shape[0], opt.num_outputs
            ).clamp_min(0.0)
            f_lower = f_mean - sqrt_beta * f_var.sqrt()
            can_certify = (f_lower[:, c_dims] >= thresholds.unsqueeze(0)).all(dim=-1)
            expected_expanders[i] = can_certify.any()

        self.assertTrue(sets.expanders_mask.any())
        self.assertTrue(
            torch.equal(sets.expanders_mask, expected_expanders),
            msg=(
                f"Vectorized expanders {sets.expanders_mask.nonzero().flatten()} != "
                f"conditioned expanders {expected_expanders.nonzero().flatten()}"
            ),
        )

    def test_safeopt_mc_closed_loop_decoupled_constraint(self) -> None:
        r"""Closed-loop SafeOpt-MC with decoupled safety constraint."""
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            grid = torch.linspace(0.0, 1.0, 31, **tkwargs).unsqueeze(-1)

            # Objective can be negative while constraint g(x) >= 0 must hold.
            # Objective peaks toward x = 0.75; constraint safe on [0.15, 0.80].
            def eval_problem(x: Tensor) -> Tensor:
                obj = -2.0 + 3.0 * x - 1.5 * (x - 0.75).pow(2)
                con = 0.6 - 5.0 * (x - 0.475).pow(2)
                return torch.cat([obj, con], dim=-1)

            seed_idx = 14  # x ~= 0.467, con ~= 0.60 > 0, obj < 0
            train_X = grid[[seed_idx]]
            train_Y = eval_problem(train_X)
            model = _build_fixed_gp(
                train_X, train_Y, lengthscale=0.25, outputscale=0.8, noise=1e-4
            )
            opt = SafeOpt(
                model=model,
                candidate_set=grid,
                safety_thresholds=[0.0],
                initial_safe_indices=[seed_idx],
                beta=1.8,
                lipschitz=None,
                scaling=[1.5, 1.0],
            )
            self.assertEqual(opt.constrained_dims, [1])
            init_safe = int(opt.compute_sets().safe_mask.sum().item())

            for _ in range(8):
                x_next = opt.next_point()
                y_next = eval_problem(x_next)
                # Constraint output 1 stays strictly safe (>= 0.0)
                self.assertGreaterEqual(float(y_next[0, 1].item()), 0.0)
                train_X = torch.cat([train_X, x_next], dim=0)
                train_Y = torch.cat([train_Y, y_next], dim=0)
                model = _build_fixed_gp(
                    train_X, train_Y, lengthscale=0.25, outputscale=0.8, noise=1e-4
                )
                opt.update(model)

            self.assertGreater(
                int(opt.compute_sets().safe_mask.sum().item()), init_safe
            )
            best_x = opt.get_best_safe_point()
            best_y = eval_problem(best_x)
            self.assertGreaterEqual(float(best_y[0, 1].item()), 0.0)
            self.assertGreater(
                float(best_y[0, 0].item()), float(eval_problem(grid[[seed_idx]])[0, 0])
            )

    def test_max_step_transition_and_fallback(self) -> None:
        r"""Test local transition cap `max_step` and fallback to safe set."""
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            grid = torch.linspace(0.0, 1.0, 11, **tkwargs).unsqueeze(-1)
            # Construct a model where all points [0.0 .. 0.6] are safe, but only
            # points near 0.6 are maximizers/expanders while x_current is at 0.0.
            train_X = grid[[0, 1, 2, 6]]
            train_Y = torch.tensor([[0.3], [0.3], [0.3], [1.5]], **tkwargs)
            model = _build_fixed_gp(
                train_X, train_Y, lengthscale=0.25, outputscale=0.5, noise=1e-4
            )
            opt = SafeOpt(
                model=model,
                candidate_set=grid,
                safety_thresholds=0.0,
                initial_safe_indices=[0, 1, 2, 6],
                beta=1.0,
                lipschitz=2.0,
                x_current=grid[0],
                max_step=0.15,
            )
            sets = opt.compute_sets()
            # Ensure at least x=0.0 and x=0.1 are safe, while x=0.6 dominates M_t/G_t
            self.assertTrue(sets.safe_mask[0].item() and sets.safe_mask[1].item())
            self.assertFalse(
                (sets.maximizers_mask[:2] | sets.expanders_mask[:2]).any().item()
            )

            # Fallback to {x in S_t : ||x - x_current|| <= 0.15} -> indices {0, 1}
            x_fallback = opt.next_point()
            self.assertLessEqual(
                float(torch.linalg.vector_norm(x_fallback - grid[[0]]).item()),
                0.15 + 1e-6,
            )

            # When max_step reaches an active expander/maximizer, step stays <= max_step
            x_medium = opt.next_point(x_current=grid[5], max_step=0.15)
            self.assertLessEqual(
                float(torch.linalg.vector_norm(x_medium - grid[[5]]).item()),
                0.15 + 1e-6,
            )

            # When no safe point exists within max_step of an out-of-range x_current:
            with self.assertRaisesRegex(RuntimeError, "No certified safe candidate"):
                opt.next_point(
                    x_current=torch.tensor([5.0], **tkwargs),
                    max_step=0.05,
                )

    def test_callable_beta_forward_and_empty_unsafe_set(self) -> None:
        r"""Test callable beta, S_t == X edge case, and forward / discrete opt."""
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            grid = torch.linspace(0.0, 1.0, 9, **tkwargs).unsqueeze(-1)
            train_X = grid[[0, 4, 8]]
            train_Y = torch.tensor([[2.0], [2.5], [2.2]], **tkwargs)
            model = _build_fixed_gp(
                train_X, train_Y, lengthscale=0.5, outputscale=0.2, noise=1e-4
            )

            called_steps: list[int] = []

            def beta_schedule(t: int) -> float:
                called_steps.append(t)
                return 1.0 + 0.5 * math.log(t + 1)

            # Threshold -10.0 certifies the entire grid: S_t == X (empty unsafe set)
            for lipschitz in (None, 2.0):
                opt = SafeOpt(
                    model=model,
                    candidate_set=grid,
                    safety_thresholds=-10.0,
                    initial_safe_indices=torch.tensor(
                        [True] + [False] * 8, device=self.device
                    ),
                    beta=beta_schedule,
                    lipschitz=lipschitz,
                )
                self.assertIn(1, called_steps)
                sets = opt.compute_sets()
                self.assertTrue(sets.safe_mask.all())
                self.assertFalse(sets.expanders_mask.any())
                self.assertTrue(sets.maximizers_mask.any())
                self.assertEqual(opt.safe_set.shape, grid.shape)
                self.assertEqual(opt.expanders.shape, torch.Size([0, 1]))
                self.assertGreater(opt.maximizers.shape[0], 0)

                # Compare next_point() with optimize_acqf_discrete on forward()
                x_next = opt.next_point()
                cand_disc, acq_val = optimize_acqf_discrete(
                    acq_function=opt, q=1, choices=grid
                )
                self.assertAllClose(x_next, cand_disc)
                self.assertTrue(torch.isfinite(acq_val).all())

                # Querying an off-grid point in forward() returns -inf
                off_grid = torch.tensor([[[0.12345]]], **tkwargs)
                self.assertEqual(float(opt(off_grid).item()), -math.inf)

                # Advance step via update() and check callable beta receives step=2
                opt.update(model)
                self.assertEqual(opt.step, 2)
                self.assertIn(2, called_steps)

    def test_safe_seed_off_grid_and_lipschitz_vector_and_posterior_list(self) -> None:
        r"""Test off-grid safe_seed, vector Lipschitz, and PosteriorList."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        grid = torch.linspace(0.0, 1.0, 10, **tkwargs).unsqueeze(-1)
        train_X = grid[[4, 5]]
        train_Y = torch.tensor([[1.0, 0.8], [1.1, 0.9]], **tkwargs)
        model = _build_fixed_gp(train_X, train_Y, lengthscale=0.3, outputscale=0.5)

        # 0.5 is not an exact point on a 10-point grid; nearest grid point is selected
        opt = SafeOpt(
            model=model,
            candidate_set=grid,
            safety_thresholds=[0.2, 0.2],
            safe_seed=torch.tensor([0.5], **tkwargs),
            beta=torch.tensor([2.0, 1.5], **tkwargs),
            lipschitz=[2.0, 1.5],
            monotonic_bounds=True,
            prev_safe_mask=torch.tensor(
                [False, False, False, False, True, True, False, False, False, False],
                device=self.device,
            ),
            prev_lower=torch.full((10, 2), -10.0, **tkwargs),
            prev_upper=torch.full((10, 2), 10.0, **tkwargs),
        )
        self.assertTrue(opt.initial_safe_mask.any())
        sets = opt.compute_sets()
        self.assertTrue(sets.safe_mask.any())

        # Exercise PosteriorList and non-interleaved MultitaskMultivariateNormal helper
        p_single = _build_fixed_gp(train_X, train_Y[:, :1]).posterior(grid)
        plist = PosteriorList(p_single, p_single)
        covs = _extract_all_output_covariances(plist, num_candidates=10)
        self.assertEqual(covs.shape, torch.Size([2, 10, 10]))

        # Non-interleaved MultitaskMultivariateNormal branch
        mtmvn_non_interleaved = MultitaskMultivariateNormal(
            mean=torch.zeros(10, 2, **tkwargs),
            covariance_matrix=torch.eye(20, **tkwargs),
            interleaved=False,
        )
        post_non_interleaved = GPyTorchPosterior(distribution=mtmvn_non_interleaved)
        covs_ni = _extract_all_output_covariances(
            post_non_interleaved, num_candidates=10
        )
        self.assertEqual(covs_ni.shape, torch.Size([2, 10, 10]))
        self.assertAllClose(covs_ni[0], torch.eye(10, **tkwargs))
        self.assertAllClose(covs_ni[1], torch.eye(10, **tkwargs))

        # Auto-scaling fallback on model without Standardize or ScaleKernel
        plain_gp = SingleTaskGP(train_X, train_Y[:, :1], outcome_transform=None)
        plain_gp.covar_module = RBFKernel()
        plain_gp.eval()
        opt_plain = SafeOpt(
            model=plain_gp,
            candidate_set=grid,
            safety_thresholds=[0.0],
            initial_safe_indices=[4],
            lipschitz=[1.5],
            scaling="auto",
        )
        self.assertAllClose(opt_plain.scaling, torch.ones(1, **tkwargs))
        opt_plain.update(plain_gp)

        # Scalar outputscale on 2-output model + 0D tensor threshold + length-m
        # lipschitz with m-1 thresholds + length-1 scaling + vector beta with GP-only
        shared_scale_gp = _build_fixed_gp(train_X, train_Y, standardize=False)
        shared_scale_gp.covar_module = ScaleKernel(RBFKernel())
        shared_scale_gp.covar_module.outputscale = 4.0
        shared_scale_gp.eval()
        opt_shared = SafeOpt(
            model=shared_scale_gp,
            candidate_set=grid,
            safety_thresholds=torch.tensor(0.1, **tkwargs),
            initial_safe_indices=[4],
            lipschitz=[2.0, 1.5],
            scaling="auto",
        )
        self.assertAllClose(opt_shared.scaling, torch.full((2,), 2.0, **tkwargs))

        opt_vec_beta_gp = SafeOpt(
            model=model,
            candidate_set=grid,
            safety_thresholds=[0.1],
            initial_safe_indices=[4],
            beta=torch.tensor([2.0, 1.5], **tkwargs),
            lipschitz=None,
            scaling=[1.5],
        )
        self.assertTrue(opt_vec_beta_gp.compute_sets().safe_mask.any())

        # TorchPosterior wrapping standard torch.distributions.MultivariateNormal
        torch_mvn = torch.distributions.MultivariateNormal(
            loc=torch.zeros(10, **tkwargs),
            covariance_matrix=torch.eye(10, **tkwargs),
        )
        covs_torch = _extract_all_output_covariances(
            TorchPosterior(distribution=torch_mvn), num_candidates=10
        )
        self.assertEqual(covs_torch.shape, torch.Size([1, 10, 10]))

    def test_validation_errors(self) -> None:
        r"""Test input validation errors across SafeOpt arguments."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        grid = torch.linspace(0.0, 1.0, 6, **tkwargs).unsqueeze(-1)
        model_1 = _build_fixed_gp(grid[[2]], torch.tensor([[1.0]], **tkwargs))
        model_2 = _build_fixed_gp(grid[[2]], torch.tensor([[1.0, 1.0]], **tkwargs))
        model_3 = _build_fixed_gp(grid[[2]], torch.tensor([[1.0, 1.0, 1.0]], **tkwargs))

        base_kwargs = {
            "model": model_1,
            "candidate_set": grid,
            "safety_thresholds": 0.0,
            "initial_safe_indices": [2],
        }

        invalid_cases = [
            {"candidate_set": torch.empty(0, 1, **tkwargs)},
            {"candidate_set": torch.ones(6, **tkwargs)},
            {"candidate_set": torch.empty(6, 0, **tkwargs)},
            {"step": 0},
            {"candidate_match_atol": -1e-3},
            {"initial_safe_indices": None, "safe_seed": None},
            {"initial_safe_indices": [2], "safe_seed": grid[[2]]},
            {"initial_safe_indices": []},
            {"initial_safe_indices": [10]},
            {"initial_safe_indices": [-1]},
            {
                "initial_safe_indices": torch.zeros(
                    6, dtype=torch.bool, device=self.device
                )
            },
            {
                "initial_safe_indices": torch.ones(
                    4, dtype=torch.bool, device=self.device
                )
            },
            {"initial_safe_indices": None, "safe_seed": torch.empty(0, 1, **tkwargs)},
            {"initial_safe_indices": None, "safe_seed": torch.ones(2, 3, **tkwargs)},
            {"safety_thresholds": [0.0, 0.1]},
            {"safety_thresholds": torch.zeros(1, 1, **tkwargs)},
            {"model": model_3, "safety_thresholds": 0.0},
            {"model": model_3, "safety_thresholds": [0.0]},
            {"beta": 0.0},
            {"beta": -1.0},
            {"beta": lambda t: -0.5},
            {"beta": [1.0, 2.0]},
            {"lipschitz": 0.0},
            {"lipschitz": -2.0},
            {"lipschitz": [1.0, 2.0, 3.0]},
            {"lipschitz": torch.ones(1, 1, **tkwargs)},
            {"scaling": 0.0},
            {"scaling": -1.0},
            {"scaling": "invalid"},
            {"scaling": [1.0, 2.0]},
            {"scaling": torch.ones(1, 1, **tkwargs)},
            {"max_step": 0.0},
            {"max_step": -0.5},
            {"x_current": torch.ones(2, 1, **tkwargs)},
            {"prev_safe_mask": torch.zeros(6, dtype=torch.bool, device=self.device)},
            {"prev_safe_mask": torch.ones(3, dtype=torch.bool, device=self.device)},
            {"prev_lower": torch.zeros(3, 1, **tkwargs)},
            {"prev_upper": torch.zeros(3, 1, **tkwargs)},
        ]

        for override in invalid_cases:
            with self.subTest(override=override):
                kwargs = dict(base_kwargs)
                kwargs.update(override)
                with self.assertRaises(ValueError):
                    SafeOpt(**kwargs)

        # max_step set without x_current raises on next_point()
        opt_no_curr = SafeOpt(**base_kwargs, max_step=0.2)
        with self.assertRaisesRegex(ValueError, "x_current"):
            opt_no_curr.next_point()

        with self.assertRaisesRegex(ValueError, "max_step"):
            opt_no_curr.next_point(x_current=grid[2], max_step=0.0)

        # update() with wrong num_outputs raises ValueError
        with self.assertRaisesRegex(ValueError, "outputs"):
            opt_no_curr.update(model_2)

        # MockPosterior without covariance_matrix raises NotImplementedError for GP-only
        mock_model = MockModel(
            MockPosterior(
                mean=torch.zeros(6, 1, **tkwargs),
                variance=torch.ones(6, 1, **tkwargs),
            )
        )
        opt_mock = SafeOpt(
            model=mock_model,
            candidate_set=grid,
            safety_thresholds=0.5,
            initial_safe_indices=[2],
            lipschitz=None,
        )
        with self.assertRaises(NotImplementedError):
            opt_mock.compute_sets()
