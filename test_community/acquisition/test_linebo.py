#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from unittest.mock import patch

import torch
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions import BotorchWarning, UnsupportedError
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from botorch_community.acquisition.linebo import (
    AscentLineBO,
    compute_line_bounds,
    CoordinateLineBO,
    LineAcquisitionFunction,
    LineBO,
    optimize_linebo_step,
    RandomLineBO,
    sample_ascent_direction,
    sample_coordinate_direction,
    sample_random_direction,
    select_linebo_pivot,
)
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from torch import Tensor


def _build_quadratic_gp(
    center: Tensor,
    bounds: Tensor,
    n_train: int = 24,
    use_normalize: bool = False,
) -> tuple[SingleTaskGP, Tensor, Tensor]:
    r"""Build a smooth low-noise GP surrogate fit to ``f(x) = -||x - center||^2``."""
    d = center.shape[0]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    u = torch.rand(n_train, d, generator=gen, dtype=center.dtype).to(
        device=center.device
    )
    train_X = bounds[0] + (bounds[1] - bounds[0]) * u
    train_Y = -((train_X - center) ** 2).sum(dim=-1, keepdim=True)
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
    covar_module.base_kernel.lengthscale = 0.8
    covar_module.outputscale = 1.0
    likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.1, 0.05))
    likelihood.noise = 1e-4
    input_transform = Normalize(d=d, bounds=bounds) if use_normalize else None
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        likelihood=likelihood,
        covar_module=covar_module,
        input_transform=input_transform,
    ).eval()
    return model, train_X, train_Y


class TestComputeLineBounds(BotorchTestCase):
    def test_interior_corners_axes_and_max_step(self) -> None:
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                self._test_interior_corners_axes_and_max_step(dtype=dtype)

    def _test_interior_corners_axes_and_max_step(self, dtype: torch.dtype) -> None:
        tkwargs = {"device": self.device, "dtype": dtype}
        atol = 1e-5 if dtype == torch.float else 1e-10

        # 1. Interior point with oblique direction
        bounds = torch.tensor([[-1.0, 0.0, 0.0], [2.0, 1.0, 2.0]], **tkwargs)
        pivot = torch.tensor([0.5, 0.25, 1.0], **tkwargs)
        raw_dir = torch.tensor([1.0, -0.5, 1.0], **tkwargs)
        direction = raw_dir / torch.linalg.vector_norm(raw_dir)
        line_bounds = compute_line_bounds(
            pivot=pivot, direction=direction, bounds=bounds
        )
        self.assertEqual(line_bounds.shape, torch.Size([2, 1]))

        # Unscaled limits: coord 0 -> [-1.5, 1.5], coord 1 -> [-1.5, 0.5],
        # coord 2 -> [-1.0, 1.0] => unscaled interval [-1.0, 0.5].
        norm = torch.linalg.vector_norm(raw_dir)
        expected = torch.tensor([[-1.0 * norm], [0.5 * norm]], **tkwargs)
        self.assertAllClose(line_bounds, expected, atol=atol)

        x_min = pivot + line_bounds[0, 0] * direction
        x_max = pivot + line_bounds[1, 0] * direction
        self.assertTrue(torch.all(x_min >= bounds[0] - 1e-6))
        self.assertTrue(torch.all(x_min <= bounds[1] + 1e-6))
        self.assertTrue(torch.all(x_max >= bounds[0] - 1e-6))
        self.assertTrue(torch.all(x_max <= bounds[1] + 1e-6))
        # Stepping beyond [alpha_min, alpha_max] exits bounds
        x_below = pivot + (line_bounds[0, 0] - 1e-2) * direction
        x_above = pivot + (line_bounds[1, 0] + 1e-2) * direction
        self.assertTrue(
            torch.any(x_below < bounds[0]) or torch.any(x_below > bounds[1])
        )
        self.assertTrue(
            torch.any(x_above < bounds[0]) or torch.any(x_above > bounds[1])
        )

        # 2. Boundary corners (diagonal into interior vs tangent touching only corner)
        unit_box = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
        corner = torch.tensor([0.0, 1.0], **tkwargs)
        inward = torch.tensor([1.0, -1.0], **tkwargs) / math.sqrt(2.0)
        lb_in = compute_line_bounds(pivot=corner, direction=inward, bounds=unit_box)
        self.assertAllClose(
            lb_in,
            torch.tensor([[0.0], [math.sqrt(2.0)]], **tkwargs),
            atol=atol,
        )

        # At corner (0, 1), direction (1, 1)/sqrt(2) exits y <= 1 for alpha > 0
        # and exits x >= 0 for alpha < 0, leaving [0, 0].
        tangent = torch.tensor([1.0, 1.0], **tkwargs) / math.sqrt(2.0)
        lb_tan = compute_line_bounds(pivot=corner, direction=tangent, bounds=unit_box)
        self.assertAllClose(
            lb_tan,
            torch.tensor([[0.0], [0.0]], **tkwargs),
            atol=atol,
        )

        # 3. Axis-aligned directions with exact zeros and near-zeros
        axis_dir = torch.tensor([0.0, 1.0, 0.0], **tkwargs)
        lb_axis = compute_line_bounds(pivot=pivot, direction=axis_dir, bounds=bounds)
        self.assertAllClose(
            lb_axis,
            torch.tensor([[-0.25], [0.75]], **tkwargs),
            atol=atol,
        )

        near_zero_dir = torch.tensor([1e-15, -1.0, -1e-16], **tkwargs)
        lb_near_zero = compute_line_bounds(
            pivot=pivot, direction=near_zero_dir, bounds=bounds
        )
        self.assertAllClose(
            lb_near_zero,
            torch.tensor([[-0.75], [0.25]], **tkwargs),
            atol=atol,
        )

        # 4. max_step symmetric and asymmetric capping
        lb_capped_sym = compute_line_bounds(
            pivot=pivot, direction=axis_dir, bounds=bounds, max_step=0.15
        )
        self.assertAllClose(
            lb_capped_sym,
            torch.tensor([[-0.15], [0.15]], **tkwargs),
            atol=atol,
        )
        lb_capped_asym = compute_line_bounds(
            pivot=pivot, direction=axis_dir, bounds=bounds, max_step=0.5
        )
        self.assertAllClose(
            lb_capped_asym,
            torch.tensor([[-0.25], [0.5]], **tkwargs),
            atol=atol,
        )

        # 5. Sub-tolerance numerical drift (<= 1e-6) accepted; (1, d) accepted
        slightly_outside = torch.tensor([[0.0 - 5e-7, 1.0 + 5e-7]], **tkwargs)
        lb_tol = compute_line_bounds(
            pivot=slightly_outside, direction=inward.unsqueeze(0), bounds=unit_box
        )
        self.assertAllClose(
            lb_tol,
            torch.tensor([[0.0], [math.sqrt(2.0)]], **tkwargs),
            atol=1e-5,
        )

    def test_validation_errors(self) -> None:
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=self.device)
        pivot = torch.tensor([0.5, 0.5], device=self.device)
        direction = torch.tensor([1.0, 0.0], device=self.device)

        # Out-of-bounds pivot
        with self.assertRaisesRegex(ValueError, "must lie within `bounds`"):
            compute_line_bounds(
                pivot=torch.tensor([-0.01, 0.5], device=self.device),
                direction=direction,
                bounds=bounds,
            )
        with self.assertRaisesRegex(ValueError, "must lie within `bounds`"):
            compute_line_bounds(
                pivot=torch.tensor([0.5, 1.01], device=self.device),
                direction=direction,
                bounds=bounds,
            )

        # Invalid bounds shape / ordering
        with self.assertRaisesRegex(ValueError, "`bounds` must have shape"):
            compute_line_bounds(pivot=pivot, direction=direction, bounds=bounds[0])
        with self.assertRaisesRegex(ValueError, "Lower bounds must not exceed"):
            compute_line_bounds(
                pivot=pivot,
                direction=direction,
                bounds=torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=self.device),
            )

        # Dimension mismatch, zero direction, invalid max_step
        with self.assertRaisesRegex(ValueError, "Dimension mismatch"):
            compute_line_bounds(
                pivot=torch.tensor([0.5, 0.5, 0.5], device=self.device),
                direction=direction,
                bounds=bounds,
            )
        with self.assertRaisesRegex(ValueError, "non-zero norm"):
            compute_line_bounds(
                pivot=pivot,
                direction=torch.zeros(2, device=self.device),
                bounds=bounds,
            )
        with self.assertRaisesRegex(ValueError, "`max_step` must be positive"):
            compute_line_bounds(
                pivot=pivot, direction=direction, bounds=bounds, max_step=0.0
            )


class TestLineAcquisitionFunction(BotorchTestCase):
    def test_single_objective_values_shapes_gradients_and_log(self) -> None:
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                self._test_single_objective(dtype=dtype)

    def _test_single_objective(self, dtype: torch.dtype) -> None:
        tkwargs = {"device": self.device, "dtype": dtype}
        d = 4
        bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
        center = torch.tensor([0.7, 0.3, 0.8, 0.2], **tkwargs)
        model, _, train_Y = _build_quadratic_gp(
            center=center, bounds=bounds, n_train=16
        )

        log_ei = LogExpectedImprovement(model=model, best_f=train_Y.max())
        ucb = UpperConfidenceBound(model=model, beta=0.5)

        pivot = torch.tensor([0.4, 0.5, 0.4, 0.5], **tkwargs)
        raw_dir = torch.tensor([2.0, -1.0, 1.5, -0.5], **tkwargs)
        unit_dir = raw_dir / torch.linalg.vector_norm(raw_dir)

        line_log_ei = LineAcquisitionFunction(
            acq_function=log_ei, pivot=pivot, direction=raw_dir
        )
        line_ucb = LineAcquisitionFunction(
            acq_function=ucb, pivot=pivot, direction=raw_dir
        )

        # Verify _log propagation, model forwarding, and direction normalization
        self.assertTrue(line_log_ei._log)
        self.assertFalse(line_ucb._log)
        self.assertIs(line_log_ei.model, model)
        self.assertAllClose(line_log_ei.direction, unit_dir)

        # Verify 0D, (q, 1), and t-batched (b1, b2, q, 1) projections & evaluations
        proj_scalar = line_log_ei.project_to_full_space(torch.tensor(0.1, **tkwargs))
        self.assertEqual(proj_scalar.shape, torch.Size([d]))
        self.assertAllClose(proj_scalar, pivot + 0.1 * unit_dir)

        for alpha_shape in [(1, 1), (5, 1, 1), (2, 3, 1, 1)]:
            alpha = torch.linspace(-0.15, 0.15, steps=math.prod(alpha_shape), **tkwargs)
            alpha = alpha.reshape(alpha_shape)
            X_full = pivot + alpha * unit_dir
            self.assertAllClose(line_log_ei(alpha), log_ei(X_full))
            self.assertAllClose(line_ucb(alpha), ucb(X_full))

        # Verify exact autograd chain rule: d acq / d alpha == grad_X(acq) . unit_dir
        alpha_grad = torch.tensor([[0.08]], **tkwargs).requires_grad_(True)
        val_1d = line_log_ei(alpha_grad)
        val_1d.backward()

        X_grad = (pivot + alpha_grad.detach() * unit_dir).detach().requires_grad_(True)
        val_full = log_ei(X_grad)
        val_full.backward()
        expected_grad = (X_grad.grad * unit_dir).sum(dim=-1, keepdim=True)
        self.assertAllClose(
            alpha_grad.grad,
            expected_grad,
            atol=1e-5 if dtype == torch.float else 1e-10,
        )

        # Verify 1D optimize_acqf integration
        line_bounds = compute_line_bounds(
            pivot=pivot, direction=line_log_ei.direction, bounds=bounds
        )
        cand_alpha, acq_val = optimize_acqf(
            acq_function=line_log_ei,
            bounds=line_bounds,
            q=1,
            num_restarts=4,
            raw_samples=32,
        )
        self.assertEqual(cand_alpha.shape, torch.Size([1, 1]))
        self.assertTrue(torch.isfinite(acq_val))
        cand_X = line_log_ei.project_to_full_space(cand_alpha)
        self.assertEqual(cand_X.shape, torch.Size([1, d]))
        self.assertTrue(torch.all(cand_X >= bounds[0] - 1e-6))
        self.assertTrue(torch.all(cand_X <= bounds[1] + 1e-6))

        # Analytic set_X_pending raises UnsupportedError
        with self.assertRaises(UnsupportedError):
            line_log_ei.set_X_pending(torch.tensor([[0.05]], **tkwargs))

        # Validation errors on LineAcquisitionFunction
        with self.assertRaisesRegex(ValueError, "must match"):
            LineAcquisitionFunction(
                acq_function=log_ei,
                pivot=pivot,
                direction=torch.ones(d + 1, **tkwargs),
            )
        with self.assertRaisesRegex(ValueError, "non-zero norm"):
            LineAcquisitionFunction(
                acq_function=log_ei,
                pivot=pivot,
                direction=torch.zeros(d, **tkwargs),
            )
        with self.assertRaisesRegex(ValueError, "trailing dimension 1"):
            line_log_ei.project_to_full_space(torch.zeros(2, 2, **tkwargs))
        with self.assertRaisesRegex(ValueError, "batch_shape x q x 1"):
            line_log_ei(torch.zeros(2, 2, **tkwargs))

    def test_multi_objective_q_batch_and_pending_points(self) -> None:
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                self._test_multi_objective(dtype=dtype)

    def _test_multi_objective(self, dtype: torch.dtype) -> None:
        tkwargs = {"device": self.device, "dtype": dtype}
        d = 3
        bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
        c1 = torch.tensor([0.8, 0.2, 0.5], **tkwargs)
        c2 = torch.tensor([0.2, 0.8, 0.5], **tkwargs)
        m1, train_X, _ = _build_quadratic_gp(center=c1, bounds=bounds, n_train=12)
        m2, _, _ = _build_quadratic_gp(center=c2, bounds=bounds, n_train=12)
        mo_model = ModelListGP(m1, m2)

        ref_point = torch.tensor([-2.0, -2.0], **tkwargs)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=7)
        nehvi = qLogNoisyExpectedHypervolumeImprovement(
            model=mo_model,
            ref_point=ref_point,
            X_baseline=train_X,
            sampler=sampler,
            prune_baseline=True,
        )

        pivot = torch.tensor([0.5, 0.5, 0.5], **tkwargs)
        direction = torch.tensor([1.0, -1.0, 0.0], **tkwargs) / math.sqrt(2.0)

        # Pre-populate a full-space pending point to verify preservation
        base_pending = torch.tensor([[0.3, 0.7, 0.5]], **tkwargs)
        nehvi.set_X_pending(base_pending)

        line_nehvi = LineAcquisitionFunction(
            acq_function=nehvi, pivot=pivot, direction=direction
        )
        self.assertTrue(line_nehvi._log)
        self.assertIsNone(line_nehvi.X_pending)

        # t-batch x q-batch evaluation (b=3, q=2, 1)
        alpha_q = torch.tensor(
            [[[-0.1], [0.1]], [[-0.2], [0.0]], [[0.05], [0.2]]], **tkwargs
        )
        X_q_full = line_nehvi.project_to_full_space(alpha_q)
        self.assertEqual(X_q_full.shape, torch.Size([3, 2, d]))
        self.assertAllClose(line_nehvi(alpha_q), nehvi(X_q_full))

        # Test set_X_pending warning with requires_grad and projection to full space
        alpha_pend_grad = torch.tensor([[0.15]], **tkwargs).requires_grad_(True)
        with self.assertWarns(BotorchWarning):
            line_nehvi.set_X_pending(alpha_pend_grad)
        self.assertAllClose(line_nehvi.X_pending, alpha_pend_grad.detach())
        expected_full_pend = torch.cat(
            [base_pending, pivot.unsqueeze(0) + 0.15 * direction.unsqueeze(0)], dim=-2
        )
        self.assertAllClose(nehvi.X_pending, expected_full_pend)

        # Clearing line pending restores pre-existing base_pending
        line_nehvi.X_pending = None
        self.assertIsNone(line_nehvi.X_pending)
        self.assertAllClose(nehvi.X_pending, base_pending)

        with self.assertRaisesRegex(ValueError, "must have shape"):
            line_nehvi.set_X_pending(torch.zeros(2, 2, **tkwargs))

        # Sequential q=2 optimization on 1D line with qLogNEHVI
        line_bounds = compute_line_bounds(
            pivot=pivot, direction=direction, bounds=bounds
        )
        cand_alpha_seq, acq_vals_seq = optimize_acqf(
            acq_function=line_nehvi,
            bounds=line_bounds,
            q=2,
            num_restarts=3,
            raw_samples=16,
            sequential=True,
        )
        self.assertEqual(cand_alpha_seq.shape, torch.Size([2, 1]))
        self.assertEqual(acq_vals_seq.shape, torch.Size([2]))
        # Base pending preserved after sequential optimization cleanup
        self.assertIsNone(line_nehvi.X_pending)
        self.assertAllClose(nehvi.X_pending, base_pending)


class TestLineSelectionStrategies(BotorchTestCase):
    def test_coordinate_and_random_strategies(self) -> None:
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                self._test_coordinate_and_random(dtype=dtype)

    def _test_coordinate_and_random(self, dtype: torch.dtype) -> None:
        tkwargs = {"device": self.device, "dtype": dtype}
        d = 4

        # Coordinate deterministic cycling
        for step_idx in range(d + 2):
            v = sample_coordinate_direction(d=d, coordinate_index=step_idx, **tkwargs)
            expected = torch.zeros(d, **tkwargs)
            expected[step_idx % d] = 1.0
            self.assertAllClose(v, expected)

        # Coordinate uniform random sampling with seed / generator
        v_seed_a = sample_coordinate_direction(d=d, seed=99, **tkwargs)
        v_seed_b = sample_coordinate_direction(d=d, seed=99, **tkwargs)
        self.assertAllClose(v_seed_a, v_seed_b)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(v_seed_a).item()), 1.0)
        self.assertEqual(int((v_seed_a != 0).sum().item()), 1)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(123)
        seen_axes = {
            int(
                torch.argmax(
                    sample_coordinate_direction(d=d, generator=gen, **tkwargs)
                ).item()
            )
            for _ in range(32)
        }
        self.assertEqual(seen_axes, {0, 1, 2, 3})

        # Random spherical direction sampling
        r1 = sample_random_direction(d=d, seed=42, **tkwargs)
        r2 = sample_random_direction(d=d, seed=42, **tkwargs)
        self.assertAllClose(r1, r2)
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(r1).item()), 1.0, places=5
        )

        gen_r = torch.Generator(device="cpu")
        gen_r.manual_seed(7)
        draws = torch.stack(
            [sample_random_direction(d=d, generator=gen_r, **tkwargs) for _ in range(8)]
        )
        norms = torch.linalg.vector_norm(draws, dim=-1)
        self.assertAllClose(norms, torch.ones(8, **tkwargs), atol=1e-5)
        # Successive draws from the same generator are distinct
        self.assertFalse(torch.allclose(draws[0], draws[1]))

        # Validation errors
        with self.assertRaisesRegex(ValueError, "`d` must be >= 1"):
            sample_coordinate_direction(d=0)
        with self.assertRaisesRegex(ValueError, "`d` must be >= 1"):
            sample_random_direction(d=0)
        with self.assertRaisesRegex(ValueError, "at most one"):
            sample_random_direction(d=3, generator=gen_r, seed=1)
        with self.assertRaisesRegex(ValueError, "at most one"):
            sample_coordinate_direction(d=3, generator=gen_r, seed=1)

    def test_ascent_strategy_mean_matheron_and_fallbacks(self) -> None:
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                self._test_ascent_strategy(dtype=dtype)

    def _test_ascent_strategy(self, dtype: torch.dtype) -> None:
        tkwargs = {"device": self.device, "dtype": dtype}
        d = 3
        bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
        center = torch.tensor([0.8, 0.75, 0.7], **tkwargs)
        model, _, _ = _build_quadratic_gp(center=center, bounds=bounds, n_train=30)

        pivot = torch.tensor([0.2, 0.25, 0.3], **tkwargs)
        true_ascent = center - pivot
        true_ascent = true_ascent / torch.linalg.vector_norm(true_ascent)

        # 1. Deterministic posterior mean gradient (sample_gradient=False)
        dir_mean = sample_ascent_direction(
            model=model, pivot=pivot, sample_gradient=False
        )
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(dir_mean).item()), 1.0, places=5
        )
        cos_sim = float(torch.dot(dir_mean, true_ascent).item())
        self.assertGreater(cos_sim, 0.95)

        # Minimization (maximize=False) negates direction
        dir_descent = sample_ascent_direction(
            model=model, pivot=pivot, sample_gradient=False, maximize=False
        )
        self.assertAllClose(dir_descent, -dir_mean, atol=1e-5)

        # 2. Thompson-sampled gradient via draw_matheron_paths (sample_gradient=True)
        with patch(
            "botorch_community.acquisition.linebo.draw_matheron_paths",
            wraps=draw_matheron_paths,
        ) as spy_matheron:
            dir_ts_1 = sample_ascent_direction(
                model=model, pivot=pivot, sample_gradient=True, seed=101
            )
            self.assertEqual(spy_matheron.call_count, 1)
        dir_ts_2 = sample_ascent_direction(
            model=model, pivot=pivot, sample_gradient=True, seed=101
        )
        self.assertAllClose(dir_ts_1, dir_ts_2, atol=1e-5)
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(dir_ts_1).item()), 1.0, places=5
        )
        # On a dense low-noise quadratic GP, posterior path gradient aligns uphill
        self.assertGreater(float(torch.dot(dir_ts_1, true_ascent).item()), 0.8)

        # 3. ModelListGP fallback (finite-difference joint posterior sample) +
        # multi-output scalarization weights / PosteriorTransform
        opposite_center = torch.tensor([0.1, 0.15, 0.2], **tkwargs)
        model_2, _, _ = _build_quadratic_gp(
            center=opposite_center, bounds=bounds, n_train=30
        )
        model_list = ModelListGP(model, model_2)

        # Mean gradient with scalarization_weights selecting output 0 vs output 1
        dir_w0 = sample_ascent_direction(
            model=model_list,
            pivot=pivot,
            sample_gradient=False,
            scalarization_weights=torch.tensor([1.0, 0.0], **tkwargs),
        )
        self.assertAllClose(dir_w0, dir_mean, atol=1e-5)

        transform_w1 = ScalarizedPosteriorTransform(
            weights=torch.tensor([0.0, 1.0], **tkwargs)
        )
        dir_w1 = sample_ascent_direction(
            model=model_list,
            pivot=pivot,
            sample_gradient=False,
            posterior_transform=transform_w1,
        )
        true_ascent_2 = opposite_center - pivot
        true_ascent_2 = true_ascent_2 / torch.linalg.vector_norm(true_ascent_2)
        self.assertGreater(float(torch.dot(dir_w1, true_ascent_2).item()), 0.9)

        # Thompson-sampled gradient on ModelListGP exercises FD joint posterior fallback
        dir_ml_ts_a = sample_ascent_direction(
            model=model_list,
            pivot=pivot,
            sample_gradient=True,
            scalarization_weights=torch.tensor([1.0, 0.0], **tkwargs),
            seed=55,
        )
        dir_ml_ts_b = sample_ascent_direction(
            model=model_list,
            pivot=pivot,
            sample_gradient=True,
            scalarization_weights=torch.tensor([1.0, 0.0], **tkwargs),
            seed=55,
        )
        self.assertAllClose(dir_ml_ts_a, dir_ml_ts_b, atol=1e-5)
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(dir_ml_ts_a).item()), 1.0, places=5
        )
        self.assertGreater(float(torch.dot(dir_ml_ts_a, true_ascent).item()), 0.75)

        # 4. Zero-gradient fallback (||g||_2 < 1e-12 -> random unit direction)
        flat_mean = torch.zeros(1, 1, **tkwargs)
        flat_samples = torch.zeros(1, 2 * d, 1, **tkwargs)
        mock_flat = MockModel(MockPosterior(mean=flat_mean, samples=flat_samples))
        dir_zero_fd = sample_ascent_direction(
            model=mock_flat, pivot=pivot, sample_gradient=True, seed=19
        )
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(dir_zero_fd).item()), 1.0, places=5
        )

        # Validation errors on scalarization_weights
        with self.assertRaisesRegex(ValueError, "at most one"):
            sample_ascent_direction(
                model=model_list,
                pivot=pivot,
                posterior_transform=transform_w1,
                scalarization_weights=torch.tensor([0.5, 0.5], **tkwargs),
            )
        with self.assertRaisesRegex(ValueError, "must match `model.num_outputs`"):
            sample_ascent_direction(
                model=model_list,
                pivot=pivot,
                scalarization_weights=torch.tensor([1.0], **tkwargs),
            )


class TestLineBOLoopAndHelpers(BotorchTestCase):
    def test_pivot_selection_and_multistep_loops(self) -> None:
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                self._test_pivot_and_loops(dtype=dtype)

    def _test_pivot_and_loops(self, dtype: torch.dtype) -> None:
        tkwargs = {"device": self.device, "dtype": dtype}
        collinear_atol = 1e-4 if dtype == torch.float else 1e-8
        d = 5
        bounds = torch.stack(
            [torch.full((d,), -1.0, **tkwargs), torch.full((d,), 2.0, **tkwargs)]
        )
        center = torch.tensor([0.8, -0.2, 1.1, 0.4, 0.0], **tkwargs)
        model, train_X, train_Y = _build_quadratic_gp(
            center=center, bounds=bounds, n_train=15, use_normalize=True
        )

        # Verify pivot selection extracts untransformed inputs even with Normalize()
        pivot_last = select_linebo_pivot(
            model=model, pivot_selection="last_observed", bounds=bounds
        )
        self.assertAllClose(pivot_last, train_X[-1], atol=1e-5)

        pivot_best = select_linebo_pivot(
            model=model, pivot_selection="best_posterior_mean", bounds=bounds
        )
        with torch.no_grad():
            expected_idx = int(torch.argmax(model.posterior(train_X).mean).item())
        self.assertAllClose(pivot_best, train_X[expected_idx], atol=1e-5)

        # Multi-step loop 1: AscentLineBO with best_posterior_mean pivot selection
        X_hist = train_X.clone()
        Y_hist = train_Y.clone()
        ascent_bo = AscentLineBO(
            bounds=bounds,
            pivot_selection="best_posterior_mean",
            sample_gradient=False,
            num_restarts=4,
            raw_samples=64,
        )
        for step in range(3):
            gp = SingleTaskGP(
                X_hist, Y_hist, input_transform=Normalize(d=d, bounds=bounds)
            ).eval()
            acqf = LogExpectedImprovement(model=gp, best_f=Y_hist.max())
            cand_X, acq_val, direction = ascent_bo.step(acqf)
            self.assertEqual(ascent_bo.step_count, step + 1)
            self.assertEqual(cand_X.shape, torch.Size([1, d]))
            self.assertTrue(torch.isfinite(acq_val))
            self.assertAlmostEqual(
                float(torch.linalg.vector_norm(direction).item()), 1.0, places=5
            )
            # Candidate strictly within bounds
            self.assertTrue(torch.all(cand_X >= bounds[0]))
            self.assertTrue(torch.all(cand_X <= bounds[1]))
            # Candidate strictly collinear with (last_pivot, direction)
            delta = cand_X.squeeze(0) - ascent_bo.last_pivot
            proj = torch.dot(delta, direction) * direction
            self.assertLess(
                float(torch.linalg.vector_norm(delta - proj).item()), collinear_atol
            )
            new_Y = -((cand_X - center) ** 2).sum(dim=-1, keepdim=True)
            X_hist = torch.cat([X_hist, cand_X], dim=0)
            Y_hist = torch.cat([Y_hist, new_Y], dim=0)

        # Multi-step loop 2: RandomLineBO & CoordinateLineBO (last_observed + max_step)
        max_step = 0.35
        for bo in (
            CoordinateLineBO(
                bounds=bounds,
                pivot_selection="last_observed",
                max_step=max_step,
                cycle_coordinates=True,
                num_restarts=4,
                raw_samples=32,
            ),
            RandomLineBO(
                bounds=bounds,
                pivot_selection="last_observed",
                max_step=max_step,
                num_restarts=4,
                raw_samples=32,
                seed=2025,
            ),
        ):
            X_run = train_X[:8].clone()
            Y_run = train_Y[:8].clone()
            for step in range(4):
                gp = SingleTaskGP(X_run, Y_run).eval()
                acqf = UpperConfidenceBound(model=gp, beta=0.4)
                expected_pivot = X_run[-1].clone()
                cand_X, _, direction = bo.step(acqf)

                # Connected sweep: pivot equals previous candidate (X_run[-1])
                self.assertAllClose(bo.last_pivot, expected_pivot, atol=1e-5)
                self.assertTrue(torch.all(cand_X >= bounds[0]))
                self.assertTrue(torch.all(cand_X <= bounds[1]))

                delta = cand_X.squeeze(0) - expected_pivot
                proj = torch.dot(delta, direction) * direction
                self.assertLess(
                    float(torch.linalg.vector_norm(delta - proj).item()),
                    collinear_atol,
                )
                self.assertLessEqual(
                    float(torch.linalg.vector_norm(delta).item()),
                    max_step + 1e-4,
                )
                if isinstance(bo, CoordinateLineBO):
                    expected_dir = torch.zeros(d, **tkwargs)
                    expected_dir[step % d] = 1.0
                    self.assertAllClose(direction, expected_dir)

                new_Y = -((cand_X - center) ** 2).sum(dim=-1, keepdim=True)
                X_run = torch.cat([X_run, cand_X], dim=0)
                Y_run = torch.cat([Y_run, new_Y], dim=0)

        # Degenerate zero-width line interval (mixed corner + supporting hyperplane dir)
        acqf_deg = LogExpectedImprovement(model=model, best_f=train_Y.max())
        corner = bounds[1].clone()
        corner[0] = bounds[0, 0]
        tangent_dir = torch.ones(d, **tkwargs) / math.sqrt(d)
        cand_deg, acq_deg, dir_deg = optimize_linebo_step(
            acq_function=acqf_deg,
            bounds=bounds,
            pivot=corner,
            direction=tangent_dir,
        )
        self.assertAllClose(cand_deg.squeeze(0), corner, atol=1e-6)
        self.assertTrue(torch.isfinite(acq_deg))
        self.assertAllClose(dir_deg, tangent_dir, atol=1e-5)

    def test_helper_validation_errors(self) -> None:
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=self.device)
        with self.assertRaisesRegex(ValueError, "Unknown `strategy="):
            LineBO(bounds=bounds, strategy="invalid")
        with self.assertRaisesRegex(ValueError, "Unknown `pivot_selection="):
            LineBO(bounds=bounds, pivot_selection="invalid")
        with self.assertRaisesRegex(ValueError, "`max_step` must be positive"):
            LineBO(bounds=bounds, max_step=-1.0)
        with self.assertRaisesRegex(ValueError, "Unknown `pivot_selection="):
            select_linebo_pivot(
                train_X=torch.rand(3, 2, device=self.device),
                pivot_selection="invalid",
            )
        with self.assertRaisesRegex(ValueError, "`model` is required"):
            select_linebo_pivot(
                train_X=torch.rand(3, 2, device=self.device),
                pivot_selection="best_posterior_mean",
            )
