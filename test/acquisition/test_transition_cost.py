#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
    PosteriorMean,
)
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.acquisition.transition_cost import (
    TransitionCostAcquisitionFunction,
    TransitionCostModel,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import NumericsWarning
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.cost import FixedCostModel
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class _DummyModelNoTrainInputs(GPyTorchModel):
    _num_outputs = 1

    def __init__(self) -> None:  # noqa: D107
        super(GPyTorchModel, self).__init__()

    def subset_output(self, idcs: list[int]) -> Model:  # pragma: no cover
        return self


class _SignedLinearAcquisition(AcquisitionFunction):
    def __init__(self, model: Model, offset: float = -2.0) -> None:  # noqa: D107
        super().__init__(model=model)
        self.offset = offset

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        return X.sum(dim=(-2, -1)) + self.offset


class TestTransitionCostModel(BotorchTestCase):
    def test_sequential_path_costs_and_norms(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            x_prev = torch.tensor([0.2, 0.5, 0.8], **tkwargs)
            fixed_cost = 1.25
            scale = 2.5

            for p in (1.0, 1.5, 2.0, float("inf")):
                cost_model = TransitionCostModel(
                    x_prev=x_prev, fixed_cost=fixed_cost, scale=scale, p=p
                )
                self.assertEqual(cost_model.num_outputs, 1)

                # Test q = 1 (unbatched 2D and batched 3D)
                X_q1_2d = torch.tensor([[0.6, 0.1, 0.8]], **tkwargs)
                expected_q1 = fixed_cost + scale * torch.linalg.vector_norm(
                    X_q1_2d - x_prev, ord=p, dim=-1, keepdim=True
                )
                self.assertAllClose(cost_model(X_q1_2d), expected_q1)

                X_q1 = torch.rand(4, 1, 3, **tkwargs)
                expected_q1_batch = fixed_cost + scale * torch.linalg.vector_norm(
                    X_q1 - x_prev, ord=p, dim=-1, keepdim=True
                )
                out_q1 = cost_model(X_q1)
                self.assertEqual(out_q1.shape, torch.Size([4, 1, 1]))
                self.assertAllClose(out_q1, expected_q1_batch)

                # Test q = 3 sequential path (multi-batch 2 x 4 x 3 x 3)
                X_q3 = torch.rand(2, 4, 3, 3, **tkwargs)
                d0 = torch.linalg.vector_norm(X_q3[..., 0, :] - x_prev, ord=p, dim=-1)
                d1 = torch.linalg.vector_norm(
                    X_q3[..., 1, :] - X_q3[..., 0, :], ord=p, dim=-1
                )
                d2 = torch.linalg.vector_norm(
                    X_q3[..., 2, :] - X_q3[..., 1, :], ord=p, dim=-1
                )
                expected_q3 = (
                    fixed_cost + scale * torch.stack([d0, d1, d2], dim=-1)
                ).unsqueeze(-1)
                out_q3 = cost_model(X_q3)
                self.assertEqual(out_q3.shape, torch.Size([2, 4, 3, 1]))
                self.assertAllClose(out_q3, expected_q3)

            # Test (1, d) x_prev input and empty q = 0 edge case
            cost_model_2d_prev = TransitionCostModel(x_prev=x_prev.unsqueeze(0))
            self.assertEqual(cost_model_2d_prev.x_prev.shape, torch.Size([3]))
            empty_out = cost_model_2d_prev(torch.empty(2, 0, 3, **tkwargs))
            self.assertEqual(empty_out.shape, torch.Size([2, 0, 1]))

    def test_weights_and_mahalanobis_metric(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            x_prev = torch.tensor([0.1, 0.4, 0.7], **tkwargs)
            weights = torch.tensor([2.0, 0.5, 3.0], **tkwargs)
            X = torch.rand(5, 3, 3, **tkwargs)

            # Weighted L_p norm
            for p in (1.0, 2.0, float("inf")):
                w_model = TransitionCostModel(
                    x_prev=x_prev, fixed_cost=0.5, scale=1.5, p=p, weights=weights
                )
                prev_X = torch.cat(
                    [x_prev.view(1, 1, 3).expand(5, 1, 3), X[:, :-1, :]], dim=-2
                )
                expected_dist = torch.linalg.vector_norm(
                    (X - prev_X) * weights, ord=p, dim=-1
                )
                expected = (0.5 + 1.5 * expected_dist).unsqueeze(-1)
                self.assertAllClose(w_model(X), expected)

            # Non-diagonal SPD Mahalanobis metric W
            factor = torch.tensor(
                [[1.2, 0.0, 0.0], [-0.4, 0.9, 0.0], [0.3, 0.5, 1.1]], **tkwargs
            )
            metric = factor @ factor.mT
            m_model = TransitionCostModel(
                x_prev=x_prev, fixed_cost=0.8, scale=2.0, metric=metric
            )
            prev_X = torch.cat(
                [x_prev.view(1, 1, 3).expand(5, 1, 3), X[:, :-1, :]], dim=-2
            )
            diff = X - prev_X
            quad = torch.einsum("...i,ij,...j->...", diff, metric, diff)
            expected_m = (0.8 + 2.0 * torch.sqrt(quad)).unsqueeze(-1)
            self.assertAllClose(m_model(X), expected_m)

    def test_zero_displacement_exact_value_and_finite_zero_gradient(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            x_prev = torch.tensor([0.25, 0.5, 0.75], **tkwargs)
            weights = torch.tensor([1.5, 2.5, 0.5], **tkwargs)
            factor = torch.tensor(
                [[1.0, 0.0, 0.0], [0.4, 1.2, 0.0], [-0.3, 0.2, 0.8]], **tkwargs
            )
            metric = factor @ factor.mT

            configs = [
                {"p": 1.0},
                {"p": 1.5},
                {"p": 2.0},
                {"p": float("inf")},
                {"p": 2.0, "weights": weights},
                {"metric": metric},
            ]
            for kwargs in configs:
                cost_model = TransitionCostModel(
                    x_prev=x_prev, fixed_cost=1.5, scale=3.0, **kwargs
                )
                # q = 1 exactly at x_prev
                X_eq = x_prev.clone().view(1, 1, 3).requires_grad_(True)
                out = cost_model(X_eq)
                self.assertAllClose(out, torch.full_like(out, 1.5), atol=0.0, rtol=0.0)
                out.sum().backward()
                self.assertIsNotNone(X_eq.grad)
                self.assertFalse(torch.isnan(X_eq.grad).any())
                self.assertAllClose(
                    X_eq.grad, torch.zeros_like(X_eq), atol=0.0, rtol=0.0
                )

                # q = 3 with repeated points (step 0 == x_prev and step 2 == step 1)
                X_seq = (
                    torch.stack(
                        [
                            x_prev,
                            torch.tensor([0.6, 0.2, 0.9], **tkwargs),
                            torch.tensor([0.6, 0.2, 0.9], **tkwargs),
                        ],
                        dim=0,
                    )
                    .unsqueeze(0)
                    .clone()
                    .requires_grad_(True)
                )
                out_seq = cost_model(X_seq)
                self.assertAllClose(
                    out_seq[0, 0, 0],
                    torch.tensor(1.5, **tkwargs),
                    atol=0.0,
                    rtol=0.0,
                )
                self.assertAllClose(
                    out_seq[0, 2, 0],
                    torch.tensor(1.5, **tkwargs),
                    atol=0.0,
                    rtol=0.0,
                )
                out_seq.sum().backward()
                self.assertIsNotNone(X_seq.grad)
                self.assertFalse(torch.isnan(X_seq.grad).any())
                self.assertTrue(torch.isfinite(X_seq.grad).all())

    def test_inverse_cost_weighted_utility_compatibility(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            x_prev = torch.zeros(2, **tkwargs)
            cost_model = TransitionCostModel(x_prev=x_prev, fixed_cost=1.0, scale=2.0)
            X = torch.tensor(
                [[[0.3, 0.4], [0.3, 1.4]], [[0.0, 0.0], [0.0, 0.5]]], **tkwargs
            )
            # Batch 0: step0 = 1 + 2*0.5 = 2.0, step1 = 1 + 2*1.0 = 3.0 -> sum = 5.0
            # Batch 1: step0 = 1 + 2*0.0 = 1.0, step1 = 1 + 2*0.5 = 2.0 -> sum = 3.0
            expected_total_cost = torch.tensor([5.0, 3.0], **tkwargs)
            deltas = torch.tensor([[10.0, 6.0], [5.0, 9.0]], **tkwargs)

            icwu = InverseCostWeightedUtility(cost_model=cost_model, use_mean=True)
            weighted = icwu(X, deltas)
            self.assertAllClose(weighted, deltas / expected_total_cost)

            icwu_log = InverseCostWeightedUtility(
                cost_model=cost_model, use_mean=True, log=True
            )
            log_weighted = icwu_log(X, deltas.log())
            self.assertAllClose(log_weighted.exp(), deltas / expected_total_cost)

    def test_validation_errors(self) -> None:
        x_prev = torch.zeros(3, device=self.device, dtype=torch.double)
        with self.assertRaisesRegex(ValueError, "`x_prev` must have shape"):
            TransitionCostModel(x_prev=torch.zeros(2, 3, device=self.device))
        with self.assertRaisesRegex(ValueError, "`x_prev` must have shape"):
            TransitionCostModel(x_prev=torch.empty(0, device=self.device))
        with self.assertRaisesRegex(ValueError, "`x_prev` must contain finite"):
            TransitionCostModel(
                x_prev=torch.tensor([0.0, float("nan")], device=self.device)
            )
        for bad_fixed in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "`fixed_cost` must be finite"):
                TransitionCostModel(x_prev=x_prev, fixed_cost=bad_fixed)
        for bad_scale in (-0.1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "`scale` must be finite"):
                TransitionCostModel(x_prev=x_prev, scale=bad_scale)
        for bad_p in (0.5, 0.0, -1.0, float("-inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "`p` must be >= 1.0"):
                TransitionCostModel(x_prev=x_prev, p=bad_p)

        with self.assertRaisesRegex(ValueError, "Cannot specify both"):
            TransitionCostModel(
                x_prev=x_prev,
                weights=torch.ones(3, device=self.device, dtype=torch.double),
                metric=torch.eye(3, device=self.device, dtype=torch.double),
            )
        with self.assertRaisesRegex(ValueError, "`weights` must be a 1D tensor"):
            TransitionCostModel(
                x_prev=x_prev,
                weights=torch.ones(2, device=self.device, dtype=torch.double),
            )
        with self.assertRaisesRegex(
            ValueError, "`weights` must be finite and strictly positive"
        ):
            TransitionCostModel(
                x_prev=x_prev,
                weights=torch.tensor(
                    [1.0, 0.0, 1.0], device=self.device, dtype=torch.double
                ),
            )
        with self.assertRaisesRegex(ValueError, "`metric` must be a 2D tensor"):
            TransitionCostModel(
                x_prev=x_prev,
                metric=torch.eye(2, device=self.device, dtype=torch.double),
            )
        with self.assertRaisesRegex(ValueError, "`metric` must contain finite"):
            bad_nan_metric = torch.eye(3, device=self.device, dtype=torch.double)
            bad_nan_metric[0, 0] = float("nan")
            TransitionCostModel(x_prev=x_prev, metric=bad_nan_metric)
        with self.assertRaisesRegex(
            ValueError, "`metric` must be symmetric positive-definite"
        ):
            nonsym = torch.eye(3, device=self.device, dtype=torch.double)
            nonsym[0, 1] = 0.5
            TransitionCostModel(x_prev=x_prev, metric=nonsym)
        with self.assertRaisesRegex(
            ValueError, "`metric` must be symmetric positive-definite"
        ):
            not_pd = torch.diag(
                torch.tensor([1.0, 0.0, 1.0], device=self.device, dtype=torch.double)
            )
            TransitionCostModel(x_prev=x_prev, metric=not_pd)

        model = TransitionCostModel(x_prev=x_prev)
        with self.assertRaisesRegex(ValueError, "at least 2 dimensions"):
            model(torch.zeros(3, device=self.device, dtype=torch.double))
        with self.assertRaisesRegex(ValueError, "Feature dimension of `X`"):
            model(torch.zeros(2, 1, 4, device=self.device, dtype=torch.double))


class TestTransitionCostAcquisitionFunction(BotorchTestCase):
    def test_single_objective_log_and_linear_acquisitions(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            torch.manual_seed(0)
            train_X = torch.rand(6, 3, **tkwargs)
            train_Y = (train_X**2).sum(dim=-1, keepdim=True)
            model = SingleTaskGP(train_X, train_Y).eval()
            best_f = train_Y.max()

            log_ei = LogExpectedImprovement(model=model, best_f=best_f)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NumericsWarning)
                ei = ExpectedImprovement(model=model, best_f=best_f)

            tc_log_ei = TransitionCostAcquisitionFunction(
                acq_function=log_ei, fixed_cost=1.0, scale=2.0, cost_exponent=1.5
            )
            tc_ei = TransitionCostAcquisitionFunction(
                acq_function=ei, fixed_cost=1.0, scale=2.0, cost_exponent=1.5
            )
            self.assertTrue(tc_log_ei._log)
            self.assertFalse(tc_ei._log)

            test_X = torch.rand(5, 1, 3, **tkwargs)
            expected_cost = 1.0 + 2.0 * torch.linalg.vector_norm(
                test_X.squeeze(-2) - train_X[-1], ord=2, dim=-1
            )
            self.assertAllClose(
                tc_log_ei(test_X),
                log_ei(test_X) - 1.5 * expected_cost.log(),
            )
            self.assertAllClose(
                tc_ei(test_X),
                ei(test_X) / expected_cost.pow(1.5),
            )
            self.assertAllClose(
                tc_log_ei(test_X).exp(), tc_ei(test_X), atol=1e-5, rtol=1e-4
            )

            # Test q > 1 MC acquisitions (qLogEI and qEI) and 2D input (q x d)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]))
            q_log_ei = qLogExpectedImprovement(
                model=model, best_f=best_f, sampler=sampler
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NumericsWarning)
                q_ei = qExpectedImprovement(model=model, best_f=best_f, sampler=sampler)
            tc_q_log_ei = TransitionCostAcquisitionFunction(
                acq_function=q_log_ei, fixed_cost=0.5, scale=1.5
            )
            tc_q_ei = TransitionCostAcquisitionFunction(
                acq_function=q_ei, fixed_cost=0.5, scale=1.5
            )
            X_q2 = torch.rand(4, 2, 3, **tkwargs)
            c_q2 = tc_q_log_ei.cost_model(X_q2).sum(dim=(-2, -1))
            self.assertAllClose(tc_q_log_ei(X_q2), q_log_ei(X_q2) - c_q2.log())
            self.assertAllClose(tc_q_ei(X_q2), q_ei(X_q2) / c_q2)
            # 2D input (q x d) returns shape (1,)
            out_2d = tc_q_log_ei(X_q2[0])
            self.assertEqual(out_2d.shape, torch.Size([1]))

            # Gradient at X == x_prev inside TransitionCostAcquisitionFunction
            X_at_prev = train_X[-1:].unsqueeze(0).clone().requires_grad_(True)
            val_at_prev = tc_log_ei(X_at_prev)
            val_at_prev.backward()
            self.assertIsNotNone(X_at_prev.grad)
            self.assertFalse(torch.isnan(X_at_prev.grad).any())
            self.assertTrue(torch.isfinite(X_at_prev.grad).all())

    def test_negative_linear_acquisition_sign_preservation(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X = torch.zeros(4, 2, **tkwargs)
            train_Y = -5.0 * torch.ones(4, 1, **tkwargs)
            model = SingleTaskGP(train_X, train_Y).eval()

            # PosteriorMean is strictly negative (~ -5.0)
            pm = PosteriorMean(model=model)
            tc_pm = TransitionCostAcquisitionFunction(
                acq_function=pm,
                x_prev=torch.zeros(2, **tkwargs),
                fixed_cost=1.0,
                scale=3.0,
                cost_exponent=2.0,
            )
            X_near = torch.zeros(1, 1, 2, **tkwargs)
            X_far = torch.ones(1, 1, 2, **tkwargs)
            val_near = tc_pm(X_near)
            val_far = tc_pm(X_far)
            # Far candidate has higher cost and negative utility -> strictly worse
            self.assertTrue(torch.all(val_far < val_near))

            # Mixed sign batch: >= 0 divided by c^gamma, < 0 multiplied by c^gamma
            signed_acqf = _SignedLinearAcquisition(model=model, offset=-1.0)
            tc_signed = TransitionCostAcquisitionFunction(
                acq_function=signed_acqf,
                x_prev=torch.zeros(2, **tkwargs),
                fixed_cost=1.0,
                scale=2.0,
                cost_exponent=1.5,
            )
            X_mixed = torch.tensor(
                [[[0.1, 0.2]], [[0.8, 0.9]]], **tkwargs
            )  # sums: 0.3 (-0.7 < 0) and 1.7 (+0.7 > 0)
            raw = signed_acqf(X_mixed)
            c = tc_signed.cost_model(X_mixed).sum(dim=(-2, -1)).pow(1.5)
            expected = torch.stack([raw[0] * c[0], raw[1] / c[1]])
            self.assertAllClose(tc_signed(X_mixed), expected)

    def test_multi_objective_qlognehvi_qlogehvi_qlognparego_and_modellist(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            torch.manual_seed(7)
            bounds = torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 10.0]], **tkwargs)
            train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(6, 3, **tkwargs)
            train_Y1 = train_X[:, :1].sin()
            train_Y2 = train_X[:, 1:2].cos()
            train_Y = torch.cat([train_Y1, train_Y2], dim=-1)
            ref_point = torch.tensor([-1.5, -1.5], **tkwargs)

            # 1. Multi-output SingleTaskGP with Normalize
            mo_model = SingleTaskGP(
                train_X, train_Y, input_transform=Normalize(3, bounds=bounds)
            ).eval()
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]))
            qlognehvi = qLogNoisyExpectedHypervolumeImprovement(
                model=mo_model,
                ref_point=ref_point,
                X_baseline=train_X,
                sampler=sampler,
                prune_baseline=True,
            )
            tc_qlognehvi = TransitionCostAcquisitionFunction(
                acq_function=qlognehvi, fixed_cost=1.0, scale=0.8, cost_exponent=1.2
            )
            self.assertTrue(tc_qlognehvi._log)
            self.assertAllClose(tc_qlognehvi.cost_model.x_prev, train_X[-1])

            X_test = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
                3, 2, 3, **tkwargs
            )
            expected_c = tc_qlognehvi.cost_model(X_test).sum(dim=(-2, -1))
            self.assertAllClose(
                tc_qlognehvi(X_test),
                qlognehvi(X_test) - 1.2 * expected_c.log(),
            )

            # Also verify qLogEHVI and qLogNParEGO
            with torch.no_grad():
                pred = mo_model.posterior(train_X).mean
            partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pred)
            qlogehvi = qLogExpectedHypervolumeImprovement(
                model=mo_model,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            tc_qlogehvi = TransitionCostAcquisitionFunction(
                acq_function=qlogehvi, fixed_cost=1.0, scale=0.5
            )
            self.assertAllClose(
                tc_qlogehvi(X_test),
                qlogehvi(X_test)
                - tc_qlogehvi.cost_model(X_test).sum(dim=(-2, -1)).log(),
            )

            qlognparego = qLogNParEGO(
                model=mo_model, X_baseline=train_X, sampler=sampler
            )
            tc_parego = TransitionCostAcquisitionFunction(
                acq_function=qlognparego, fixed_cost=1.0, scale=0.5
            )
            self.assertTrue(tc_parego._log)
            self.assertAllClose(
                tc_parego(X_test),
                qlognparego(X_test)
                - tc_parego.cost_model(X_test).sum(dim=(-2, -1)).log(),
            )

            # 2. ModelListGP with two distinct Normalize objects (InputTransform.equals)
            gp1 = SingleTaskGP(
                train_X, train_Y1, input_transform=Normalize(3, bounds=bounds.clone())
            )
            gp2 = SingleTaskGP(
                train_X, train_Y2, input_transform=Normalize(3, bounds=bounds.clone())
            )
            model_list = ModelListGP(gp1, gp2)
            qlognehvi_ml = qLogNoisyExpectedHypervolumeImprovement(
                model=model_list,
                ref_point=ref_point,
                X_baseline=train_X,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([32])),
            )
            for tf_weight in (False, True):
                tc_ml = TransitionCostAcquisitionFunction(
                    acq_function=qlognehvi_ml,
                    fixed_cost=1.0,
                    scale=1.5,
                    transformed_weighting=tf_weight,
                )
                val = tc_ml(X_test)
                self.assertEqual(val.shape, torch.Size([3]))
                self.assertTrue(torch.isfinite(val).all())

    def test_transformed_weighting_real_vs_normalized_space(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.tensor([[0.0, 0.0], [10.0, 100.0]], **tkwargs)
            train_X = torch.tensor([[1.0, 20.0], [4.0, 50.0], [2.0, 30.0]], **tkwargs)
            train_Y = train_X[:, :1] * 0.1 + train_X[:, 1:2] * 0.01
            norm_tf = Normalize(2, bounds=bounds)
            # Deliberately do NOT call .eval() beforehand to test pre-eval handling
            model = SingleTaskGP(train_X, train_Y, input_transform=norm_tf)
            log_ei = LogExpectedImprovement(model=model, best_f=train_Y.max())

            X_cand = torch.tensor([[[7.0, 80.0]]], **tkwargs)

            # Real space: ||[7, 80] - [2, 30]||_2 = sqrt(25 + 2500) = sqrt(2525)
            acqf_real = TransitionCostAcquisitionFunction(
                acq_function=log_ei,
                fixed_cost=1.0,
                scale=1.0,
                transformed_weighting=False,
            )
            expected_real_cost = 1.0 + (2525.0**0.5)
            self.assertAllClose(
                acqf_real(X_cand),
                log_ei(X_cand) - torch.tensor([expected_real_cost], **tkwargs).log(),
            )

            # Normalized space: x_prev_tf = [0.2, 0.3], X_tf = [0.7, 0.8]
            # ||[0.5, 0.5]||_2 = sqrt(0.5)
            model_unfit = SingleTaskGP(
                train_X, train_Y, input_transform=Normalize(2, bounds=bounds)
            )
            log_ei_unfit = LogExpectedImprovement(
                model=model_unfit, best_f=train_Y.max()
            )
            acqf_tf = TransitionCostAcquisitionFunction(
                acq_function=log_ei_unfit,
                fixed_cost=1.0,
                scale=1.0,
                transformed_weighting=True,
            )
            expected_tf_cost = 1.0 + (0.5**0.5)
            self.assertAllClose(
                acqf_tf(X_cand),
                log_ei_unfit(X_cand)
                - torch.tensor([expected_tf_cost], **tkwargs).log(),
            )

            # Explicit x_prev with transformed_weighting=True (with/without transform)
            explicit_prev = torch.tensor([2.0, 30.0], **tkwargs)
            acqf_tf_explicit = TransitionCostAcquisitionFunction(
                acq_function=log_ei,
                x_prev=explicit_prev,
                fixed_cost=1.0,
                scale=1.0,
                transformed_weighting=True,
            )
            self.assertAllClose(acqf_tf_explicit(X_cand), acqf_tf(X_cand))

            model_no_tf = SingleTaskGP(train_X, train_Y).eval()
            log_ei_no_tf = LogExpectedImprovement(
                model=model_no_tf, best_f=train_Y.max()
            )
            acqf_no_tf = TransitionCostAcquisitionFunction(
                acq_function=log_ei_no_tf,
                x_prev=explicit_prev.unsqueeze(0),
                fixed_cost=1.0,
                scale=1.0,
                transformed_weighting=True,
            )
            self.assertAllClose(
                acqf_no_tf(X_cand),
                log_ei_no_tf(X_cand)
                - torch.tensor([expected_real_cost], **tkwargs).log(),
            )

    def test_custom_cost_model_and_x_pending(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            torch.manual_seed(3)
            train_X = torch.rand(6, 2, **tkwargs)
            train_Y = (train_X**2).sum(dim=-1, keepdim=True)
            obj_model = SingleTaskGP(train_X, train_Y).eval()

            # Fit a GP cost model with strictly positive mean (~ 2.0 + x_0)
            train_cost = 2.0 + train_X[:, :1]
            gp_cost_model = SingleTaskGP(train_X, train_cost).eval()

            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]))
            X_pend_init = torch.tensor([[0.2, 0.3]], **tkwargs)
            q_log_ei = qLogExpectedImprovement(
                model=obj_model,
                best_f=train_Y.max(),
                sampler=sampler,
                X_pending=X_pend_init,
            )
            acqf = TransitionCostAcquisitionFunction(
                acq_function=q_log_ei, cost_model=gp_cost_model
            )
            self.assertAllClose(acqf.X_pending, X_pend_init)

            X_test = torch.rand(3, 2, 2, **tkwargs)
            expected_gp_cost = gp_cost_model.posterior(X_test).mean.sum(dim=(-2, -1))
            self.assertAllClose(acqf(X_test), q_log_ei(X_test) - expected_gp_cost.log())

            # Update X_pending and clear X_pending
            X_pend_new = torch.tensor([[0.7, 0.8], [0.1, 0.9]], **tkwargs)
            acqf.set_X_pending(X_pend_new)
            self.assertAllClose(acqf.X_pending, X_pend_new)
            acqf.set_X_pending(None)
            self.assertIsNone(acqf.X_pending)

            # Deterministic and multi-output FixedCostModel custom cost models
            det_pos = GenericDeterministicModel(f=lambda X: 3.0 + X[..., :1].abs())
            self.assertEqual(
                TransitionCostAcquisitionFunction(
                    acq_function=q_log_ei, cost_model=det_pos
                )(X_test).shape,
                torch.Size([3]),
            )
            fixed_multi = FixedCostModel(fixed_cost=torch.tensor([1.0, 2.5], **tkwargs))
            fixed_acqf = TransitionCostAcquisitionFunction(
                acq_function=q_log_ei, cost_model=fixed_multi
            )
            self.assertAllClose(
                fixed_acqf(X_test),
                q_log_ei(X_test) - torch.tensor(7.0, **tkwargs).log(),
            )

            # Verify error when custom cost model returns non-positive cost
            det_nonpos = GenericDeterministicModel(
                f=lambda X: -torch.ones(
                    *X.shape[:-1], 1, dtype=X.dtype, device=X.device
                )
            )
            bad_cost_acqf = TransitionCostAcquisitionFunction(
                acq_function=q_log_ei, cost_model=det_nonpos
            )
            with self.assertRaisesRegex(ValueError, "strictly positive"):
                bad_cost_acqf(X_test)

            # Fully Bayesian / MCMC-style ensemble cost model (_is_ensemble = True)
            mcmc_mean = torch.full((3, 4, 2, 1), 2.0, **tkwargs)
            ens_cost_model = MockModel(MockPosterior(mean=mcmc_mean))
            ens_cost_model._is_ensemble = True
            ens_acqf = TransitionCostAcquisitionFunction(
                acq_function=q_log_ei, cost_model=ens_cost_model
            )
            self.assertAllClose(
                ens_acqf(X_test),
                q_log_ei(X_test) - torch.tensor(4.0, **tkwargs).log(),
            )

            # Analytic acqf raises UnsupportedError on set_X_pending
            log_ei = LogExpectedImprovement(model=obj_model, best_f=train_Y.max())
            tc_analytic = TransitionCostAcquisitionFunction(acq_function=log_ei)
            self.assertIsNone(tc_analytic.X_pending)
            with self.assertRaises(UnsupportedError):
                tc_analytic.set_X_pending(X_pend_init)

    def test_optimize_acqf_pulls_candidates_toward_x_prev(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        torch.manual_seed(11)
        # Peak near (0.9, 0.9), while x_prev is anchored at (0.05, 0.05)
        train_X = torch.tensor(
            [
                [0.05, 0.05],
                [0.25, 0.20],
                [0.50, 0.55],
                [0.80, 0.85],
                [0.90, 0.90],
                [0.05, 0.05],
            ],
            **tkwargs,
        )
        train_Y = -((train_X - 0.9) ** 2).sum(dim=-1, keepdim=True) + 2.0
        model = SingleTaskGP(train_X, train_Y).eval()
        bounds = torch.stack([torch.zeros(2, **tkwargs), torch.ones(2, **tkwargs)])
        x_prev = train_X[-1]

        # q = 1 optimization including initial condition right at x_prev
        log_ei = LogExpectedImprovement(model=model, best_f=train_Y.max())
        acqf_low = TransitionCostAcquisitionFunction(
            acq_function=log_ei, fixed_cost=1.0, scale=0.0
        )
        acqf_high = TransitionCostAcquisitionFunction(
            acq_function=log_ei, fixed_cost=1.0, scale=25.0
        )
        batch_ics = torch.stack(
            [
                x_prev.unsqueeze(0),
                torch.tensor([[0.5, 0.5]], **tkwargs),
                torch.tensor([[0.85, 0.85]], **tkwargs),
            ],
            dim=0,
        )
        cand_low, _ = optimize_acqf(
            acq_function=acqf_low,
            bounds=bounds,
            q=1,
            num_restarts=3,
            batch_initial_conditions=batch_ics,
        )
        cand_high, _ = optimize_acqf(
            acq_function=acqf_high,
            bounds=bounds,
            q=1,
            num_restarts=3,
            batch_initial_conditions=batch_ics,
        )
        dist_low = torch.linalg.vector_norm(cand_low.squeeze(0) - x_prev)
        dist_high = torch.linalg.vector_norm(cand_high.squeeze(0) - x_prev)
        self.assertLess(dist_high.item(), dist_low.item())

        # q = 2 joint optimization with qLogEI
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]))
        q_log_ei = qLogExpectedImprovement(
            model=model, best_f=train_Y.max(), sampler=sampler
        )
        q_acqf_low = TransitionCostAcquisitionFunction(
            acq_function=q_log_ei, fixed_cost=1.0, scale=0.0
        )
        q_acqf_high = TransitionCostAcquisitionFunction(
            acq_function=q_log_ei, fixed_cost=1.0, scale=20.0
        )
        cands_q2_low, _ = optimize_acqf(
            acq_function=q_acqf_low,
            bounds=bounds,
            q=2,
            num_restarts=4,
            raw_samples=16,
        )
        cands_q2_high, _ = optimize_acqf(
            acq_function=q_acqf_high,
            bounds=bounds,
            q=2,
            num_restarts=4,
            raw_samples=16,
        )
        path_low = torch.linalg.vector_norm(
            cands_q2_low[0] - x_prev
        ) + torch.linalg.vector_norm(cands_q2_low[1] - cands_q2_low[0])
        path_high = torch.linalg.vector_norm(
            cands_q2_high[0] - x_prev
        ) + torch.linalg.vector_norm(cands_q2_high[1] - cands_q2_high[0])
        self.assertLess(path_high.item(), path_low.item())

        # Sequential greedy q = 2 optimization (exercises set_X_pending)
        cands_seq, _ = optimize_acqf(
            acq_function=q_acqf_high,
            bounds=bounds,
            q=2,
            num_restarts=3,
            raw_samples=12,
            sequential=True,
        )
        self.assertEqual(cands_seq.shape, torch.Size([2, 2]))

    def test_validation_errors(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X = torch.rand(5, 3, **tkwargs)
        train_Y = train_X.sum(dim=-1, keepdim=True)
        model = SingleTaskGP(train_X, train_Y).eval()
        log_ei = LogExpectedImprovement(model=model, best_f=0.0)
        det_cost = TransitionCostModel(x_prev=train_X[-1])

        for bad_exp in (-0.5, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "`cost_exponent` must be finite"):
                TransitionCostAcquisitionFunction(
                    acq_function=log_ei, cost_exponent=bad_exp
                )

        with self.assertRaisesRegex(ValueError, "Cannot specify `x_prev`"):
            TransitionCostAcquisitionFunction(
                acq_function=log_ei, cost_model=det_cost, x_prev=train_X[-1]
            )
        with self.assertRaisesRegex(
            ValueError, "`transformed_weighting=True` is not supported"
        ):
            TransitionCostAcquisitionFunction(
                acq_function=log_ei, cost_model=det_cost, transformed_weighting=True
            )
        with self.assertRaisesRegex(ValueError, "`x_prev` must have shape"):
            TransitionCostAcquisitionFunction(
                acq_function=log_ei, x_prev=torch.zeros(2, 3, **tkwargs)
            )

        # Model without train_inputs
        dummy_acqf = LogExpectedImprovement(
            model=_DummyModelNoTrainInputs(), best_f=0.0
        )
        with self.assertRaisesRegex(UnsupportedError, "must have `train_inputs`"):
            TransitionCostAcquisitionFunction(acq_function=dummy_acqf)

        # Multi-batch model
        batched_X = torch.rand(2, 5, 3, **tkwargs)
        batched_Y = batched_X.sum(dim=-1, keepdim=True)
        batched_model = SingleTaskGP(batched_X, batched_Y).eval()
        batched_acqf = LogExpectedImprovement(
            model=batched_model, best_f=torch.zeros(2, **tkwargs)
        )
        with self.assertRaisesRegex(UnsupportedError, "single-batch model"):
            TransitionCostAcquisitionFunction(acq_function=batched_acqf)

        # ModelListGP with unequal train_inputs
        ml_unequal_x = ModelListGP(
            SingleTaskGP(train_X[:-1], train_Y[:-1]),
            SingleTaskGP(train_X, train_Y),
        )
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([8]))
        acqf_unequal_x = qLogNoisyExpectedHypervolumeImprovement(
            model=ml_unequal_x,
            ref_point=torch.zeros(2, **tkwargs),
            X_baseline=train_X,
            sampler=sampler,
        )
        with self.assertRaisesRegex(UnsupportedError, "identical training inputs"):
            TransitionCostAcquisitionFunction(acq_function=acqf_unequal_x)

        # ModelListGP with non-equivalent Normalize transforms
        bounds_a = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], **tkwargs)
        bounds_b = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], **tkwargs)
        ml_unequal_tf = ModelListGP(
            SingleTaskGP(
                train_X, train_Y, input_transform=Normalize(3, bounds=bounds_a)
            ),
            SingleTaskGP(
                train_X, train_Y, input_transform=Normalize(3, bounds=bounds_b)
            ),
        )
        acqf_unequal_tf = qLogNoisyExpectedHypervolumeImprovement(
            model=ml_unequal_tf,
            ref_point=torch.zeros(2, **tkwargs),
            X_baseline=train_X,
            sampler=sampler,
        )
        with self.assertRaisesRegex(UnsupportedError, "identical input transforms"):
            TransitionCostAcquisitionFunction(acq_function=acqf_unequal_tf)

        # Model=None, empty ModelListGP/train_inputs, and transform_on_train=False
        model_less_acqf = _SignedLinearAcquisition(model=None)  # type: ignore[arg-type]
        tc_modelless = TransitionCostAcquisitionFunction(
            acq_function=model_less_acqf,
            x_prev=train_X[-1],
            transformed_weighting=True,
        )
        self.assertEqual(tc_modelless(train_X[-1:].unsqueeze(0)).shape, torch.Size([1]))

        empty_ml_acqf = _SignedLinearAcquisition(model=ModelListGP())
        with self.assertRaisesRegex(UnsupportedError, "cannot be empty"):
            TransitionCostAcquisitionFunction(
                acq_function=empty_ml_acqf,
                x_prev=train_X[-1],
                transformed_weighting=True,
            )

        empty_ti_model = _DummyModelNoTrainInputs()
        empty_ti_model.train_inputs = ()
        with self.assertRaisesRegex(UnsupportedError, "must have `train_inputs`"):
            TransitionCostAcquisitionFunction(
                acq_function=_SignedLinearAcquisition(model=empty_ti_model)
            )

        zero_rows_model = _DummyModelNoTrainInputs()
        zero_rows_model.train_inputs = (torch.empty(0, 3, **tkwargs),)
        with self.assertRaisesRegex(UnsupportedError, "cannot be empty"):
            TransitionCostAcquisitionFunction(
                acq_function=_SignedLinearAcquisition(model=zero_rows_model)
            )

        eval_only_tf = Normalize(3, bounds=bounds_b, transform_on_train=False)
        eval_only_gp = SingleTaskGP(
            train_X, train_Y, input_transform=eval_only_tf
        ).eval()
        tc_eval_only = TransitionCostAcquisitionFunction(
            acq_function=LogExpectedImprovement(model=eval_only_gp, best_f=0.0),
            transformed_weighting=True,
        )
        self.assertAllClose(
            tc_eval_only.cost_model.x_prev,
            eval_only_tf.transform(train_X[-1:]).squeeze(0),
        )
