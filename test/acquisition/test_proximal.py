#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.acquisition import LinearMCObjective, ScalarizedPosteriorTransform
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.acquisition.proximal import ProximalAcquisitionFunction
from botorch.exceptions.errors import UnsupportedError
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal


class DummyModel(GPyTorchModel):
    num_outputs = 1

    def __init__(self):  # noqa: D107
        super(GPyTorchModel, self).__init__()

    def subset_output(self, idcs: list[int]) -> Model:
        pass


class DummyAcquisitionFunction(AcquisitionFunction):
    def forward(self, X):
        pass


class NegativeAcquisitionFunction(AcquisitionFunction):
    def forward(self, X):
        return torch.ones(*X.shape[:-2], device=X.device, dtype=X.dtype) * -1.0


class TestProximalAcquisitionFunction(BotorchTestCase):
    def _make_spd_matrix(self, d: int, dtype: torch.dtype) -> Tensor:
        gen = torch.Generator(device="cpu").manual_seed(1234 + d)
        A = torch.randn(d, d, generator=gen, dtype=dtype).to(device=self.device)
        eye = torch.eye(d, device=self.device, dtype=dtype)
        return A @ A.mT + 0.5 * eye

    def test_proximal(self):
        for dtype in (torch.float, torch.double):
            # test single point evaluation with and without input transform
            normalize = Normalize(
                3,
                bounds=torch.tensor(
                    ((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
                    device=self.device,
                    dtype=dtype,
                ),
            )
            for input_transform, x_scale in [(None, 1), (normalize, 2)]:
                train_X = torch.rand(5, 3, device=self.device, dtype=dtype) * x_scale
                train_Y = train_X.norm(dim=-1, keepdim=True)

                # test with and without transformed weights
                for transformed_weighting in [True, False]:
                    # test with single outcome model
                    model = SingleTaskGP(
                        train_X, train_Y, input_transform=input_transform
                    )
                    model = model.to(device=self.device, dtype=dtype).eval()

                    EI = ExpectedImprovement(model, best_f=0.0)

                    proximal_weights = torch.ones(3, device=self.device, dtype=dtype)
                    last_X = train_X[-1]
                    test_X = torch.rand(1, 3, device=self.device, dtype=dtype)
                    EI_prox = ProximalAcquisitionFunction(
                        EI,
                        proximal_weights=proximal_weights,
                        transformed_weighting=transformed_weighting,
                    )
                    self.assertFalse(EI_prox._log)

                    ei = EI(test_X)

                    # modify last_X/test_X depending on transformed_weighting
                    proximal_test_X = test_X.clone()
                    if transformed_weighting:
                        if input_transform is not None:
                            last_X = input_transform(train_X[-1].unsqueeze(0))
                            proximal_test_X = input_transform(test_X)

                    mv_normal = MultivariateNormal(last_X, torch.diag(proximal_weights))
                    test_prox_weight = torch.exp(
                        mv_normal.log_prob(proximal_test_X)
                    ) / torch.exp(mv_normal.log_prob(last_X))

                    ei_prox = EI_prox(test_X)
                    self.assertAllClose(ei_prox, ei * test_prox_weight)
                    self.assertEqual(ei_prox.shape, torch.Size([1]))

                    # test with beta specified
                    EI_prox_beta = ProximalAcquisitionFunction(
                        EI,
                        proximal_weights=proximal_weights,
                        transformed_weighting=transformed_weighting,
                        beta=1.0,
                    )
                    self.assertFalse(EI_prox_beta._log)

                    # SoftPlus transformed value of the acquisition function
                    ei = torch.nn.functional.softplus(EI(test_X), beta=1.0)

                    # modify last_X/test_X depending on transformed_weighting
                    proximal_test_X = test_X.clone()
                    if transformed_weighting:
                        if input_transform is not None:
                            last_X = input_transform(train_X[-1].unsqueeze(0))
                            proximal_test_X = input_transform(test_X)

                    mv_normal = MultivariateNormal(last_X, torch.diag(proximal_weights))
                    test_prox_weight = torch.exp(
                        mv_normal.log_prob(proximal_test_X) - mv_normal.log_prob(last_X)
                    )

                    ei_prox_beta = EI_prox_beta(test_X)
                    self.assertAllClose(ei_prox_beta, ei * test_prox_weight)
                    self.assertEqual(ei_prox_beta.shape, torch.Size([1]))

                    # test t-batch with broadcasting
                    test_X = torch.rand(4, 1, 3, device=self.device, dtype=dtype)
                    proximal_test_X = test_X.clone()
                    if transformed_weighting:
                        if input_transform is not None:
                            last_X = input_transform(train_X[-1].unsqueeze(0))
                            proximal_test_X = input_transform(test_X)

                    ei = EI(test_X)
                    mv_normal = MultivariateNormal(last_X, torch.diag(proximal_weights))
                    test_prox_weight = torch.exp(
                        mv_normal.log_prob(proximal_test_X)
                    ) / torch.exp(mv_normal.log_prob(last_X))

                    ei_prox = EI_prox(test_X)
                    self.assertTrue(
                        torch.allclose(ei_prox, ei * test_prox_weight.flatten())
                    )
                    self.assertEqual(ei_prox.shape, torch.Size([4]))

                    # test q-based MC acquisition function
                    qEI = qExpectedImprovement(model, best_f=0.0)
                    test_X = torch.rand(4, 1, 3, device=self.device, dtype=dtype)
                    proximal_test_X = test_X.clone()
                    if transformed_weighting:
                        if input_transform is not None:
                            last_X = input_transform(train_X[-1].unsqueeze(0))
                            proximal_test_X = input_transform(test_X)

                    qEI_prox = ProximalAcquisitionFunction(
                        qEI,
                        proximal_weights=proximal_weights,
                        transformed_weighting=transformed_weighting,
                    )

                    qei = qEI(test_X)
                    mv_normal = MultivariateNormal(last_X, torch.diag(proximal_weights))
                    test_prox_weight = torch.exp(
                        mv_normal.log_prob(proximal_test_X)
                    ) / torch.exp(mv_normal.log_prob(last_X))

                    qei_prox = qEI_prox(test_X)
                    self.assertTrue(
                        torch.allclose(qei_prox, qei * test_prox_weight.flatten())
                    )
                    self.assertEqual(qei_prox.shape, torch.Size([4]))

                    # test acquisition function with
                    # negative values w/o SoftPlus transform specified
                    negative_acqf = NegativeAcquisitionFunction(model)
                    bad_neg_prox = ProximalAcquisitionFunction(
                        negative_acqf, proximal_weights=proximal_weights
                    )

                    with self.assertRaisesRegex(
                        RuntimeError, "Cannot use proximal biasing for negative"
                    ):
                        bad_neg_prox(test_X)

            # test gradient
            test_X = torch.rand(
                1, 3, device=self.device, dtype=dtype, requires_grad=True
            )
            ei_prox = EI_prox(test_X)
            ei_prox.backward()

            # test model without train_inputs
            bad_model = DummyModel()
            with self.assertRaises(UnsupportedError):
                ProximalAcquisitionFunction(
                    ExpectedImprovement(bad_model, 0.0), proximal_weights
                )

            # test proximal weights that do not match training_inputs or are non-SPD
            train_X = torch.rand(5, 1, 3, device=self.device, dtype=dtype)
            train_Y = train_X.norm(dim=-1, keepdim=True)
            model = SingleTaskGP(train_X, train_Y).to(device=self.device).eval()
            with self.assertRaises(ValueError):
                ProximalAcquisitionFunction(
                    ExpectedImprovement(model, 0.0), proximal_weights[:1]
                )

            non_sym = torch.eye(3, device=self.device, dtype=dtype)
            non_sym[0, 1] = 1.0
            with self.assertRaises(ValueError):
                ProximalAcquisitionFunction(ExpectedImprovement(model, 0.0), non_sym)

            # test for x_pending points
            pending_acq = DummyAcquisitionFunction(model)
            pending_acq.set_X_pending(torch.rand(3, 3, device=self.device, dtype=dtype))
            with self.assertRaises(UnsupportedError):
                ProximalAcquisitionFunction(pending_acq, proximal_weights)

            # test model with multi-batch training inputs
            train_X = torch.rand(5, 2, 3, device=self.device, dtype=dtype)
            train_Y = train_X.norm(dim=-1, keepdim=True)
            bad_single_task = (
                SingleTaskGP(train_X, train_Y).to(device=self.device).eval()
            )
            with self.assertRaises(UnsupportedError):
                ProximalAcquisitionFunction(
                    ExpectedImprovement(bad_single_task, 0.0), proximal_weights
                )

            # test a multi-output SingleTaskGP model
            train_X = torch.rand(5, 3, device=self.device, dtype=dtype)
            train_Y = torch.rand(5, 2, device=self.device, dtype=dtype)

            multi_output_model = SingleTaskGP(train_X, train_Y).to(device=self.device)
            ptransform = ScalarizedPosteriorTransform(
                weights=torch.ones(2, dtype=dtype, device=self.device)
            )
            ei = ExpectedImprovement(
                multi_output_model, 0.0, posterior_transform=ptransform
            )
            acq = ProximalAcquisitionFunction(ei, proximal_weights)
            acq(test_X)

    def test_proximal_model_list(self):
        for dtype in (torch.float, torch.double):
            proximal_weights = torch.ones(3, device=self.device, dtype=dtype)

            # test with model-list model for complex objective optimization
            train_X = torch.rand(5, 3, device=self.device, dtype=dtype)
            train_Y = train_X.norm(dim=-1, keepdim=True)

            gp = SingleTaskGP(train_X, train_Y).to(device=self.device)
            model = ModelListGP(gp, gp)

            scalarized_posterior_transform = ScalarizedPosteriorTransform(
                torch.ones(2, device=self.device, dtype=dtype)
            )
            mc_linear_objective = LinearMCObjective(
                torch.ones(2, device=self.device, dtype=dtype)
            )

            EI = ExpectedImprovement(
                model, best_f=0.0, posterior_transform=scalarized_posterior_transform
            )

            test_X = torch.rand(1, 3, device=self.device, dtype=dtype)
            EI_prox = ProximalAcquisitionFunction(EI, proximal_weights=proximal_weights)

            ei = EI(test_X)
            mv_normal = MultivariateNormal(train_X[-1], torch.diag(proximal_weights))
            test_prox_weight = torch.exp(mv_normal.log_prob(test_X)) / torch.exp(
                mv_normal.log_prob(train_X[-1])
            )

            # test calculation
            ei_prox = EI_prox(test_X)

            self.assertAllClose(ei_prox, ei * test_prox_weight)
            self.assertEqual(ei_prox.shape, torch.Size([1]))

            # test MC acquisition function
            qEI = qExpectedImprovement(model, best_f=0.0, objective=mc_linear_objective)
            test_X = torch.rand(4, 1, 3, device=self.device, dtype=dtype)
            qEI_prox = ProximalAcquisitionFunction(
                qEI, proximal_weights=proximal_weights
            )

            qei = qEI(test_X)
            mv_normal = MultivariateNormal(train_X[-1], torch.diag(proximal_weights))
            test_prox_weight = torch.exp(mv_normal.log_prob(test_X)) / torch.exp(
                mv_normal.log_prob(train_X[-1])
            )

            qei_prox = qEI_prox(test_X)
            self.assertAllClose(qei_prox, qei * test_prox_weight.flatten())
            self.assertEqual(qei_prox.shape, torch.Size([4]))

            # test gradient
            test_X = torch.rand(
                1, 3, device=self.device, dtype=dtype, requires_grad=True
            )
            ei_prox = EI_prox(test_X)
            ei_prox.backward()

            # test proximal weights that do not match training_inputs
            expected_shape_err_msg = (
                "`proximal_weights` must be a one-dimensional tensor of shape "
                "`\\(d,\\)` or a two-dimensional tensor of shape `\\(d, d\\)`"
            )
            with self.assertRaisesRegex(ValueError, expected_shape_err_msg):
                ProximalAcquisitionFunction(
                    ExpectedImprovement(
                        model, 0.0, posterior_transform=scalarized_posterior_transform
                    ),
                    proximal_weights[:1],
                )

            with self.assertRaisesRegex(ValueError, expected_shape_err_msg):
                ProximalAcquisitionFunction(
                    ExpectedImprovement(
                        model, 0.0, posterior_transform=scalarized_posterior_transform
                    ),
                    torch.rand(3, 2, device=self.device, dtype=dtype),
                )

            non_sym = torch.eye(3, device=self.device, dtype=dtype)
            non_sym[0, 1] = 0.5
            with self.assertRaisesRegex(
                ValueError, "symmetric positive-definite matrix"
            ):
                ProximalAcquisitionFunction(
                    ExpectedImprovement(
                        model, 0.0, posterior_transform=scalarized_posterior_transform
                    ),
                    non_sym,
                )

            # test for x_pending points
            pending_acq = DummyAcquisitionFunction(model)
            pending_acq.set_X_pending(torch.rand(3, 3, device=self.device, dtype=dtype))
            with self.assertRaises(UnsupportedError):
                ProximalAcquisitionFunction(pending_acq, proximal_weights)

            # test model with multi-batch training inputs
            train_X = torch.rand(5, 2, 3, device=self.device, dtype=dtype)
            train_Y = train_X.norm(dim=-1, keepdim=True)
            bad_model = ModelListGP(
                SingleTaskGP(train_X, train_Y).to(device=self.device),
                SingleTaskGP(train_X, train_Y).to(device=self.device),
            )
            with self.assertRaises(UnsupportedError):
                ProximalAcquisitionFunction(
                    ExpectedImprovement(
                        bad_model,
                        0.0,
                        posterior_transform=scalarized_posterior_transform,
                    ),
                    proximal_weights,
                )

            # try using unequal training sets
            train_X = torch.rand(5, 3, device=self.device, dtype=dtype)
            train_Y = train_X.norm(dim=-1, keepdim=True)
            bad_model = ModelListGP(
                SingleTaskGP(train_X[:-1], train_Y[:-1]).to(device=self.device),
                SingleTaskGP(train_X, train_Y).to(device=self.device),
            )
            with self.assertRaises(UnsupportedError):
                ProximalAcquisitionFunction(
                    ExpectedImprovement(
                        bad_model,
                        0.0,
                        posterior_transform=scalarized_posterior_transform,
                    ),
                    proximal_weights,
                )

            # try with unequal input transforms (one Normalize, one None)
            train_X = torch.rand(5, 3, device=self.device, dtype=dtype)
            train_Y = train_X.norm(dim=-1, keepdim=True)
            bad_model = ModelListGP(
                SingleTaskGP(train_X, train_Y, input_transform=Normalize(3)).to(
                    device=self.device
                ),
                SingleTaskGP(train_X, train_Y).to(device=self.device),
            )
            with self.assertRaises(UnsupportedError):
                ProximalAcquisitionFunction(
                    ExpectedImprovement(
                        bad_model,
                        0.0,
                        posterior_transform=scalarized_posterior_transform,
                    ),
                    proximal_weights,
                )

            # try with two Normalize transforms having different bounds
            b1 = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=self.device, dtype=dtype
            )
            b2 = torch.tensor(
                [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], device=self.device, dtype=dtype
            )
            bad_bounds_model = ModelListGP(
                SingleTaskGP(
                    train_X, train_Y, input_transform=Normalize(3, bounds=b1)
                ).to(device=self.device),
                SingleTaskGP(
                    train_X, train_Y, input_transform=Normalize(3, bounds=b2)
                ).to(device=self.device),
            )
            with self.assertRaisesRegex(
                UnsupportedError, "non-identical input transforms"
            ):
                ProximalAcquisitionFunction(
                    ExpectedImprovement(
                        bad_bounds_model,
                        0.0,
                        posterior_transform=scalarized_posterior_transform,
                    ),
                    proximal_weights,
                )

            # separate equivalent Normalize(3) instances on submodels must succeed
            eq_model = ModelListGP(
                SingleTaskGP(train_X, train_Y, input_transform=Normalize(3)).to(
                    device=self.device
                ),
                SingleTaskGP(train_X, train_Y, input_transform=Normalize(3)).to(
                    device=self.device
                ),
            ).eval()
            eq_ei = ExpectedImprovement(
                eq_model, best_f=0.0, posterior_transform=scalarized_posterior_transform
            )
            eq_prox = ProximalAcquisitionFunction(eq_ei, proximal_weights)
            val = eq_prox(torch.rand(2, 1, 3, device=self.device, dtype=dtype))
            self.assertEqual(val.shape, torch.Size([2]))

    def test_proximal_log_acquisition_functions(self):
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor(
                [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], device=self.device, dtype=dtype
            )
            train_X = torch.tensor(
                [
                    [0.2, 0.4, 0.6],
                    [0.5, 1.1, 0.8],
                    [1.2, 0.7, 1.5],
                    [1.6, 1.4, 0.9],
                    [1.0, 1.0, 1.0],
                ],
                device=self.device,
                dtype=dtype,
            )
            train_Y = train_X.norm(dim=-1, keepdim=True)
            test_X = torch.tensor(
                [[[0.4, 0.8, 1.2]], [[1.5, 1.2, 0.7]], [[1.0, 1.0, 1.0]]],
                device=self.device,
                dtype=dtype,
            )
            weights_1d = torch.tensor([0.5, 0.8, 1.2], device=self.device, dtype=dtype)
            M_2d = self._make_spd_matrix(3, dtype=dtype)

            for use_transform in (False, True):
                tf = Normalize(3, bounds=bounds) if use_transform else None
                model = (
                    SingleTaskGP(train_X, train_Y, input_transform=tf)
                    .to(device=self.device, dtype=dtype)
                    .eval()
                )
                best_f = train_Y.max().item()
                ei = ExpectedImprovement(model, best_f=best_f)
                log_ei = LogExpectedImprovement(model, best_f=best_f)

                for transformed_weighting in (True, False):
                    for pw in (weights_1d, M_2d):
                        prox_ei = ProximalAcquisitionFunction(
                            ei,
                            proximal_weights=pw,
                            transformed_weighting=transformed_weighting,
                        )
                        prox_log_ei = ProximalAcquisitionFunction(
                            log_ei,
                            proximal_weights=pw,
                            transformed_weighting=transformed_weighting,
                        )
                        self.assertTrue(prox_log_ei._log)
                        self.assertFalse(prox_ei._log)

                        # Compute reference log proximal weight via MultivariateNormal
                        cov = torch.diag(pw.square()) if pw.ndim == 1 else pw
                        if use_transform and transformed_weighting:
                            ref_last = model.input_transform(train_X[-1:])
                            ref_X = model.input_transform(test_X)
                        else:
                            ref_last = train_X[-1:]
                            ref_X = test_X
                        mvn = MultivariateNormal(ref_last, covariance_matrix=cov)
                        expected_log_weight = (
                            mvn.log_prob(ref_X) - mvn.log_prob(ref_last)
                        ).squeeze(-1)

                        log_vals = prox_log_ei(test_X)
                        # Exact additive identity in log space
                        self.assertAllClose(
                            log_vals, log_ei(test_X) + expected_log_weight
                        )
                        # Exponentiated LogEI matches linear EI * prox_weight
                        self.assertAllClose(
                            torch.exp(log_vals),
                            prox_ei(test_X),
                            rtol=1e-4 if dtype == torch.float else 1e-6,
                            atol=1e-6,
                        )

                # Explicit beta on a _log=True acqf uses softplus in linear space
                prox_log_ei_beta = ProximalAcquisitionFunction(
                    log_ei, proximal_weights=weights_1d, beta=1.5
                )
                self.assertFalse(prox_log_ei_beta._log)
                cov_1d = torch.diag(weights_1d.square())
                ref_last = (
                    model.input_transform(train_X[-1:])
                    if use_transform
                    else train_X[-1:]
                )
                ref_X = model.input_transform(test_X) if use_transform else test_X
                mvn_1d = MultivariateNormal(ref_last, covariance_matrix=cov_1d)
                w_1d = torch.exp(
                    mvn_1d.log_prob(ref_X) - mvn_1d.log_prob(ref_last)
                ).squeeze(-1)
                expected_beta_val = (
                    torch.nn.functional.softplus(log_ei(test_X), beta=1.5) * w_1d
                )
                self.assertAllClose(prox_log_ei_beta(test_X), expected_beta_val)

            # Test underflow / vanishing-gradient regime in log space
            far_log_ei = LogExpectedImprovement(model, best_f=50.0)
            tight_weights = torch.full((3,), 0.03, device=self.device, dtype=dtype)
            prox_far_log_ei = ProximalAcquisitionFunction(
                far_log_ei, proximal_weights=tight_weights, transformed_weighting=False
            )
            X_grad = torch.tensor(
                [[0.1, 1.9, 0.2]], device=self.device, dtype=dtype, requires_grad=True
            )
            far_val = prox_far_log_ei(X_grad)
            self.assertTrue(torch.isfinite(far_val).all())
            self.assertTrue((far_val < -500.0).all())
            far_val.sum().backward()
            self.assertIsNotNone(X_grad.grad)
            self.assertTrue(torch.isfinite(X_grad.grad).all())
            self.assertGreater(X_grad.grad.norm().item(), 1.0)

            # Test qLogExpectedImprovement vs qExpectedImprovement
            sampler_qei = SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=0)
            sampler_qlogei = SobolQMCNormalSampler(
                sample_shape=torch.Size([64]), seed=0
            )
            qei = qExpectedImprovement(model, best_f=0.5, sampler=sampler_qei)
            qlog_ei = qLogExpectedImprovement(
                model, best_f=0.5, sampler=sampler_qlogei, tau_relu=1e-8
            )
            prox_qei = ProximalAcquisitionFunction(qei, proximal_weights=M_2d)
            prox_qlog_ei = ProximalAcquisitionFunction(qlog_ei, proximal_weights=M_2d)
            self.assertTrue(prox_qlog_ei._log)
            self.assertAllClose(
                torch.exp(prox_qlog_ei(test_X)),
                prox_qei(test_X),
                rtol=1e-4 if dtype == torch.float else 1e-6,
                atol=1e-5,
            )

        # Test MOBO with qLogNEHVI & qLogNParEGO (both multi-output SingleTaskGP and
        # ModelListGP with separate Normalize(d) transforms) + optimize_acqf
        dtype = torch.double
        bounds_mo = torch.tensor(
            [[0.0, 0.0], [2.0, 3.0]], device=self.device, dtype=dtype
        )
        train_X_mo = torch.tensor(
            [[0.2, 0.5], [0.8, 1.2], [1.4, 2.1], [1.1, 1.6]],
            device=self.device,
            dtype=dtype,
        )
        train_Y_mo = torch.stack(
            [train_X_mo.sum(dim=-1), (2.0 - train_X_mo[:, 0]) + train_X_mo[:, 1]],
            dim=-1,
        )
        ref_point = torch.tensor([0.0, 0.0], device=self.device, dtype=dtype)
        M_mo = self._make_spd_matrix(2, dtype=dtype)
        test_X_mo = torch.tensor(
            [[[0.5, 1.0]], [[1.1, 1.6]], [[1.7, 2.5]]],
            device=self.device,
            dtype=dtype,
        )

        mo_stgp = (
            SingleTaskGP(
                train_X_mo,
                train_Y_mo,
                input_transform=Normalize(2, bounds=bounds_mo),
            )
            .to(device=self.device, dtype=dtype)
            .eval()
        )
        mo_list_gp = ModelListGP(
            SingleTaskGP(
                train_X_mo,
                train_Y_mo[:, :1],
                input_transform=Normalize(2, bounds=bounds_mo),
            ),
            SingleTaskGP(
                train_X_mo,
                train_Y_mo[:, 1:],
                input_transform=Normalize(2, bounds=bounds_mo),
            ),
        ).to(device=self.device, dtype=dtype)
        mo_list_gp.eval()

        for mo_model in (mo_stgp, mo_list_gp):
            qlog_nehvi = qLogNoisyExpectedHypervolumeImprovement(
                model=mo_model,
                ref_point=ref_point,
                X_baseline=train_X_mo,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([32]), seed=0),
                tau_relu=1e-8,
            )
            qnehvi = qNoisyExpectedHypervolumeImprovement(
                model=mo_model,
                ref_point=ref_point,
                X_baseline=train_X_mo,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([32]), seed=0),
            )
            prox_qlog_nehvi = ProximalAcquisitionFunction(
                qlog_nehvi, proximal_weights=M_mo, transformed_weighting=False
            )
            prox_qnehvi = ProximalAcquisitionFunction(
                qnehvi, proximal_weights=M_mo, transformed_weighting=False
            )
            self.assertTrue(prox_qlog_nehvi._log)

            mvn_mo = MultivariateNormal(train_X_mo[-1:], covariance_matrix=M_mo)
            log_w_mo = (
                mvn_mo.log_prob(test_X_mo) - mvn_mo.log_prob(train_X_mo[-1:])
            ).squeeze(-1)
            self.assertAllClose(
                prox_qlog_nehvi(test_X_mo), qlog_nehvi(test_X_mo) + log_w_mo
            )
            self.assertAllClose(
                torch.exp(prox_qlog_nehvi(test_X_mo)),
                prox_qnehvi(test_X_mo),
                rtol=1e-4,
                atol=1e-5,
            )

        # Also test qLogNParEGO with ModelListGP & learned Normalize(2) bounds
        parego_model = ModelListGP(
            SingleTaskGP(train_X_mo, train_Y_mo[:, :1], input_transform=Normalize(2)),
            SingleTaskGP(train_X_mo, train_Y_mo[:, 1:], input_transform=Normalize(2)),
        ).to(device=self.device, dtype=dtype)
        parego_model.eval()
        qlog_parego = qLogNParEGO(
            model=parego_model,
            X_baseline=train_X_mo,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([32])),
        )
        prox_parego = ProximalAcquisitionFunction(qlog_parego, proximal_weights=M_mo)
        self.assertTrue(prox_parego._log)
        parego_vals = prox_parego(test_X_mo)
        self.assertEqual(parego_vals.shape, torch.Size([3]))
        self.assertTrue(torch.isfinite(parego_vals).all())

        # End-to-end optimize_acqf check: strong proximal biasing pulls candidate to
        # the last observed point train_X_mo[-1]
        tight_M = 1e-8 * torch.eye(2, device=self.device, dtype=dtype)
        prox_mobo_tight = ProximalAcquisitionFunction(
            qLogNoisyExpectedHypervolumeImprovement(
                model=mo_list_gp,
                ref_point=ref_point,
                X_baseline=train_X_mo,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16])),
            ),
            proximal_weights=tight_M,
            transformed_weighting=False,
        )
        cand, acq_val = optimize_acqf(
            acq_function=prox_mobo_tight,
            bounds=bounds_mo,
            q=1,
            num_restarts=2,
            raw_samples=16,
        )
        self.assertEqual(cand.shape, torch.Size([1, 2]))
        self.assertTrue(torch.isfinite(acq_val).all())
        self.assertAllClose(cand.squeeze(0), train_X_mo[-1], atol=1e-2, rtol=1e-2)

    def test_proximal_metric_matrix(self):
        for dtype in (torch.float, torch.double):
            # 1D vs 2D diagonal equivalence (w in 1D corresponds to M = diag(w^2))
            train_X = torch.rand(5, 3, device=self.device, dtype=dtype)
            train_Y = train_X.norm(dim=-1, keepdim=True)
            model = SingleTaskGP(train_X, train_Y).to(device=self.device).eval()
            log_ei = LogExpectedImprovement(model, best_f=0.5)
            ei = ExpectedImprovement(model, best_f=0.5)

            w_1d = torch.tensor([0.4, 1.3, 2.2], device=self.device, dtype=dtype)
            M_diag = torch.diag(w_1d.square())
            test_X = torch.rand(6, 1, 3, device=self.device, dtype=dtype)

            for base_acqf in (ei, log_ei):
                p_1d = ProximalAcquisitionFunction(base_acqf, proximal_weights=w_1d)
                p_2d = ProximalAcquisitionFunction(base_acqf, proximal_weights=M_diag)
                self.assertAllClose(p_1d(test_X), p_2d(test_X))

            # Full non-diagonal SPD matrix M with off-diagonal cross-terms
            M_full = self._make_spd_matrix(3, dtype=dtype)
            bounds = torch.tensor(
                [[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]], device=self.device, dtype=dtype
            )
            for tf, scale in [(None, 1.0), (Normalize(3, bounds=bounds), 2.0)]:
                tX = torch.rand(5, 3, device=self.device, dtype=dtype) * scale
                tY = tX.norm(dim=-1, keepdim=True)
                m = SingleTaskGP(tX, tY, input_transform=tf).to(self.device).eval()
                X_eval = torch.rand(5, 1, 3, device=self.device, dtype=dtype) * scale

                for tw in (True, False):
                    ref_last = (
                        m.input_transform(tX[-1:])
                        if (tf is not None and tw)
                        else tX[-1:]
                    )
                    ref_X = (
                        m.input_transform(X_eval) if (tf is not None and tw) else X_eval
                    )
                    mvn = MultivariateNormal(ref_last, covariance_matrix=M_full)
                    log_w = (mvn.log_prob(ref_X) - mvn.log_prob(ref_last)).squeeze(-1)

                    ei_m = ExpectedImprovement(m, best_f=0.0)
                    prox_ei_m = ProximalAcquisitionFunction(
                        ei_m, proximal_weights=M_full, transformed_weighting=tw
                    )
                    self.assertAllClose(prox_ei_m(X_eval), ei_m(X_eval) * log_w.exp())

                    log_ei_m = LogExpectedImprovement(m, best_f=0.0)
                    prox_log_ei_m = ProximalAcquisitionFunction(
                        log_ei_m, proximal_weights=M_full, transformed_weighting=tw
                    )
                    self.assertAllClose(prox_log_ei_m(X_eval), log_ei_m(X_eval) + log_w)

            # Gradient & second-derivative (Hessian) stability at X == last_X (diff = 0)
            X_at_last = train_X[-1:].clone().detach().requires_grad_(True)
            for pw in (w_1d, M_full):
                prox_at_last = ProximalAcquisitionFunction(log_ei, proximal_weights=pw)
                out = prox_at_last(X_at_last)
                grad = torch.autograd.grad(out.sum(), X_at_last, create_graph=True)[0]
                hess_diag = torch.autograd.grad(grad.sum(), X_at_last)[0]
                self.assertTrue(torch.isfinite(grad).all())
                self.assertTrue(torch.isfinite(hess_diag).all())

            # d = 1 edge case for both (1,) and (1, 1) proximal_weights
            tX_1d = torch.rand(4, 1, device=self.device, dtype=dtype)
            tY_1d = tX_1d.square()
            m_1d = SingleTaskGP(tX_1d, tY_1d).to(device=self.device).eval()
            log_ei_1d = LogExpectedImprovement(m_1d, best_f=0.1)
            p1 = ProximalAcquisitionFunction(
                log_ei_1d, torch.tensor([0.7], device=self.device, dtype=dtype)
            )
            p2 = ProximalAcquisitionFunction(
                log_ei_1d, torch.tensor([[0.49]], device=self.device, dtype=dtype)
            )
            X_test_1d = torch.rand(3, 1, 1, device=self.device, dtype=dtype)
            self.assertEqual(p1(X_test_1d).shape, torch.Size([3]))
            self.assertAllClose(p1(X_test_1d), p2(X_test_1d))

    def test_proximal_validation_errors(self):
        for dtype in (torch.float, torch.double):
            train_X = torch.rand(5, 3, device=self.device, dtype=dtype)
            train_Y = train_X.norm(dim=-1, keepdim=True)
            model = SingleTaskGP(train_X, train_Y).to(device=self.device).eval()
            acqf = LogExpectedImprovement(model, best_f=0.0)

            # Non-floating-point weights
            with self.assertRaisesRegex(ValueError, "floating-point tensor"):
                ProximalAcquisitionFunction(
                    acqf, torch.ones(3, device=self.device, dtype=torch.long)
                )

            # Wrong 1D length, wrong 2D shape, 0D scalar, and 3D tensor
            for bad_shape_w in (
                torch.ones(2, device=self.device, dtype=dtype),
                torch.ones(4, device=self.device, dtype=dtype),
                torch.ones(3, 2, device=self.device, dtype=dtype),
                torch.ones(2, 3, device=self.device, dtype=dtype),
                torch.tensor(1.0, device=self.device, dtype=dtype),
                torch.ones(3, 3, 3, device=self.device, dtype=dtype),
            ):
                with self.assertRaisesRegex(
                    ValueError, "one-dimensional tensor of shape"
                ):
                    ProximalAcquisitionFunction(acqf, bad_shape_w)

            # Non-positive 1D weights (zero, negative, NaN)
            for bad_1d in (
                torch.tensor([1.0, 0.0, 1.0], device=self.device, dtype=dtype),
                torch.tensor([1.0, -0.5, 1.0], device=self.device, dtype=dtype),
                torch.tensor([1.0, float("nan"), 1.0], device=self.device, dtype=dtype),
            ):
                with self.assertRaisesRegex(ValueError, "strictly positive"):
                    ProximalAcquisitionFunction(acqf, bad_1d)

            # Asymmetric 2D matrices (both unit scale and small scale)
            asym = torch.eye(3, device=self.device, dtype=dtype)
            asym[0, 2] = 0.3
            asym_small = 1e-10 * asym
            for bad_asym in (asym, asym_small):
                with self.assertRaisesRegex(
                    ValueError, "symmetric positive-definite matrix"
                ):
                    ProximalAcquisitionFunction(acqf, bad_asym)

            # Symmetric but non-positive-definite (singular PSD, indefinite, NaN/Inf)
            singular_psd = torch.diag(
                torch.tensor([1.0, 0.0, 1.0], device=self.device, dtype=dtype)
            )
            indefinite = torch.diag(
                torch.tensor([1.0, -0.2, 1.0], device=self.device, dtype=dtype)
            )
            nan_mat = torch.eye(3, device=self.device, dtype=dtype)
            nan_mat[0, 0] = float("nan")
            inf_mat = torch.eye(3, device=self.device, dtype=dtype)
            inf_mat[0, 0] = float("inf")
            for bad_2d in (singular_psd, indefinite, nan_mat, inf_mat):
                with self.assertRaisesRegex(
                    ValueError, "symmetric positive-definite matrix"
                ):
                    ProximalAcquisitionFunction(acqf, bad_2d)

    def test_proximal_multidim_t_batch(self):
        for dtype in (torch.float, torch.double):
            train_X = torch.rand(5, 3, device=self.device, dtype=dtype)
            train_Y = train_X.norm(dim=-1, keepdim=True)
            model = SingleTaskGP(train_X, train_Y).to(device=self.device).eval()

            ei = ExpectedImprovement(model, best_f=0.0)
            log_ei = LogExpectedImprovement(model, best_f=0.0)
            qei = qExpectedImprovement(
                model,
                best_f=0.0,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16])),
            )
            qlog_ei = qLogExpectedImprovement(
                model,
                best_f=0.0,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16])),
            )

            w_1d = torch.tensor([0.6, 1.1, 1.5], device=self.device, dtype=dtype)
            M_2d = self._make_spd_matrix(3, dtype=dtype)

            for batch_shape in ((2, 3), (2, 3, 2)):
                X_batch = torch.rand(
                    *batch_shape, 1, 3, device=self.device, dtype=dtype
                )
                X_flat = X_batch.reshape(-1, 1, 3)

                for base_acqf in (ei, log_ei, qei, qlog_ei):
                    for pw in (w_1d, M_2d):
                        prox = ProximalAcquisitionFunction(
                            base_acqf, proximal_weights=pw
                        )
                        out_batch = prox(X_batch)
                        out_flat = prox(X_flat).reshape(batch_shape)
                        self.assertEqual(out_batch.shape, torch.Size(batch_shape))
                        self.assertAllClose(out_batch, out_flat)
