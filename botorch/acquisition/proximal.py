#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A wrapper around AcquisitionFunctions to add proximal weighting of the
acquisition function, see [Roussel2021mobo]_ and [Roussel2021proximal]_ for
details. Supports both positive linear-space acquisition functions and
log-space acquisition functions (``_log = True``, see [Ament2023logei]_).

References

.. [Roussel2021mobo]
    R. Roussel, A. Hanuka, and A. Edelen. Multiobjective Bayesian optimization
    for online accelerator tuning. Physical Review Accelerators and Beams,
    24(6):062801, 2021.

.. [Roussel2021proximal]
    R. Roussel and A. Edelen. Proximal Biasing for Bayesian Optimization and
    Characterization of Physical Systems. Advances in Neural Information
    Processing Systems Workshop on Machine Learning and the Physical Sciences,
    2021.
"""

from __future__ import annotations

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.exceptions.errors import UnsupportedError
from botorch.models import ModelListGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.utils import t_batch_mode_transform
from torch import Tensor


class ProximalAcquisitionFunction(AcquisitionFunction):
    r"""A wrapper around AcquisitionFunctions to add proximal weighting of the
    acquisition function.

    The acquisition function is weighted via a squared exponential centered at
    the last training point, with lengthscales or metric covariance matrix
    corresponding to ``proximal_weights`` [Roussel2021mobo]_,
    [Roussel2021proximal]_:

    ``alpha_prox(x) = alpha(x) * exp(-0.5 * d_M(x, x_last)^2)``

    where ``d_M(x, x_last)^2 = ||(x - x_last) / proximal_weights||_2^2`` when
    ``proximal_weights`` is a 1D tensor of positive lengthscales of shape
    ``(d,)``, or ``(x - x_last)^T M^{-1} (x - x_last)`` when
    ``proximal_weights`` is a 2D symmetric positive-definite matrix ``M`` of
    shape ``(d, d)``.

    When wrapping a log-space acquisition function (``acq_function._log == True``,
    e.g., ``LogExpectedImprovement`` or
    ``qLogNoisyExpectedHypervolumeImprovement``, see [Ament2023logei]_) and
    ``beta is None``, the proximal penalty is applied additively in log space
    (``log_alpha(x) - 0.5 * d_M(x, x_last)^2``) and ``self._log`` is set to
    ``True``. Otherwise, the base acquisition function must be non-negative or
    ``beta`` must be specified to apply a SoftPlus transform before proximal
    weighting. Can only be used with acquisition functions based on single-batch
    models.

    Small values of ``proximal_weights`` correspond to strong biasing towards
    recently observed points, which smooths optimization with a small potential
    decrease in convergence rate.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> LogEI = LogExpectedImprovement(model, best_f=0.0)
        >>> proximal_weights = torch.ones(d)
        >>> LogEI_proximal = ProximalAcquisitionFunction(LogEI, proximal_weights)
        >>> eip = LogEI_proximal(test_X)
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        proximal_weights: Tensor,
        transformed_weighting: bool = True,
        beta: float | None = None,
    ) -> None:
        r"""Derived Acquisition Function weighted by proximity to recently
        observed point.

        Args:
            acq_function: The base acquisition function, operating on input tensors
                of feature dimension ``d``.
            proximal_weights: Either a ``(d,)``-dim tensor of strictly positive
                lengthscales biasing locality along each axis, or a ``(d, d)``-dim
                symmetric positive-definite covariance/metric matrix ``M`` defining
                the Mahalanobis squared distance
                ``(x - x_last)^T M^{-1} (x - x_last)``.
            transformed_weighting: If True, the proximal weights are applied in
                the transformed input space given by
                ``acq_function.model.input_transform`` (if available), otherwise
                proximal weights are applied in real input space.
            beta: If not None, apply a softplus transform to the base acquisition
                function, allowing negative base acquisition function values in
                linear space. When ``None`` and ``acq_function._log`` is ``True``,
                proximal weighting is computed directly in log space without a
                softplus transform.
        """
        super().__init__(model=acq_function.model)
        self.acq_func = acq_function

        if hasattr(acq_function, "X_pending"):
            if acq_function.X_pending is not None:
                raise UnsupportedError(
                    "Proximal acquisition function requires `X_pending` to be None."
                )
            self.X_pending = acq_function.X_pending

        self.register_buffer(
            "transformed_weighting", torch.tensor(transformed_weighting)
        )
        self.register_buffer("beta", None if beta is None else torch.tensor(beta))
        self._log: bool = bool(getattr(acq_function, "_log", False)) and beta is None

        _validate_model(self.model, proximal_weights)
        self.register_buffer("proximal_weights", proximal_weights)
        if proximal_weights.ndim == 2:
            sym_weights = 0.5 * (proximal_weights + proximal_weights.mT)
            proximal_cholesky: Tensor | None = torch.linalg.cholesky(sym_weights)
        else:
            proximal_cholesky = None
        self.register_buffer("_proximal_cholesky", proximal_cholesky)

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate base acquisition function with proximal weighting.

        Args:
            X: Input tensor of shape ``batch_shape x 1 x d``.

        Returns:
            Base acquisition function evaluated on tensor ``X`` with proximal
            weighting applied (in log space if ``self._log`` is ``True``, or
            multiplied in linear space otherwise), of shape ``batch_shape``.
        """
        model = self.acq_func.model
        if isinstance(model, ModelListGP):
            model = model.models[0]

        train_inputs = _get_single_train_inputs(model)
        input_transform = _get_input_transform(model)
        last_X = train_inputs[-1].reshape(1, 1, -1)

        # if transformed_weighting, transform X to calculate diff
        # (proximal weighting in transformed space)
        # otherwise, un-transform the last observed point to real space
        # (proximal weighting in real space)
        if input_transform is not None:
            if self.transformed_weighting:
                diff = input_transform.transform(X) - last_X
            else:
                diff = X - input_transform.untransform(last_X)
        else:
            diff = X - last_X

        if self._proximal_cholesky is not None:
            L = self._proximal_cholesky.to(diff)
            M = (
                torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
                .squeeze(-1)
                .square()
                .sum(dim=-1)
            )
        else:
            M = (diff / self.proximal_weights.to(diff)).square().sum(dim=-1)

        base_acqf = self.acq_func(X)
        if self._log:
            return base_acqf - 0.5 * M.squeeze(-1)

        proximal_acq_weight = torch.exp(-0.5 * M)
        if self.beta is None:
            if torch.any(base_acqf < 0):
                raise RuntimeError(
                    "Cannot use proximal biasing for negative "
                    "acquisition function values, set a value for beta to "
                    "fix this with a softplus transform"
                )
        else:
            base_acqf = torch.nn.functional.softplus(base_acqf, beta=self.beta)

        return base_acqf * proximal_acq_weight.squeeze(-1)


def _validate_model(model: Model, proximal_weights: Tensor) -> None:
    r"""Validate model and proximal weights.

    Perform validation checks on model used in base acquisition function and on
    ``proximal_weights`` to make sure they are compatible with proximal
    weighting.

    Args:
        model: Model associated with base acquisition function to be validated.
        proximal_weights: Either a ``(d,)`` tensor of positive lengthscales or a
            ``(d, d)`` symmetric positive-definite matrix.
    """
    # check model for train_inputs and single batch
    if not hasattr(model, "train_inputs"):
        raise UnsupportedError("Acquisition function model must have `train_inputs`.")

    # get train inputs for each type of possible model
    if isinstance(model, ModelListGP):
        # ModelListGP models
        # check to make sure that the training inputs and input transformers for each
        # model match and are reversible
        train_inputs = _get_single_train_inputs(model.models[0])
        input_transform = _get_input_transform(model.models[0])

        for submodel in model.models:
            if not torch.equal(train_inputs, _get_single_train_inputs(submodel)):
                raise UnsupportedError(
                    "Proximal acquisition function does not support unequal "
                    "training inputs"
                )

            if not _input_transforms_equal(
                input_transform, _get_input_transform(submodel)
            ):
                raise UnsupportedError(
                    "Proximal acquisition function does not support non-identical "
                    "input transforms"
                )
    else:
        # any non-ModelListGP model
        train_inputs = _get_single_train_inputs(model)

    # check to make sure that the model is single t-batch (q-batches are allowed)
    if model.batch_shape != torch.Size([]) and train_inputs.shape[1] != 1:
        raise UnsupportedError(
            "Proximal acquisition function requires a single batch model"
        )

    if not proximal_weights.is_floating_point():
        raise ValueError("`proximal_weights` must be a floating-point tensor.")

    d = train_inputs.shape[-1]
    shape_err_msg = (
        "`proximal_weights` must be a one-dimensional tensor of shape `(d,)` "
        "or a two-dimensional tensor of shape `(d, d)`, where `d` is the "
        "feature dimension of the model."
    )
    if proximal_weights.ndim == 1:
        if proximal_weights.shape[0] != d:
            raise ValueError(shape_err_msg)
        if not torch.all(proximal_weights > 0):
            raise ValueError(
                "1D `proximal_weights` must contain strictly positive values."
            )
    elif proximal_weights.ndim == 2:
        if proximal_weights.shape != (d, d):
            raise ValueError(shape_err_msg)
        if not torch.isfinite(proximal_weights).all():
            raise ValueError(
                "2D `proximal_weights` must be a symmetric positive-definite matrix."
            )
        eps = torch.finfo(proximal_weights.dtype).eps
        max_abs = proximal_weights.abs().max().item()
        if not torch.allclose(
            proximal_weights, proximal_weights.mT, rtol=1e-5, atol=1e2 * eps * max_abs
        ):
            raise ValueError(
                "2D `proximal_weights` must be a symmetric positive-definite matrix."
            )
        sym_weights = 0.5 * (proximal_weights + proximal_weights.mT)
        _, info = torch.linalg.cholesky_ex(sym_weights)
        if info.item() != 0:
            raise ValueError(
                "2D `proximal_weights` must be a symmetric positive-definite matrix."
            )
    else:
        raise ValueError(shape_err_msg)


def _get_single_train_inputs(model: Model) -> Tensor:
    """Get training inputs for a single output of a model.

    Args:
        model: A BoTorch model with ``train_inputs``.

    Returns:
        Training input tensor of shape ``(batch_shape x) n x d``.
    """
    train_inputs = model.train_inputs[0]
    if isinstance(model, BatchedMultiOutputGPyTorchModel) and model.num_outputs > 1:
        return train_inputs[0]
    return train_inputs


def _get_input_transform(model: Model) -> InputTransform | None:
    """Get input transform if defined.

    Args:
        model: A BoTorch model.

    Returns:
        The model's ``InputTransform`` if present, otherwise ``None``.
    """
    try:
        return model.input_transform
    except AttributeError:
        return None


def _input_transforms_equal(
    tf1: InputTransform | None, tf2: InputTransform | None
) -> bool:
    """Check whether two optional input transforms are equivalent.

    Args:
        tf1: First optional ``InputTransform``.
        tf2: Second optional ``InputTransform``.

    Returns:
        ``True`` if both are ``None`` or ``tf1.equals(tf2)``, else ``False``.
    """
    if tf1 is None or tf2 is None:
        return tf1 is tf2
    return tf1.equals(tf2)
