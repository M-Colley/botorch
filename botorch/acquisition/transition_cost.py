#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Transition-cost models and cost-ratio acquisition wrappers for sequential and batch
Bayesian optimization with parameter switching or movement costs.

Cost-ratio acquisition weighting dates to Expected Improvement per unit cost
[Snoek2012practical]_ and heterogeneous cost-cooled Bayesian optimization
[Lee2020cost]_, while sequential state-transition penalties arise in movement-penalized
Bayesian optimization [Ramesh2022movement]_. For log-space acquisition functions (such
as ``LogExpectedImprovement`` and ``qLogNoisyExpectedHypervolumeImprovement``; see
[Ament2023logei]_), ``TransitionCostAcquisitionFunction`` evaluates the exact log-space
cost ratio ``alpha_log(X) - cost_exponent * log(c(X))``, preserving smooth gradients
without sign inversions on negative log-improvements.

References

.. [Snoek2012practical]
    J. Snoek, H. Larochelle, and R. P. Adams.
    Practical Bayesian Optimization of Machine Learning Algorithms.
    Advances in Neural Information Processing Systems 25, pp. 2951-2959, 2012.

.. [Lee2020cost]
    E. H. Lee, V. Perrone, C. Archambeau, and M. Seeger.
    Cost-aware Bayesian Optimization.
    ICML Workshop on Automated Machine Learning (AutoML), arXiv:2003.10870, 2020.

.. [Ramesh2022movement]
    S. S. Ramesh, P. G. Sessa, A. Krause, and I. Bogunovic.
    Movement Penalized Bayesian Optimization with Application to Wind Energy Systems.
    Advances in Neural Information Processing Systems 35, pp. 27036-27048, 2022.
"""

from __future__ import annotations

import math

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions.errors import UnsupportedError
from botorch.models.deterministic import DeterministicModel
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import InputTransform
from botorch.utils.transforms import is_ensemble, t_batch_mode_transform
from torch import Tensor


class TransitionCostModel(DeterministicModel):
    r"""Deterministic transition-cost model anchored at a previous operating point.

    Models sequential parameter switching / movement cost from a reference state
    ``x_prev`` across a ``q``-batch of candidate points ``X``:

    - Step ``0`` incurs cost ``fixed_cost + scale * d(X[..., 0, :], x_prev)``.
    - Step ``j >= 1`` incurs cost
      ``fixed_cost + scale * d(X[..., j, :], X[..., j-1, :])``.

    Returns a ``batch_shape x q x 1``-dim tensor of strictly positive per-step costs,
    making ``TransitionCostModel`` directly compatible with both
    ``TransitionCostAcquisitionFunction`` and ``InverseCostWeightedUtility`` (e.g., for
    ``qKnowledgeGradient`` and ``qHypervolumeKnowledgeGradient``).

    Example:
        >>> x_prev = torch.zeros(3)
        >>> cost_model = TransitionCostModel(x_prev=x_prev, fixed_cost=1.0, scale=0.5)
        >>> costs = cost_model(torch.rand(4, 2, 3))  # shape: 4 x 2 x 1
    """

    def __init__(
        self,
        x_prev: Tensor,
        fixed_cost: float = 1.0,
        scale: float = 1.0,
        p: float = 2.0,
        weights: Tensor | None = None,
        metric: Tensor | None = None,
    ) -> None:
        r"""Initialize the deterministic sequential transition-cost model.

        Args:
            x_prev: A ``(d,)`` or ``(1, d)``-dim tensor representing the previous or
                reference operating point from which transition distance is measured.
            fixed_cost: Strictly positive baseline evaluation cost ``c_0 > 0`` incurred
                by each candidate evaluation even at zero movement.
            scale: Non-negative multiplier scaling the transition distance ``d(x, y)``.
            p: Order of the ``L_p`` norm (``p >= 1`` or ``float("inf")``) used when
                ``metric`` is ``None``.
            weights: Optional ``(d,)``-dim tensor of strictly positive coordinate
                scales applied as ``|x_i - y_i| * weights[i]`` inside the ``L_p`` norm.
                Mutually exclusive with ``metric``.
            metric: Optional ``(d, d)``-dim symmetric positive-definite Mahalanobis
                transition matrix ``W`` defining
                ``d(x, y) = sqrt((x - y)^T W (x - y))``. Mutually exclusive with
                ``weights``.
        """
        if x_prev.ndim == 2 and x_prev.shape[0] == 1:
            x_prev = x_prev.squeeze(0)
        if x_prev.ndim != 1 or x_prev.numel() == 0:
            raise ValueError("`x_prev` must have shape `(d,)` or `(1, d)`.")
        if not torch.isfinite(x_prev).all():
            raise ValueError("`x_prev` must contain finite values.")

        if math.isnan(fixed_cost) or math.isinf(fixed_cost) or fixed_cost <= 0.0:
            raise ValueError("`fixed_cost` must be finite and strictly positive.")
        if math.isnan(scale) or math.isinf(scale) or scale < 0.0:
            raise ValueError("`scale` must be finite and non-negative.")
        if math.isnan(p) or p < 1.0:
            raise ValueError("`p` must be >= 1.0 or `float('inf')`.")
        if weights is not None and metric is not None:
            raise ValueError("Cannot specify both `weights` and `metric`.")

        d = x_prev.shape[0]
        if weights is not None:
            if weights.ndim != 1 or weights.shape[0] != d:
                raise ValueError(
                    f"`weights` must be a 1D tensor of shape `({d},)` matching "
                    "`x_prev`."
                )
            if not torch.isfinite(weights).all() or not torch.all(weights > 0.0):
                raise ValueError("`weights` must be finite and strictly positive.")

        metric_sym: Tensor | None = None
        metric_chol: Tensor | None = None
        if metric is not None:
            if metric.ndim != 2 or metric.shape != (d, d):
                raise ValueError(
                    f"`metric` must be a 2D tensor of shape `({d}, {d})` matching "
                    "`x_prev`."
                )
            if not torch.isfinite(metric).all():
                raise ValueError("`metric` must contain finite values.")
            if not torch.allclose(metric, metric.mT, atol=1e-6, rtol=1e-5):
                raise ValueError("`metric` must be symmetric positive-definite.")
            metric_sym = 0.5 * (metric + metric.mT)
            metric_chol, info = torch.linalg.cholesky_ex(metric_sym)
            if info.item() != 0:
                raise ValueError("`metric` must be symmetric positive-definite.")

        super().__init__()
        self.register_buffer("x_prev", x_prev.clone())
        self.register_buffer(
            "weights", weights.clone() if weights is not None else None
        )
        self.register_buffer(
            "metric", metric_sym.clone() if metric_sym is not None else None
        )
        self.register_buffer(
            "_metric_chol", metric_chol.clone() if metric_chol is not None else None
        )
        self.fixed_cost = float(fixed_cost)
        self.scale = float(scale)
        self.p = float(p)
        self._num_outputs = 1

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate sequential per-step transition costs on candidate set ``X``.

        Args:
            X: A ``batch_shape x q x d``-dim tensor of candidate points.

        Returns:
            A ``batch_shape x q x 1``-dim tensor of strictly positive step costs.
        """
        if X.ndim < 2:
            raise ValueError(
                "`X` must have at least 2 dimensions (`batch_shape x q x d`)."
            )
        d = self.x_prev.shape[-1]
        if X.shape[-1] != d:
            raise ValueError(
                f"Feature dimension of `X` ({X.shape[-1]}) must match `x_prev` ({d})."
            )
        if X.shape[-2] == 0:
            return torch.full(
                (*X.shape[:-1], 1),
                self.fixed_cost,
                dtype=X.dtype,
                device=X.device,
            )

        x_prev = self.x_prev.to(X)
        if X.shape[-2] == 1:
            diff = X - x_prev
        else:
            first_prev = x_prev.view(*([1] * (X.ndim - 2)), 1, d).expand(
                *X.shape[:-2], 1, d
            )
            prev_X = torch.cat([first_prev, X[..., :-1, :]], dim=-2)
            diff = X - prev_X

        # Double-where guard ensures exact 0 distance AND finite 0.0 gradient
        # when a candidate coincides with its predecessor (`diff == 0`).
        is_zero = (diff == 0).all(dim=-1)
        safe_diff = torch.where(is_zero.unsqueeze(-1), torch.ones_like(diff), diff)
        if self._metric_chol is not None:
            projected = safe_diff @ self._metric_chol.to(X)
            raw_dist = torch.linalg.vector_norm(projected, ord=2, dim=-1)
        else:
            scaled_diff = (
                safe_diff * self.weights.to(X)
                if self.weights is not None
                else safe_diff
            )
            raw_dist = torch.linalg.vector_norm(scaled_diff, ord=self.p, dim=-1)
        dist = torch.where(is_zero, torch.zeros_like(raw_dist), raw_dist)
        return (self.fixed_cost + self.scale * dist).unsqueeze(-1)


class TransitionCostAcquisitionFunction(AcquisitionFunction):
    r"""Cost-ratio acquisition wrapper accounting for transition or evaluation costs.

    Wraps any single-objective or multi-objective ``AcquisitionFunction`` (supporting
    ``q >= 1``, ``SingleTaskGP``, multi-output ``SingleTaskGP``, and ``ModelListGP``)
    and discounts utility by total batch cost ``c(X) ** cost_exponent``:

    - For log-space acquisition functions (``acq_function._log == True``, e.g.,
      ``LogExpectedImprovement``, ``qLogExpectedImprovement``,
      ``qLogExpectedHypervolumeImprovement``,
      ``qLogNoisyExpectedHypervolumeImprovement``, ``qLogNParEGO``):
      ``alpha_cost(X) = alpha_log(X) - cost_exponent * log(c(X))``.
    - For linear-space acquisition functions (``acq_function._log == False``):
      ``alpha(X) / (c(X) ** cost_exponent)`` when ``alpha(X) >= 0``, and
      ``alpha(X) * (c(X) ** cost_exponent)`` when ``alpha(X) < 0`` so negative utility
      values are penalized rather than rewarded by high cost.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> log_ei = LogExpectedImprovement(model, best_f=train_Y.max())
        >>> acqf = TransitionCostAcquisitionFunction(
        ...     acq_function=log_ei, fixed_cost=1.0, scale=0.5
        ... )
        >>> val = acqf(test_X)
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        cost_model: Model | None = None,
        x_prev: Tensor | None = None,
        fixed_cost: float = 1.0,
        scale: float = 1.0,
        p: float = 2.0,
        weights: Tensor | None = None,
        metric: Tensor | None = None,
        cost_exponent: float = 1.0,
        transformed_weighting: bool = False,
    ) -> None:
        r"""Initialize the transition-cost-modulated acquisition function.

        Args:
            acq_function: The base single-objective or multi-objective acquisition
                function to wrap.
            cost_model: Optional explicit ``Model`` (e.g., ``TransitionCostModel``,
                ``AffineFidelityCostModel``, or ``SingleTaskGP``) whose posterior mean
                sums across the ``q``-batch to strictly positive total cost ``c(X)``.
                If omitted, constructs a ``TransitionCostModel`` anchored at ``x_prev``.
            x_prev: Optional ``(d,)`` or ``(1, d)``-dim reference state in real
                (untransformed) input space. If omitted and ``cost_model`` is ``None``,
                inferred from the last observed training input ``train_inputs[-1]`` of
                ``acq_function.model``.
            fixed_cost: Baseline per-step evaluation cost ``c_0 > 0`` when constructing
                an internal ``TransitionCostModel``.
            scale: Non-negative distance multiplier when constructing an internal
                ``TransitionCostModel``.
            p: Order of the ``L_p`` norm when constructing an internal
                ``TransitionCostModel`` without ``metric``.
            weights: Optional ``(d,)``-dim positive coordinate weights for the internal
                ``TransitionCostModel``.
            metric: Optional ``(d, d)``-dim symmetric positive-definite Mahalanobis
                matrix for the internal ``TransitionCostModel``.
            cost_exponent: Non-negative exponent ``gamma >= 0`` controlling cost
                sensitivity via ``c(X) ** gamma``.
            transformed_weighting: If ``True`` (and ``cost_model`` is ``None``),
                computes transition distances in the model-transformed input space
                defined by ``acq_function.model.input_transform`` (when present); if
                ``False``, computes distances in real (untransformed) input space.
        """
        if (
            math.isnan(cost_exponent)
            or math.isinf(cost_exponent)
            or cost_exponent < 0.0
        ):
            raise ValueError("`cost_exponent` must be finite and non-negative.")

        super().__init__(model=getattr(acq_function, "model", None))
        self.acq_func: AcquisitionFunction = acq_function
        self._log: bool = bool(getattr(acq_function, "_log", False))
        self.cost_exponent = float(cost_exponent)
        self.transformed_weighting = bool(transformed_weighting)

        if cost_model is not None:
            if any(arg is not None for arg in (x_prev, weights, metric)):
                raise ValueError(
                    "Cannot specify `x_prev`, `weights`, or `metric` when "
                    "`cost_model` is provided."
                )
            if transformed_weighting:
                raise ValueError(
                    "`transformed_weighting=True` is not supported when an explicit "
                    "`cost_model` is provided."
                )
            self.cost_model: Model = cost_model
            self._input_transform: InputTransform | None = None
        else:
            resolved_x_prev, input_transform = _resolve_transition_reference(
                model=self.model,
                x_prev=x_prev,
                transformed_weighting=self.transformed_weighting,
            )
            self._input_transform = input_transform
            self.cost_model = TransitionCostModel(
                x_prev=resolved_x_prev,
                fixed_cost=fixed_cost,
                scale=scale,
                p=p,
                weights=weights,
                metric=metric,
            )

    @property
    def X_pending(self) -> Tensor | None:
        r"""Return pending design points from the wrapped acquisition function."""
        return getattr(self.acq_func, "X_pending", None)

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        r"""Inform the wrapped acquisition function about pending design points.

        Args:
            X_pending: ``n x d`` Tensor with ``n`` ``d``-dim design points that have
                been submitted for evaluation but have not yet been evaluated.
        """
        self.acq_func.set_X_pending(X_pending=X_pending)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the cost-modulated acquisition function on candidate set ``X``.

        Args:
            X: A ``batch_shape x q x d``-dim tensor of candidate points.

        Returns:
            A ``batch_shape``-dim tensor of cost-modulated acquisition values.
        """
        base_val = self.acq_func(X)
        X_for_cost = (
            self._input_transform.transform(X)
            if (self.transformed_weighting and self._input_transform is not None)
            else X
        )
        cost_mean = self.cost_model.posterior(X_for_cost).mean
        if is_ensemble(self.cost_model) and cost_mean.ndim > X_for_cost.ndim:
            cost_mean = cost_mean.mean(dim=-3)
        total_cost = cost_mean.sum(dim=(-2, -1))
        if not torch.all(total_cost > 0.0):
            raise ValueError("Total batch evaluation cost must be strictly positive.")

        if self._log:
            return base_val - self.cost_exponent * torch.log(total_cost)
        scaled_cost = total_cost.pow(self.cost_exponent)
        return torch.where(
            base_val >= 0.0, base_val / scaled_cost, base_val * scaled_cost
        )


def _get_input_transform(model: Model | None) -> InputTransform | None:
    r"""Extract ``model.input_transform`` if present, otherwise return ``None``."""
    if model is None:
        return None
    try:
        transform = model.input_transform
    except AttributeError:
        return None
    return transform if isinstance(transform, InputTransform) else None


def _extract_single_batch_train_X(model: Model) -> Tensor:
    r"""Validate single-batch training inputs on ``model`` and return ``(n, d)``."""
    if not hasattr(model, "train_inputs") or not model.train_inputs:
        raise UnsupportedError(
            "Acquisition function model must have `train_inputs` when `x_prev` is "
            "omitted."
        )
    batch_shape = getattr(model, "batch_shape", None)
    if batch_shape is not None and batch_shape.numel() > 1:
        raise UnsupportedError(
            "TransitionCostAcquisitionFunction requires a single-batch model when "
            "`x_prev` is inferred from `train_inputs`."
        )
    train_X = model.train_inputs[0]
    if isinstance(model, BatchedMultiOutputGPyTorchModel) and model.num_outputs > 1:
        train_X = train_X[..., 0, :, :]
    if train_X.ndim > 2 and math.prod(train_X.shape[:-2]) > 1:
        raise UnsupportedError(
            "TransitionCostAcquisitionFunction requires a single-batch model when "
            "`x_prev` is inferred from `train_inputs`."
        )
    train_X_2d = train_X.reshape(-1, train_X.shape[-1])
    if train_X_2d.shape[0] == 0:
        raise UnsupportedError(
            "Model `train_inputs` cannot be empty when `x_prev` is omitted."
        )
    return train_X_2d


def _validate_and_get_model_input_transform(
    model: Model | None,
) -> InputTransform | None:
    r"""Validate ``ModelListGP`` input-transform consistency via ``equals``."""
    if model is None:
        return None
    if isinstance(model, ModelListGP):
        if len(model.models) == 0:
            raise UnsupportedError("ModelListGP cannot be empty.")
        first_tf = _get_input_transform(model.models[0])
        for submodel in model.models[1:]:
            sub_tf = _get_input_transform(submodel)
            if (first_tf is None) != (sub_tf is None) or (
                first_tf is not None
                and sub_tf is not None
                and not first_tf.equals(sub_tf)
            ):
                raise UnsupportedError(
                    "ModelListGP submodels must have identical input transforms."
                )
        return first_tf
    return _get_input_transform(model)


def _resolve_transition_reference(
    model: Model | None,
    x_prev: Tensor | None,
    transformed_weighting: bool,
) -> tuple[Tensor, InputTransform | None]:
    r"""Resolve ``x_prev`` and optional ``input_transform`` for transition costs."""
    if x_prev is not None:
        if x_prev.ndim == 2 and x_prev.shape[0] == 1:
            x_prev_2d = x_prev
        elif x_prev.ndim == 1 and x_prev.numel() > 0:
            x_prev_2d = x_prev.unsqueeze(0)
        else:
            raise ValueError("`x_prev` must have shape `(d,)` or `(1, d)`.")
        if not transformed_weighting:
            return x_prev_2d.squeeze(0), None
        input_transform = _validate_and_get_model_input_transform(model)
        if input_transform is not None:
            with torch.no_grad():
                resolved = input_transform.transform(x_prev_2d).squeeze(0)
            return resolved, input_transform
        return x_prev_2d.squeeze(0), None

    if model is None or not hasattr(model, "train_inputs"):
        raise UnsupportedError(
            "Acquisition function model must have `train_inputs` when `x_prev` is "
            "omitted."
        )
    if hasattr(model, "eval"):
        model.eval()

    input_transform = _validate_and_get_model_input_transform(model)
    if isinstance(model, ModelListGP):
        ref_submodel = model.models[0]
        first_train_X = _extract_single_batch_train_X(ref_submodel)
        for submodel in model.models[1:]:
            sub_train_X = _extract_single_batch_train_X(submodel)
            if first_train_X.shape != sub_train_X.shape or not torch.allclose(
                first_train_X, sub_train_X
            ):
                raise UnsupportedError(
                    "ModelListGP submodels must have identical training inputs when "
                    "`x_prev` is omitted."
                )
    else:
        ref_submodel = model
        first_train_X = _extract_single_batch_train_X(ref_submodel)

    last_X = first_train_X[-1:].detach()
    is_stored_transformed = (
        input_transform is not None
        and getattr(ref_submodel, "_has_transformed_inputs", False)
        and input_transform.transform_on_train
    )
    if transformed_weighting:
        if input_transform is not None and not is_stored_transformed:
            with torch.no_grad():
                resolved_x_prev = input_transform.transform(last_X).squeeze(0)
        else:
            resolved_x_prev = last_X.squeeze(0)
        return resolved_x_prev, input_transform

    if input_transform is not None and is_stored_transformed:
        with torch.no_grad():
            resolved_x_prev = input_transform.untransform(last_X).squeeze(0)
    else:
        resolved_x_prev = last_X.squeeze(0)
    return resolved_x_prev, None
