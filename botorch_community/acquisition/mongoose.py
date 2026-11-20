#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
MONGOOSE: Meta-learning Of Non-myopic Global Optimisation fOr Smooth Exploration.

Implements the recurrent memory-based meta-learned Bayesian optimization policy
of [yang2023mongoose]_ for generating path-wise smooth optimization trajectories
under non-negligible movement or reconfiguration costs. Instead of myopically
maximizing a one-step acquisition function, MONGOOSE meta-trains an LSTM policy
directly on differentiable Gaussian Process prior sample paths via
Backpropagation Through Time (BPTT) to balance long-horizon global optimization
against cumulative step-to-step movement penalties.

References

.. [yang2023mongoose]
    A. X. Yang, L. Aitchison, and H. B. Moss.
    MONGOOSE: Path-wise Smooth Bayesian Optimisation via Meta-learning.
    arXiv preprint arXiv:2302.11533, 2023 (ICML 2024 Workshop on Structured
    Probabilistic Inference & Generative Modeling).

Contributor: M-Colley
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.sampling.pathwise import draw_kernel_feature_paths
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.models import ExactGP, GP
from torch import nn, Size, Tensor
from torch.distributions import Dirichlet, Gamma


SUPPORTED_RFF_KERNELS = ("rbf", "matern32", "matern_32", "matern52", "matern_52")
SUPPORTED_SCALARIZATIONS = ("linear", "chebyshev")


class _KernelPriorGP(ExactGP, GPyTorchModel):
    r"""Lightweight zero-observation ExactGP wrapper for pathwise prior sampling."""

    _num_outputs = 1

    def __init__(
        self,
        input_dim: int,
        covar_module: Kernel,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        r"""Initialize the zero-data ExactGP prior wrapper.

        Args:
            input_dim: Dimensionality ``d`` of the input domain.
            covar_module: GPyTorch covariance kernel defining the GP prior.
            device: Torch device for empty training placeholders.
            dtype: Torch dtype for empty training placeholders.
        """
        empty_x = torch.empty(0, input_dim, device=device, dtype=dtype)
        empty_y = torch.empty(0, device=device, dtype=dtype)
        likelihood = GaussianLikelihood().to(device=device, dtype=dtype)
        super().__init__(empty_x, empty_y, likelihood)
        self.mean_module = ZeroMean().to(device=device, dtype=dtype)
        self.covar_module = covar_module


def _validate_bounds(bounds: Tensor, input_dim: int) -> None:
    r"""Validate a ``2 x d`` domain bounds tensor.

    Args:
        bounds: Tensor of lower and upper bounds with expected shape ``(2, d)``.
        input_dim: Expected input dimension ``d``.
    """
    if bounds.ndim != 2 or bounds.shape != (2, input_dim):
        raise ValueError(
            f"Expected `bounds` of shape (2, {input_dim}), got {tuple(bounds.shape)}."
        )
    if not torch.all(bounds[1] > bounds[0]):
        raise ValueError(
            "`bounds[1]` must be strictly greater than `bounds[0]` in all dimensions."
        )


def _build_mlp(
    in_dim: int,
    hidden_sizes: Sequence[int],
    out_dim: int | None,
    activation: type[nn.Module],
) -> nn.Sequential:
    r"""Construct a feed-forward multilayer perceptron.

    Args:
        in_dim: Input feature dimension.
        hidden_sizes: Sequence of hidden layer widths.
        out_dim: Optional final linear output dimension without trailing activation.
        activation: Activation ``nn.Module`` class applied after each hidden layer.

    Returns:
        An ``nn.Sequential`` module.
    """
    layers: list[nn.Module] = []
    curr_dim = in_dim
    for width in hidden_sizes:
        if width < 1:
            raise ValueError(f"Hidden layer sizes must be >= 1, got {width}.")
        layers.append(nn.Linear(curr_dim, width))
        layers.append(activation())
        curr_dim = width
    if out_dim is not None:
        layers.append(nn.Linear(curr_dim, out_dim))
    return nn.Sequential(*layers)


def _normalize_preference_weights(
    weights: Tensor,
    batch_size: int,
    num_objectives: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    r"""Validate and normalize preference weights onto the probability simplex.

    Args:
        weights: Preference tensor of shape ``(m,)`` or ``(batch_size, m)``.
        batch_size: Batch size ``B``.
        num_objectives: Number of objectives ``m``.
        device: Target torch device.
        dtype: Target torch dtype.

    Returns:
        Normalized simplex weights of shape ``(batch_size, m)``.
    """
    w = weights.to(device=device, dtype=dtype)
    if w.ndim == 1:
        if w.shape[0] != num_objectives:
            raise ValueError(
                f"Expected `weights` of length {num_objectives}, got {w.shape[0]}."
            )
        w = w.unsqueeze(0).expand(batch_size, num_objectives)
    elif w.ndim == 2:
        if w.shape != (batch_size, num_objectives):
            raise ValueError(
                f"Expected `weights` of shape ({batch_size}, {num_objectives}), "
                f"got {tuple(w.shape)}."
            )
    else:
        raise ValueError(f"`weights` must be 1D or 2D, got {w.ndim}D.")
    if torch.any(w < 0):
        raise ValueError("`weights` must be non-negative in all entries.")
    w_sum = w.sum(dim=-1, keepdim=True)
    if torch.any(w_sum <= 0):
        raise ValueError("`weights` must have strictly positive sum along dim=-1.")
    return w / w_sum


class MongoosePolicy(nn.Module):
    r"""Recurrent LSTM policy for path-wise smooth Bayesian optimization.

    Following [yang2023mongoose]_, ``MongoosePolicy`` processes sequential
    normalized query-observation-horizon tuples ``(x_norm_t, y_t, tau_t)``
    (optionally augmented with movement-cost weight context ``log(1 + alpha)``
    and/or multi-objective simplex preference weights ``w``) through an input
    encoder MLP, one or more stacked ``nn.LSTMCell`` layers, and a decoder MLP
    with ``sigmoid`` output mapped to ``bounds``.
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tensor | None = None,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_objectives: int = 1,
        encoder_hidden_sizes: Sequence[int] | None = None,
        decoder_hidden_sizes: Sequence[int] | None = None,
        condition_on_alpha: bool = False,
        condition_on_preferences: bool | None = None,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        r"""Initialize ``MongoosePolicy``.

        Args:
            input_dim: Dimensionality ``d`` of the search space.
            bounds: Optional ``2 x d`` tensor of lower and upper bounds. Defaults
                to the unit hypercube ``[0, 1]^d``.
            hidden_size: Hidden and cell state dimension of each ``LSTMCell``
                (128 in [yang2023mongoose]_).
            num_layers: Number of stacked recurrent ``LSTMCell`` layers (``>= 1``).
            num_objectives: Number of observed objectives ``m`` (``>= 1``).
            encoder_hidden_sizes: Optional hidden layer sizes for the input encoder
                MLP. Defaults to ``(hidden_size,)``.
            decoder_hidden_sizes: Optional hidden layer sizes for the decoder MLP
                mapping the top LSTM hidden state to ``d`` logits. Defaults to
                ``(hidden_size,)``.
            condition_on_alpha: If ``True``, conditions the policy on the movement
                penalty weight ``alpha`` via ``log(1 + alpha)`` so a single policy
                can adapt trajectory smoothness at deployment time.
            condition_on_preferences: If ``True``, conditions the policy on
                scalarization weights ``w`` in the ``(m - 1)``-simplex. Defaults to
                ``True`` when ``num_objectives > 1`` and ``False`` otherwise.
            activation: Activation module class used in encoder and decoder MLPs.
        """
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"`input_dim` must be >= 1, got {input_dim}.")
        if hidden_size < 1:
            raise ValueError(f"`hidden_size` must be >= 1, got {hidden_size}.")
        if num_layers < 1:
            raise ValueError(f"`num_layers` must be >= 1, got {num_layers}.")
        if num_objectives < 1:
            raise ValueError(f"`num_objectives` must be >= 1, got {num_objectives}.")

        if bounds is None:
            default_bounds = torch.zeros(2, input_dim, dtype=torch.get_default_dtype())
            default_bounds[1] = 1.0
            bounds_buf = default_bounds
        else:
            _validate_bounds(bounds=bounds, input_dim=input_dim)
            bounds_buf = bounds.detach().clone()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_objectives = num_objectives
        self.condition_on_alpha = condition_on_alpha
        self.condition_on_preferences = (
            num_objectives > 1
            if condition_on_preferences is None
            else condition_on_preferences
        )
        self.register_buffer("bounds", bounds_buf)

        enc_sizes = (
            (hidden_size,)
            if encoder_hidden_sizes is None
            else tuple(encoder_hidden_sizes)
        )
        dec_sizes = (
            (hidden_size,)
            if decoder_hidden_sizes is None
            else tuple(decoder_hidden_sizes)
        )

        feature_dim = (
            input_dim
            + num_objectives
            + 1  # normalized horizon progress tau_t
            + (1 if self.condition_on_alpha else 0)
            + (num_objectives if self.condition_on_preferences else 0)
        )

        self.encoder = _build_mlp(
            in_dim=feature_dim,
            hidden_sizes=enc_sizes,
            out_dim=None,
            activation=activation,
        )
        lstm_in_dim = enc_sizes[-1] if len(enc_sizes) > 0 else feature_dim

        self.lstm_cells = nn.ModuleList(
            [
                nn.LSTMCell(lstm_in_dim if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ]
        )
        self.decoder = _build_mlp(
            in_dim=hidden_size,
            hidden_sizes=dec_sizes,
            out_dim=input_dim,
            activation=activation,
        )

        # Ensure module parameters match explicit bounds dtype/device if provided.
        if bounds is not None:
            self.to(device=bounds_buf.device, dtype=bounds_buf.dtype)

    def init_hidden(
        self,
        batch_size: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        r"""Create zero-initialized ``(h, c)`` states for all LSTM layers.

        Args:
            batch_size: Number of parallel trajectories ``B`` (``>= 1``).
            device: Optional target device. Defaults to the policy's device.
            dtype: Optional target dtype. Defaults to the policy's parameter dtype.

        Returns:
            List of ``num_layers`` ``(h_l, c_l)`` tuples, each of shape
            ``(batch_size, hidden_size)``.
        """
        if batch_size < 1:
            raise ValueError(f"`batch_size` must be >= 1, got {batch_size}.")
        param = next(self.parameters())
        dev = param.device if device is None else device
        dt = param.dtype if dtype is None else dtype
        return [
            (
                torch.zeros(batch_size, self.hidden_size, device=dev, dtype=dt),
                torch.zeros(batch_size, self.hidden_size, device=dev, dtype=dt),
            )
            for _ in range(self.num_layers)
        ]

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        tau: float | Tensor,
        state: list[tuple[Tensor, Tensor]] | None = None,
        alpha: float | Tensor | None = None,
        weights: Tensor | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        r"""Advance the recurrent policy by one step and output the next query.

        Args:
            x: Current query point(s) in ``bounds`` space of shape ``(d,)`` or
                ``(batch_size, d)``.
            y: Observed function value(s) at ``x`` of shape ``(m,)`` / ``()``
                (unbatched) or ``(batch_size, m)`` / ``(batch_size,)`` (batched).
            tau: Normalized horizon progress ``t / T`` in ``[0, 1]``, passed as a
                float or tensor broadcastable to ``(batch_size, 1)``.
            state: Optional list of ``num_layers`` ``(h_l, c_l)`` tuples from the
                previous step. If ``None``, initialized to zeros.
            alpha: Optional movement-penalty weight(s) ``>= 0`` required when
                ``condition_on_alpha=True``.
            weights: Optional multi-objective simplex weights of shape ``(m,)`` or
                ``(batch_size, m)``. Defaults to uniform weights when
                ``condition_on_preferences=True`` and ``weights is None``.

        Returns:
            A tuple ``(x_next, next_state)`` where ``x_next`` lies strictly inside
            ``bounds`` with shape ``(d,)`` (if ``x`` was 1D) or ``(batch_size, d)``
            (if ``x`` was 2D), and ``next_state`` is the updated LSTM state list.
        """
        if x.ndim == 1:
            is_unbatched = True
            x_b = x.unsqueeze(0)
        elif x.ndim == 2:
            is_unbatched = False
            x_b = x
        else:
            raise ValueError(f"`x` must be 1D or 2D, got {x.ndim}D.")

        batch_size, d = x_b.shape
        if d != self.input_dim:
            raise ValueError(f"Expected `x` last dim {self.input_dim}, got {d}.")

        device, dtype = x_b.device, x_b.dtype
        y_b = self._format_observations(
            y=y, batch_size=batch_size, is_unbatched=is_unbatched
        )
        tau_b = self._format_scalar_context(
            val=tau, name="tau", batch_size=batch_size, device=device, dtype=dtype
        )

        if state is None:
            state = self.init_hidden(batch_size=batch_size, device=device, dtype=dtype)
        else:
            self._validate_state(state=state, batch_size=batch_size)

        bounds = self.bounds.to(device=device, dtype=dtype)
        x_norm = normalize(x_b, bounds=bounds)

        features = [x_norm, y_b, tau_b]
        if self.condition_on_alpha:
            if alpha is None:
                raise ValueError(
                    "`alpha` must be provided when `condition_on_alpha=True`."
                )
            alpha_b = self._format_scalar_context(
                val=alpha,
                name="alpha",
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            if torch.any(alpha_b < 0):
                raise ValueError("`alpha` must be non-negative.")
            features.append(torch.log1p(alpha_b))

        if self.condition_on_preferences:
            if weights is None:
                w_b = torch.full(
                    (batch_size, self.num_objectives),
                    1.0 / self.num_objectives,
                    device=device,
                    dtype=dtype,
                )
            else:
                w_b = _normalize_preference_weights(
                    weights=weights,
                    batch_size=batch_size,
                    num_objectives=self.num_objectives,
                    device=device,
                    dtype=dtype,
                )
            features.append(w_b)

        step_in = torch.cat(features, dim=-1)
        h_in = self.encoder(step_in)

        next_state: list[tuple[Tensor, Tensor]] = []
        for layer_idx, cell in enumerate(self.lstm_cells):
            h_prev, c_prev = state[layer_idx]
            h_next, c_next = cell(h_in, (h_prev, c_prev))
            next_state.append((h_next, c_next))
            h_in = h_next

        u_next = torch.sigmoid(self.decoder(h_in))
        x_next = unnormalize(u_next, bounds=bounds)
        if is_unbatched:
            x_next = x_next.squeeze(0)
        return x_next, next_state

    def _format_observations(
        self, y: Tensor, batch_size: int, is_unbatched: bool
    ) -> Tensor:
        r"""Reshape and validate observation tensor ``y`` to ``(batch_size, m)``."""
        if is_unbatched:
            if y.ndim == 0 and self.num_objectives == 1:
                y_b = y.reshape(1, 1)
            elif y.ndim == 1 and y.shape[0] == self.num_objectives:
                y_b = y.unsqueeze(0)
            elif y.ndim == 2 and y.shape == (1, self.num_objectives):
                y_b = y
            else:
                raise ValueError(
                    f"Incompatible unbatched `y` shape {tuple(y.shape)} for "
                    f"`num_objectives={self.num_objectives}`."
                )
        else:
            if y.ndim == 1 and self.num_objectives == 1 and y.shape[0] == batch_size:
                y_b = y.unsqueeze(-1)
            elif y.ndim == 2 and y.shape == (batch_size, self.num_objectives):
                y_b = y
            else:
                raise ValueError(
                    f"Expected batched `y` shape ({batch_size}, {self.num_objectives})"
                    f", got {tuple(y.shape)}."
                )
        return y_b

    @staticmethod
    def _format_scalar_context(
        val: float | Tensor,
        name: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        r"""Convert scalar or per-batch context into shape ``(batch_size, 1)``."""
        if isinstance(val, (int, float)):
            return torch.full((batch_size, 1), float(val), device=device, dtype=dtype)
        t = val.to(device=device, dtype=dtype)
        if t.ndim == 0:
            return t.reshape(1, 1).expand(batch_size, 1)
        if t.ndim == 1:
            if t.shape[0] == 1:
                return t.unsqueeze(-1).expand(batch_size, 1)
            if t.shape[0] == batch_size:
                return t.unsqueeze(-1)
        if t.ndim == 2 and t.shape in ((1, 1), (batch_size, 1)):
            return t.expand(batch_size, 1)
        raise ValueError(
            f"`{name}` of shape {tuple(t.shape)} cannot broadcast to ({batch_size}, 1)."
        )

    def _validate_state(
        self, state: list[tuple[Tensor, Tensor]], batch_size: int
    ) -> None:
        r"""Validate recurrent hidden state list dimensions."""
        if len(state) != self.num_layers:
            raise ValueError(
                f"Expected `state` of length {self.num_layers}, got {len(state)}."
            )
        expected_shape = (batch_size, self.hidden_size)
        for idx, (h, c) in enumerate(state):
            if h.shape != expected_shape or c.shape != expected_shape:
                raise ValueError(
                    f"Layer {idx} state must have shape {expected_shape}, got "
                    f"{tuple(h.shape)} and {tuple(c.shape)}."
                )


class GPPriorPathBatch(nn.Module):
    r"""Differentiable batch of synthetic functions sampled from a GP prior."""

    def __init__(
        self,
        input_dim: int,
        batch_size: int,
        num_objectives: int,
        bounds: Tensor | None = None,
        omega: Tensor | None = None,
        weights: Tensor | None = None,
        outputscale: Tensor | None = None,
        pathwise_sampler: nn.Module | None = None,
        inducing_x: Tensor | None = None,
        nystrom_alpha: Tensor | None = None,
        covar_module: Kernel | None = None,
    ) -> None:
        r"""Initialize ``GPPriorPathBatch``.

        Args:
            input_dim: Input dimension ``d``.
            batch_size: Number of sampled functions ``B``.
            num_objectives: Number of output objectives ``m``.
            bounds: Optional ``2 x d`` bounds tensor used to normalize inputs to
                ``[0, 1]^d`` prior to kernel feature evaluation.
            omega: Optional RFF frequency tensor of shape ``(B, m, M // 2, d)``
                pre-scaled by inverse lengthscales.
            weights: Optional linear RFF weight tensor of shape ``(B, m, M)``.
            outputscale: Optional variance scale tensor of shape ``(B, m)``.
            pathwise_sampler: Optional BoTorch ``SamplePath`` produced by
                ``draw_kernel_feature_paths``.
            inducing_x: Optional Nystrom inducing locations of shape ``(N, d)``.
            nystrom_alpha: Optional Nystrom dual weights ``L^{-T} z`` of shape
                ``(B, m, N)``.
            covar_module: Optional GPyTorch kernel used with Nystrom fallback.
        """
        super().__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_objectives = num_objectives
        self.pathwise_sampler = pathwise_sampler
        self.covar_module = covar_module

        if bounds is not None:
            self.register_buffer("bounds", bounds.detach().clone())
        else:
            self.bounds = None
        if omega is not None:
            self.register_buffer("omega", omega)
        else:
            self.omega = None
        if weights is not None:
            self.register_buffer("weights", weights)
        else:
            self.weights = None
        if outputscale is not None:
            self.register_buffer("outputscale", outputscale)
        else:
            self.outputscale = None
        if inducing_x is not None:
            self.register_buffer("inducing_x", inducing_x)
        else:
            self.inducing_x = None
        if nystrom_alpha is not None:
            self.register_buffer("nystrom_alpha", nystrom_alpha)
        else:
            self.nystrom_alpha = None

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the sampled GP prior functions differentiably at ``X``.

        Args:
            X: Query tensor of shape ``(batch_size, d)`` for per-function single-point
                evaluation or ``(batch_size, q, d)`` / ``(1, q, d)`` for multi-point
                evaluation.

        Returns:
            Function values of shape ``(batch_size, m)`` when ``X`` is 2D, or
            ``(batch_size, q, m)`` when ``X`` is 3D.
        """
        if X.ndim == 2:
            if X.shape != (self.batch_size, self.input_dim):
                raise ValueError(
                    f"Expected 2D `X` of shape ({self.batch_size}, {self.input_dim}), "
                    f"got {tuple(X.shape)}."
                )
            squeeze_q = True
            X_q = X.unsqueeze(1)
        elif X.ndim == 3:
            if X.shape[-1] != self.input_dim or X.shape[0] not in (1, self.batch_size):
                raise ValueError(
                    f"Expected 3D `X` of shape ({self.batch_size} or 1, q, "
                    f"{self.input_dim}), got {tuple(X.shape)}."
                )
            squeeze_q = False
            X_q = X.expand(self.batch_size, X.shape[1], self.input_dim)
        else:
            raise ValueError(f"`X` must be 2D or 3D, got {X.ndim}D.")

        if self.bounds is not None:
            bounds = self.bounds.to(device=X_q.device, dtype=X_q.dtype)
            X_eval = normalize(X_q, bounds=bounds)
        else:
            X_eval = X_q

        if self.omega is not None and self.weights is not None:
            # X_eval: (B, q, d) -> (B, 1, q, d); omega: (B, m, M/2, d)
            proj = torch.matmul(X_eval.unsqueeze(1), self.omega.transpose(-2, -1))
            num_features = self.weights.shape[-1]
            scale = math.sqrt(2.0 / num_features)
            phi = scale * torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
            # phi: (B, m, q, M), weights: (B, m, M) -> raw: (B, m, q)
            raw = torch.matmul(phi, self.weights.unsqueeze(-1)).squeeze(-1)
            if self.outputscale is not None:
                raw = raw * torch.sqrt(self.outputscale).unsqueeze(-1)
            out = raw.transpose(-2, -1)  # (B, q, m)
        elif self.pathwise_sampler is not None:
            # GeneralizedLinearPath weight has shape (B, m, M); pass (B, 1, q, d)
            raw = self.pathwise_sampler(X_eval.unsqueeze(1))  # (B, m, q)
            out = raw.transpose(-2, -1)  # (B, q, m)
        elif (
            self.covar_module is not None
            and self.inducing_x is not None
            and self.nystrom_alpha is not None
        ):
            # Nystrom kernel path for general custom kernels: k(X, Z) @ L^{-T} z
            k_xz = self.covar_module(X_eval, self.inducing_x).to_dense()  # (B, q, N)
            # nystrom_alpha: (B, m, N) -> out: (B, q, m)
            out = torch.matmul(k_xz, self.nystrom_alpha.transpose(-2, -1))
        else:  # pragma: no cover
            raise RuntimeError("Invalid `GPPriorPathBatch` configuration.")

        return out.squeeze(1) if squeeze_q else out


def _resolve_positive_param(
    param: float | Tensor | None,
    param_range: tuple[float, float],
    shape: tuple[int, ...],
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    r"""Resolve a fixed positive hyperparameter or sample uniformly in a range."""
    if param is not None:
        if isinstance(param, (int, float)):
            if param <= 0:
                raise ValueError(f"`{name}` must be > 0, got {param}.")
            return torch.full(shape, float(param), device=device, dtype=dtype)
        t = param.to(device=device, dtype=dtype)
        if torch.any(t <= 0):
            raise ValueError(f"`{name}` must be strictly positive in all entries.")
        try:
            return t.expand(shape).clone()
        except RuntimeError as exc:
            raise ValueError(
                f"`{name}` of shape {tuple(t.shape)} cannot broadcast to {shape}."
            ) from exc

    low, high = param_range
    if low <= 0 or high < low:
        raise ValueError(
            f"`{name}_range` must satisfy 0 < low <= high, got {param_range}."
        )
    if low == high:
        return torch.full(shape, float(low), device=device, dtype=dtype)
    u = torch.rand(shape, device=device, dtype=dtype)
    return low + (high - low) * u


def sample_gp_prior_paths(
    input_dim: int,
    batch_size: int = 1,
    num_objectives: int = 1,
    kernel: str | Kernel | GP = "matern52",
    num_rff_features: int = 256,
    lengthscale: float | Tensor | None = None,
    lengthscale_range: tuple[float, float] = (0.15, 0.5),
    outputscale: float | Tensor | None = None,
    outputscale_range: tuple[float, float] = (1.0, 1.0),
    bounds: Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> GPPriorPathBatch:
    r"""Sample differentiable synthetic functions from a Gaussian Process prior.

    Supports fast vectorized Bochner Random Fourier Features (RFF) with per-episode
    sampled or fixed lengthscales and outputscales for stationary kernels (``"rbf"``,
    ``"matern32"``, ``"matern52"``), as well as arbitrary user-supplied GPyTorch
    ``Kernel`` or ``GP`` priors via ``draw_kernel_feature_paths`` from
    ``botorch.sampling.pathwise`` (falling back to exact Nystrom inducing-point
    paths for non-RFF kernels).

    Args:
        input_dim: Input dimension ``d`` (``>= 1``).
        batch_size: Number of independent function draws ``B`` (``>= 1``).
        num_objectives: Number of independent output dimensions ``m`` (``>= 1``).
        kernel: Kernel specification: one of ``"rbf"``, ``"matern32"``, ``"matern52"``,
            a GPyTorch ``Kernel`` instance, or a GPyTorch ``GP`` model.
        num_rff_features: Even number of random Fourier features ``M >= 2``.
        lengthscale: Optional fixed lengthscale(s) ``> 0`` (scalar, ``(d,)``, or
            broadcastable to ``(batch_size, num_objectives, d)``) used when ``kernel``
            is a string name.
        lengthscale_range: ``(min_ls, max_ls)`` uniform range used when ``lengthscale``
            is ``None`` and ``kernel`` is a string name.
        outputscale: Optional fixed kernel variance ``> 0`` used when ``kernel`` is a
            string name.
        outputscale_range: ``(min_os, max_os)`` uniform range used when ``outputscale``
            is ``None`` and ``kernel`` is a string name.
        bounds: Optional ``2 x d`` domain bounds. When provided, inputs to the
            returned ``GPPriorPathBatch`` are normalized to ``[0, 1]^d``.
        device: Torch device for sampled basis parameters.
        dtype: Torch dtype for sampled basis parameters.

    Returns:
        A ``GPPriorPathBatch`` callable module mapping ``X`` to differentiable GP
        function evaluations.
    """
    if input_dim < 1:
        raise ValueError(f"`input_dim` must be >= 1, got {input_dim}.")
    if batch_size < 1:
        raise ValueError(f"`batch_size` must be >= 1, got {batch_size}.")
    if num_objectives < 1:
        raise ValueError(f"`num_objectives` must be >= 1, got {num_objectives}.")
    if num_rff_features < 2 or num_rff_features % 2 != 0:
        raise ValueError(
            f"`num_rff_features` must be an even integer >= 2, got {num_rff_features}."
        )

    dev = torch.device("cpu") if device is None else device
    dt = torch.get_default_dtype() if dtype is None else dtype
    if bounds is not None:
        _validate_bounds(bounds=bounds, input_dim=input_dim)
        if device is None:
            dev = bounds.device
        if dtype is None:
            dt = bounds.dtype

    if isinstance(kernel, str):
        kernel_key = kernel.lower()
        if kernel_key not in SUPPORTED_RFF_KERNELS:
            raise ValueError(
                f"Unsupported kernel string {kernel!r}. Expected one of "
                f"{SUPPORTED_RFF_KERNELS} or a GPyTorch `Kernel` / `GP`."
            )
        ls = _resolve_positive_param(
            param=lengthscale,
            param_range=lengthscale_range,
            shape=(batch_size, num_objectives, input_dim),
            name="lengthscale",
            device=dev,
            dtype=dt,
        )
        os = _resolve_positive_param(
            param=outputscale,
            param_range=outputscale_range,
            shape=(batch_size, num_objectives),
            name="outputscale",
            device=dev,
            dtype=dt,
        )
        half_m = num_rff_features // 2
        normals = torch.randn(
            batch_size, num_objectives, half_m, input_dim, device=dev, dtype=dt
        )
        if kernel_key == "rbf":
            raw_omega = normals
        else:
            nu_val = 1.5 if kernel_key in ("matern32", "matern_32") else 2.5
            nu_t = torch.tensor(nu_val, device=dev, dtype=dt)
            gamma_draws = Gamma(nu_t, nu_t).rsample(
                (batch_size, num_objectives, half_m, 1)
            )
            raw_omega = normals * torch.rsqrt(gamma_draws)

        omega = raw_omega / ls.unsqueeze(-2)
        weights = torch.randn(
            batch_size, num_objectives, num_rff_features, device=dev, dtype=dt
        )
        return GPPriorPathBatch(
            input_dim=input_dim,
            batch_size=batch_size,
            num_objectives=num_objectives,
            bounds=bounds,
            omega=omega,
            weights=weights,
            outputscale=os,
        )

    if isinstance(kernel, GP):
        gp_model = copy.deepcopy(kernel).to(device=dev, dtype=dt)
        if not isinstance(gp_model, Model):
            gp_model = _KernelPriorGP(
                input_dim=input_dim,
                covar_module=gp_model.covar_module,
                device=dev,
                dtype=dt,
            )
        sample_paths = draw_kernel_feature_paths(
            model=gp_model,
            sample_shape=Size([batch_size, num_objectives]),
            num_features=num_rff_features,
        )
        return GPPriorPathBatch(
            input_dim=input_dim,
            batch_size=batch_size,
            num_objectives=num_objectives,
            bounds=bounds,
            pathwise_sampler=sample_paths,
        )

    if isinstance(kernel, Kernel):
        covar = copy.deepcopy(kernel).to(device=dev, dtype=dt)
        prior_gp = _KernelPriorGP(
            input_dim=input_dim, covar_module=covar, device=dev, dtype=dt
        )
        try:
            sample_paths = draw_kernel_feature_paths(
                model=prior_gp,
                sample_shape=Size([batch_size, num_objectives]),
                num_features=num_rff_features,
            )
            return GPPriorPathBatch(
                input_dim=input_dim,
                batch_size=batch_size,
                num_objectives=num_objectives,
                bounds=bounds,
                pathwise_sampler=sample_paths,
            )
        except (UnsupportedError, NotImplementedError):
            # Fallback for non-stationary / custom kernels without RFF dispatchers.
            num_inducing = min(num_rff_features, 128)
            unit_bounds = torch.zeros(2, input_dim, device=dev, dtype=dt)
            unit_bounds[1] = 1.0
            inducing_x = draw_sobol_samples(
                bounds=unit_bounds, n=num_inducing, q=1
            ).squeeze(-2)
            k_zz = covar(inducing_x, inducing_x).to_dense()
            eye = torch.eye(num_inducing, device=dev, dtype=dt)
            jitter = 1e-6 if dt == torch.float64 else 1e-4
            chol = torch.linalg.cholesky(k_zz + jitter * eye)
            z = torch.randn(
                batch_size, num_objectives, num_inducing, 1, device=dev, dtype=dt
            )
            # L^{-T} z solves L^T alpha = z
            alpha = torch.linalg.solve_triangular(
                chol.transpose(-2, -1), z, upper=True
            ).squeeze(-1)
            return GPPriorPathBatch(
                input_dim=input_dim,
                batch_size=batch_size,
                num_objectives=num_objectives,
                bounds=bounds,
                inducing_x=inducing_x,
                nystrom_alpha=alpha,
                covar_module=covar,
            )

    raise TypeError(
        "`kernel` must be a kernel name string, GPyTorch `Kernel`, or GPyTorch `GP`."
    )


def compute_movement_cost(
    x_next: Tensor,
    x_curr: Tensor,
    movement_weights: Tensor | None = None,
    movement_metric: Tensor | None = None,
    cost_exponent: float = 1.0,
    smooth_eps: float = 1e-8,
) -> Tensor:
    r"""Compute differentiable step-to-step transition cost ``c(x_{t+1}, x_t)``.

    Evaluates ``(\|x_{t+1} - x_t\|_W^2 + \epsilon)^{p / 2}`` (or exact
    ``\|x_{t+1} - x_t\|_W^2`` when ``cost_exponent == 2.0``), where ``W`` is the
    identity, ``diag(movement_weights)``, or a symmetric positive definite matrix
    ``movement_metric``.

    Args:
        x_next: Next query tensor of shape ``(..., d)``.
        x_curr: Current query tensor of shape ``(..., d)``.
        movement_weights: Optional non-negative per-dimension weights of shape
            ``(d,)``. Mutually exclusive with ``movement_metric``.
        movement_metric: Optional ``d x d`` symmetric positive definite metric
            tensor ``W``. Mutually exclusive with ``movement_weights``.
        cost_exponent: Positive exponent ``p`` applied to the smooth Euclidean
            distance (typically ``1.0`` for Euclidean travel distance or ``2.0``
            for quadratic movement cost).
        smooth_eps: Positive smoothing constant ``\epsilon`` ensuring finite
            gradients at zero displacement when ``cost_exponent != 2.0``.

    Returns:
        Transition cost tensor of shape ``x_next.shape[:-1]``.
    """
    if x_next.shape != x_curr.shape:
        raise ValueError(
            f"`x_next` shape {tuple(x_next.shape)} must match `x_curr` shape "
            f"{tuple(x_curr.shape)}."
        )
    if movement_weights is not None and movement_metric is not None:
        raise ValueError(
            "Specify at most one of `movement_weights` or `movement_metric`."
        )
    if cost_exponent <= 0:
        raise ValueError(f"`cost_exponent` must be > 0, got {cost_exponent}.")
    if smooth_eps <= 0:
        raise ValueError(f"`smooth_eps` must be > 0, got {smooth_eps}.")

    diff = x_next - x_curr
    d = diff.shape[-1]
    device, dtype = diff.device, diff.dtype

    if movement_weights is not None:
        w = movement_weights.to(device=device, dtype=dtype)
        if w.ndim != 1 or w.shape[0] != d:
            raise ValueError(
                f"Expected `movement_weights` of shape ({d},), got {tuple(w.shape)}."
            )
        if torch.any(w < 0):
            raise ValueError("`movement_weights` must be non-negative.")
        sq_dist = (diff.pow(2) * w).sum(dim=-1)
    elif movement_metric is not None:
        m_mat = movement_metric.to(device=device, dtype=dtype)
        if m_mat.ndim != 2 or m_mat.shape != (d, d):
            raise ValueError(
                f"Expected `movement_metric` of shape ({d}, {d}), got "
                f"{tuple(m_mat.shape)}."
            )
        if not torch.allclose(m_mat, m_mat.transpose(-1, -2), atol=1e-6, rtol=1e-5):
            raise ValueError("`movement_metric` must be symmetric.")
        sym_m = 0.5 * (m_mat + m_mat.transpose(-1, -2))
        _, info = torch.linalg.cholesky_ex(sym_m)
        if int(info.item()) != 0:
            raise ValueError("`movement_metric` must be symmetric positive definite.")
        sq_dist = ((diff @ sym_m) * diff).sum(dim=-1).clamp_min(0.0)
    else:
        sq_dist = diff.pow(2).sum(dim=-1)

    if cost_exponent == 2.0:
        return sq_dist
    return (sq_dist + smooth_eps).pow(0.5 * cost_exponent)


@dataclass
class MongooseTrainingResult:
    r"""Container for MONGOOSE BPTT meta-training results and diagnostics.

    Attributes:
        policy: The meta-trained ``MongoosePolicy``.
        losses: Mean path-wise meta-loss at each gradient update step.
        best_values: Mean best objective value ``max_t f(x_t)`` per step.
        path_lengths: Mean cumulative trajectory movement cost per step.
    """

    policy: MongoosePolicy
    losses: list[float]
    best_values: list[float]
    path_lengths: list[float]


def _scalarize_objectives(
    f_vals: Tensor,
    weights: Tensor | None,
    scalarization: str,
) -> Tensor:
    r"""Scalarize ``(batch_size, m)`` objective values to ``(batch_size,)``."""
    if f_vals.shape[-1] == 1 and weights is None:
        return f_vals.squeeze(-1)
    if weights is None:  # pragma: no cover
        raise ValueError("`weights` is required when `num_objectives > 1`.")
    if scalarization == "linear":
        return (f_vals * weights).sum(dim=-1)
    if scalarization == "chebyshev":
        return (f_vals * weights).amin(dim=-1)
    raise ValueError(
        f"Unsupported `scalarization` {scalarization!r}. Expected one of "
        f"{SUPPORTED_SCALARIZATIONS}."
    )


def meta_train_mongoose(
    policy: MongoosePolicy,
    num_steps: int = 200,
    batch_size: int = 64,
    horizon: int = 20,
    alpha: float = 0.1,
    alpha_range: tuple[float, float] | None = None,
    lr: float = 3e-3,
    max_grad_norm: float | None = 1.0,
    tau_max: float | None = 0.1,
    noise_std: float = 0.01,
    kernel: str | Kernel | GP = "matern52",
    num_rff_features: int = 256,
    lengthscale: float | Tensor | None = None,
    lengthscale_range: tuple[float, float] = (0.15, 0.5),
    outputscale: float | Tensor | None = None,
    outputscale_range: tuple[float, float] = (1.0, 1.0),
    x0: Tensor | None = None,
    movement_weights: Tensor | None = None,
    movement_metric: Tensor | None = None,
    cost_exponent: float = 1.0,
    smooth_eps: float = 1e-8,
    weights: Tensor | None = None,
    scalarization: str = "linear",
) -> MongooseTrainingResult:
    r"""Meta-train a ``MongoosePolicy`` via BPTT on differentiable GP prior paths.

    Minimizes the MONGOOSE path-wise meta-loss [yang2023mongoose]_:

    .. math::

        \mathcal{L}(\theta) = \mathbb{E}_{f, x_0}\left[
            -\max_{t=0 \dots T} f(x_t)
            + \alpha \sum_{t=0}^{T-1} c(x_{t+1}, x_t)
        \right]

    Args:
        policy: ``MongoosePolicy`` instance to train in-place.
        num_steps: Number of meta-gradient optimization steps (``>= 1``).
        batch_size: Number of GP prior functions rolled out per step (``>= 1``).
        horizon: Optimization horizon ``T`` (number of policy transitions, ``>= 1``).
        alpha: Non-negative movement cost weight used when ``alpha_range is None``.
        alpha_range: Optional ``(min_alpha, max_alpha)`` range from which per-episode
            ``alpha`` values are sampled uniformly at each training step.
        lr: Adam learning rate (``> 0``).
        max_grad_norm: Optional ``L2`` gradient norm clipping threshold (``> 0``).
        tau_max: Optional smooth-max temperature ``>= 0`` using
            ``tau_max * logsumexp(f / tau_max)`` for dense gradients across time
            steps. Pass ``None`` or ``0.0`` to use exact ``torch.amax``.
        noise_std: Standard deviation ``>= 0`` of additive Gaussian observation
            noise ``y_t = f(x_t) + \epsilon_t`` fed to the policy.
        kernel: GP prior kernel specification passed to ``sample_gp_prior_paths``.
        num_rff_features: Number of RFF basis features for ``sample_gp_prior_paths``.
        lengthscale: Optional fixed GP lengthscale for string kernels.
        lengthscale_range: Uniform lengthscale sampling range for string kernels.
        outputscale: Optional fixed GP outputscale for string kernels.
        outputscale_range: Uniform outputscale sampling range for string kernels.
        x0: Optional fixed initial anchor of shape ``(d,)`` or ``(batch_size, d)``
            within ``policy.bounds``. If ``None``, sampled uniformly in ``bounds``.
        movement_weights: Optional per-parameter weights for ``compute_movement_cost``.
        movement_metric: Optional SPD metric ``W`` for ``compute_movement_cost``.
        cost_exponent: Movement distance exponent ``p > 0`` (``1.0`` or ``2.0``).
        smooth_eps: Smoothing epsilon ``> 0`` for ``compute_movement_cost``.
        weights: Optional fixed multi-objective preference weights. When ``None`` and
            ``policy.num_objectives > 1``, sampled uniformly on the simplex per task.
        scalarization: Multi-objective scalarization rule (``"linear"`` or
            ``"chebyshev"``).

    Returns:
        A ``MongooseTrainingResult`` with the trained policy and step-by-step metrics.
    """
    if num_steps < 1:
        raise ValueError(f"`num_steps` must be >= 1, got {num_steps}.")
    if batch_size < 1:
        raise ValueError(f"`batch_size` must be >= 1, got {batch_size}.")
    if horizon < 1:
        raise ValueError(f"`horizon` must be >= 1, got {horizon}.")
    if alpha < 0:
        raise ValueError(f"`alpha` must be >= 0, got {alpha}.")
    if alpha_range is not None:
        if alpha_range[0] < 0 or alpha_range[1] < alpha_range[0]:
            raise ValueError(
                f"`alpha_range` must satisfy 0 <= low <= high, got {alpha_range}."
            )
    if lr <= 0:
        raise ValueError(f"`lr` must be > 0, got {lr}.")
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError(f"`max_grad_norm` must be > 0, got {max_grad_norm}.")
    if tau_max is not None and tau_max < 0:
        raise ValueError(f"`tau_max` must be >= 0 or None, got {tau_max}.")
    if noise_std < 0:
        raise ValueError(f"`noise_std` must be >= 0, got {noise_std}.")
    if scalarization not in SUPPORTED_SCALARIZATIONS:
        raise ValueError(
            f"Unsupported `scalarization` {scalarization!r}. Expected one of "
            f"{SUPPORTED_SCALARIZATIONS}."
        )

    param = next(policy.parameters())
    device, dtype = param.device, param.dtype
    bounds = policy.bounds.to(device=device, dtype=dtype)
    d = policy.input_dim
    m = policy.num_objectives

    fixed_x0: Tensor | None = None
    if x0 is not None:
        x0_t = x0.to(device=device, dtype=dtype)
        if x0_t.ndim == 1 and x0_t.shape[0] == d:
            fixed_x0 = x0_t.unsqueeze(0).expand(batch_size, d)
        elif x0_t.ndim == 2 and x0_t.shape == (batch_size, d):
            fixed_x0 = x0_t
        else:
            raise ValueError(
                f"Expected `x0` of shape ({d},) or ({batch_size}, {d}), got "
                f"{tuple(x0_t.shape)}."
            )
        if torch.any(fixed_x0 < bounds[0]) or torch.any(fixed_x0 > bounds[1]):
            raise ValueError("`x0` must lie within `policy.bounds`.")

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()

    losses: list[float] = []
    best_values: list[float] = []
    path_lengths: list[float] = []

    dirichlet_dist = (
        Dirichlet(torch.ones(m, device=device, dtype=dtype))
        if (m > 1 or policy.condition_on_preferences) and weights is None
        else None
    )

    for _ in range(num_steps):
        optimizer.zero_grad()

        paths = sample_gp_prior_paths(
            input_dim=d,
            batch_size=batch_size,
            num_objectives=m,
            kernel=kernel,
            num_rff_features=num_rff_features,
            lengthscale=lengthscale,
            lengthscale_range=lengthscale_range,
            outputscale=outputscale,
            outputscale_range=outputscale_range,
            bounds=bounds,
            device=device,
            dtype=dtype,
        )

        if alpha_range is not None:
            low_a, high_a = alpha_range
            alpha_batch = low_a + (high_a - low_a) * torch.rand(
                batch_size, device=device, dtype=dtype
            )
        else:
            alpha_batch = torch.full(
                (batch_size,), float(alpha), device=device, dtype=dtype
            )

        if weights is not None:
            w_batch: Tensor | None = _normalize_preference_weights(
                weights=weights,
                batch_size=batch_size,
                num_objectives=m,
                device=device,
                dtype=dtype,
            )
        elif dirichlet_dist is not None:
            w_batch = dirichlet_dist.sample((batch_size,))
        else:
            w_batch = None

        if fixed_x0 is not None:
            x_t = fixed_x0.clone()
        else:
            u0 = torch.rand(batch_size, d, device=device, dtype=dtype)
            x_t = unnormalize(u0, bounds=bounds)

        state = policy.init_hidden(batch_size=batch_size, device=device, dtype=dtype)
        policy_alpha = alpha_batch if policy.condition_on_alpha else None

        f_t = paths(x_t)
        step_scores: list[Tensor] = [
            _scalarize_objectives(f_t, weights=w_batch, scalarization=scalarization)
        ]
        step_costs: list[Tensor] = []

        for t in range(horizon):
            y_t = f_t + noise_std * torch.randn_like(f_t) if noise_std > 0.0 else f_t
            tau_t = t / horizon
            x_next, state = policy(
                x=x_t,
                y=y_t,
                tau=tau_t,
                state=state,
                alpha=policy_alpha,
                weights=w_batch,
            )
            cost_t = compute_movement_cost(
                x_next=x_next,
                x_curr=x_t,
                movement_weights=movement_weights,
                movement_metric=movement_metric,
                cost_exponent=cost_exponent,
                smooth_eps=smooth_eps,
            )
            step_costs.append(cost_t)

            x_t = x_next
            f_t = paths(x_t)
            step_scores.append(
                _scalarize_objectives(f_t, weights=w_batch, scalarization=scalarization)
            )

        scores_mat = torch.stack(step_scores, dim=-1)  # (B, T + 1)
        exact_best = torch.amax(scores_mat, dim=-1)
        if tau_max is None or tau_max == 0.0:
            max_term = exact_best
        else:
            max_term = tau_max * torch.logsumexp(scores_mat / tau_max, dim=-1)

        total_movement = torch.stack(step_costs, dim=-1).sum(dim=-1)  # (B,)
        loss = (-max_term + alpha_batch * total_movement).mean()
        loss.backward()

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        losses.append(float(loss.detach().item()))
        best_values.append(float(exact_best.detach().mean().item()))
        path_lengths.append(float(total_movement.detach().mean().item()))

    return MongooseTrainingResult(
        policy=policy,
        losses=losses,
        best_values=best_values,
        path_lengths=path_lengths,
    )


class MongooseOptimizer:
    r"""Deployment wrapper for stateful and stateless MONGOOSE trajectory suggestion.

    Provides both a stateful incremental loop (``reset(x0, y0) -> x_1`` followed by
    repeated ``step(y_t) -> x_{t+1}``) and a stateless
    ``suggest(X_observed, Y_observed)`` method that replays observation history
    through the LSTM hidden state in sub-millisecond inference time.
    """

    def __init__(
        self,
        policy: MongoosePolicy,
        horizon: int = 50,
        alpha: float | Tensor | None = None,
        weights: Tensor | None = None,
        maximize: bool = True,
        y_shift: float | Tensor = 0.0,
        y_scale: float | Tensor = 1.0,
    ) -> None:
        r"""Initialize ``MongooseOptimizer``.

        Args:
            policy: Meta-trained ``MongoosePolicy``.
            horizon: Planned optimization horizon ``T >= 1`` used to compute
                normalized horizon progress ``tau_t = min(t / T, 1.0)``.
            alpha: Movement penalty context passed to ``policy`` when
                ``policy.condition_on_alpha=True`` (defaults to ``0.0`` if ``None``).
            weights: Optional multi-objective preference weights in the
                ``(m - 1)``-simplex.
            maximize: If ``False``, negates observed values before feeding them to
                the maximization-trained policy.
            y_shift: Optional additive location shift subtracted from observations.
            y_scale: Optional positive scale factor dividing shifted observations.
        """
        if horizon < 1:
            raise ValueError(f"`horizon` must be >= 1, got {horizon}.")
        if isinstance(y_scale, (int, float)):
            if y_scale <= 0:
                raise ValueError(f"`y_scale` must be > 0, got {y_scale}.")
        elif torch.any(y_scale <= 0):
            raise ValueError("`y_scale` must be > 0 in all entries.")

        self.policy = policy
        self.horizon = horizon
        self.alpha = 0.0 if (alpha is None and policy.condition_on_alpha) else alpha
        self.weights = weights
        self.maximize = maximize
        self.y_shift = y_shift
        self.y_scale = y_scale

        self._state: list[tuple[Tensor, Tensor]] | None = None
        self._pending_x: Tensor | None = None
        self._step_count: int = 0
        self._is_unbatched: bool = True
        self._x_history: list[Tensor] = []
        self._y_history: list[Tensor] = []

    @property
    def step_count(self) -> int:
        r"""Number of observations processed so far in the stateful loop."""
        return self._step_count

    @property
    def pending_x(self) -> Tensor | None:
        r"""Most recently suggested candidate awaiting observation in ``step``."""
        return self._pending_x

    @property
    def X_observed(self) -> Tensor | None:
        r"""Stacked history of evaluated points from ``reset`` and ``step``."""
        if not self._x_history:
            return None
        dim = 0 if self._is_unbatched else 1
        return torch.stack(self._x_history, dim=dim)

    @property
    def Y_observed(self) -> Tensor | None:
        r"""Stacked history of raw observations from ``reset`` and ``step``."""
        if not self._y_history:
            return None
        dim = 0 if self._is_unbatched else 1
        return torch.stack(self._y_history, dim=dim)

    def _transform_y(self, y: Tensor) -> Tensor:
        r"""Apply causal shift/scale and maximization sign convention to ``y``."""
        shift = (
            torch.as_tensor(self.y_shift, device=y.device, dtype=y.dtype)
            if not isinstance(self.y_shift, (int, float))
            else float(self.y_shift)
        )
        scale = (
            torch.as_tensor(self.y_scale, device=y.device, dtype=y.dtype)
            if not isinstance(self.y_scale, (int, float))
            else float(self.y_scale)
        )
        y_proc = (y - shift) / scale
        return y_proc if self.maximize else -y_proc

    @torch.no_grad()
    def reset(self, x0: Tensor, y0: Tensor) -> Tensor:
        r"""Reset recurrent state with initial observation ``(x0, y0)``.

        Args:
            x0: Initial evaluated configuration of shape ``(d,)`` or ``(B, d)``.
            y0: Observed objective value(s) at ``x0`` of shape ``()`` / ``(m,)``
                (unbatched) or ``(B,)`` / ``(B, m)`` (batched).

        Returns:
            First suggested query point ``x_1`` with the same shape as ``x0``.
        """
        if x0.ndim not in (1, 2):
            raise ValueError(f"`x0` must be 1D or 2D, got {x0.ndim}D.")
        self._is_unbatched = x0.ndim == 1
        batch_size = 1 if self._is_unbatched else x0.shape[0]

        self.policy.eval()
        state = self.policy.init_hidden(
            batch_size=batch_size, device=x0.device, dtype=x0.dtype
        )
        y0_proc = self._transform_y(y0)
        tau_0 = 0.0
        policy_alpha = self.alpha if self.policy.condition_on_alpha else None

        x1, next_state = self.policy(
            x=x0,
            y=y0_proc,
            tau=tau_0,
            state=state,
            alpha=policy_alpha,
            weights=self.weights,
        )

        self._state = next_state
        self._pending_x = x1.detach().clone()
        self._step_count = 1
        self._x_history = [x0.detach().clone()]
        self._y_history = [y0.detach().clone()]
        return x1

    @torch.no_grad()
    def step(self, y_new: Tensor, x_new: Tensor | None = None) -> Tensor:
        r"""Update LSTM state with the latest observation and suggest ``x_{t+1}``.

        Args:
            y_new: Newly observed value(s) at the last suggested point ``pending_x``
                (or at ``x_new`` if explicitly provided).
            x_new: Optional actual evaluated point overriding ``pending_x`` (useful
                if a physical actuator settled slightly away from ``pending_x``).

        Returns:
            Next suggested query point ``x_{t+1}``.
        """
        if self._state is None or self._pending_x is None:
            raise RuntimeError(
                "Optimizer must be initialized via `reset(x0, y0)` before `step`."
            )
        x_eval = self._pending_x if x_new is None else x_new
        if x_eval.shape != self._pending_x.shape:
            raise ValueError(
                f"Expected `x_new` of shape {tuple(self._pending_x.shape)}, got "
                f"{tuple(x_eval.shape)}."
            )

        self.policy.eval()
        tau_t = min(self._step_count / self.horizon, 1.0)
        y_proc = self._transform_y(y_new)
        policy_alpha = self.alpha if self.policy.condition_on_alpha else None

        x_next, next_state = self.policy(
            x=x_eval,
            y=y_proc,
            tau=tau_t,
            state=self._state,
            alpha=policy_alpha,
            weights=self.weights,
        )

        self._state = next_state
        self._pending_x = x_next.detach().clone()
        self._step_count += 1
        self._x_history.append(x_eval.detach().clone())
        self._y_history.append(y_new.detach().clone())
        return x_next

    @torch.no_grad()
    def suggest(
        self,
        X_observed: Tensor,
        Y_observed: Tensor,
        horizon: int | None = None,
        alpha: float | Tensor | None = None,
        weights: Tensor | None = None,
    ) -> Tensor:
        r"""Statelessly replay ``(X_observed, Y_observed)`` to suggest ``x_{t+1}``.

        Args:
            X_observed: Observed query history of shape ``(n, d)`` (unbatched) or
                ``(B, n, d)`` (batched), with ``n >= 1``.
            Y_observed: Observed objective history of shape ``(n, m)`` / ``(n,)``
                (unbatched) or ``(B, n, m)`` / ``(B, n)`` (batched).
            horizon: Optional override for horizon ``T >= 1``. Defaults to
                ``self.horizon``.
            alpha: Optional override for movement penalty context ``alpha``.
            weights: Optional override for multi-objective preference weights.

        Returns:
            Suggested next query point ``x_n`` of shape ``(d,)`` (unbatched) or
            ``(B, d)`` (batched).
        """
        eff_horizon = self.horizon if horizon is None else horizon
        if eff_horizon < 1:
            raise ValueError(f"`horizon` must be >= 1, got {eff_horizon}.")

        if X_observed.ndim == 2:
            is_unbatched = True
            X_seq = X_observed.unsqueeze(0)
        elif X_observed.ndim == 3:
            is_unbatched = False
            X_seq = X_observed
        else:
            raise ValueError(
                f"`X_observed` must be 2D `(n, d)` or 3D `(B, n, d)`, got "
                f"{X_observed.ndim}D."
            )

        batch_size, seq_len, d = X_seq.shape
        if seq_len < 1:
            raise ValueError("`X_observed` must contain at least one observation.")
        if d != self.policy.input_dim:
            raise ValueError(
                f"Expected `X_observed` last dim {self.policy.input_dim}, got {d}."
            )

        m = self.policy.num_objectives
        if is_unbatched:
            if Y_observed.ndim == 1 and m == 1 and Y_observed.shape[0] == seq_len:
                Y_seq = Y_observed.reshape(1, seq_len, 1)
            elif Y_observed.ndim == 2 and Y_observed.shape == (seq_len, m):
                Y_seq = Y_observed.unsqueeze(0)
            else:
                raise ValueError(
                    f"Expected `Y_observed` of shape ({seq_len}, {m}), got "
                    f"{tuple(Y_observed.shape)}."
                )
        else:
            if (
                Y_observed.ndim == 2
                and m == 1
                and Y_observed.shape == (batch_size, seq_len)
            ):
                Y_seq = Y_observed.unsqueeze(-1)
            elif Y_observed.ndim == 3 and Y_observed.shape == (batch_size, seq_len, m):
                Y_seq = Y_observed
            else:
                raise ValueError(
                    f"Expected `Y_observed` of shape ({batch_size}, {seq_len}, {m}), "
                    f"got {tuple(Y_observed.shape)}."
                )

        self.policy.eval()
        state = self.policy.init_hidden(
            batch_size=batch_size, device=X_seq.device, dtype=X_seq.dtype
        )
        eff_alpha = self.alpha if alpha is None else alpha
        policy_alpha = eff_alpha if self.policy.condition_on_alpha else None
        eff_weights = self.weights if weights is None else weights
        Y_proc = self._transform_y(Y_seq)

        x_next = X_seq[:, 0, :]
        for t in range(seq_len):
            tau_t = min(t / eff_horizon, 1.0)
            x_next, state = self.policy(
                x=X_seq[:, t, :],
                y=Y_proc[:, t, :],
                tau=tau_t,
                state=state,
                alpha=policy_alpha,
                weights=eff_weights,
            )

        return x_next.squeeze(0) if is_unbatched else x_next
