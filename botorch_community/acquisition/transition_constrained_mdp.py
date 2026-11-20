#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Transition-Constrained Bayesian Optimization via Markov Decision Processes.

Implements finite-horizon transition-constrained MDPs, confidence-bound potential
maximizer sets (single-objective and Pareto multi-objective), concave experimental
design utilities (A-optimality and D-optimality) over state-visitation measures via
a numerically stable precision/Woodbury identity, and Frank-Wolfe policy-mixture
and history-dependent receding-horizon planning from [folch2024transitionmdp]_
and [mutny2023markovdesign]_.

References

.. [folch2024transitionmdp]
    J. P. Folch, C. Tsay, R. M. Lee, B. Shafei, W. Ormaniec, A. Krause,
    M. van der Wilk, R. Misener, and M. Mutný.
    Transition Constrained Bayesian Optimization via Markov Decision Processes.
    Advances in Neural Information Processing Systems (NeurIPS), 37, 2024.
.. [mutny2023markovdesign]
    M. Mutný, T. Janik, and A. Krause.
    Active Exploration via Experiment Design in Markov Chains.
    Proceedings of The 26th International Conference on Artificial Intelligence
    and Statistics (AISTATS), PMLR 206:7349-7374, 2023.

Contributor: M-Colley
"""

from __future__ import annotations

import math

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from botorch.posteriors.posterior_list import PosteriorList
from torch import Tensor

SUPPORTED_CRITERIA = ("a_optimal", "d_optimal")
SUPPORTED_STEP_SIZE_RULES = ("line_search", "classic")


class TransitionMDP:
    r"""Finite-horizon Markov Decision Process for transition-constrained BO.

    Models a discrete or discretized design space ``states`` of size ``S x d``
    equipped with a Markov transition kernel ``transition_probs`` of shape
    ``S x A x S`` (supporting both deterministic graph/step transitions and
    general stochastic Markov transitions) over a planning horizon ``H >= 1``.
    Starting from ``initial_state`` :math:`s_0`, a policy selects actions
    :math:`a_0, \dots, a_{H-1}` generating evaluated states
    :math:`s_1, \dots, s_H` with expected visit counts
    :math:`\eta^\pi(s) = \sum_{h=1}^{H} \mathbb{P}^\pi(s_h = s \mid s_0)`
    satisfying :math:`\sum_{s=0}^{S-1} \eta^\pi(s) = H`.
    """

    def __init__(
        self,
        states: Tensor,
        transition_probs: Tensor,
        horizon: int = 10,
        action_mask: Tensor | None = None,
        atol: float = 1e-5,
    ) -> None:
        r"""Initialize a finite-horizon ``TransitionMDP``.

        Args:
            states: A ``S x d``-dim floating-point Tensor of discrete state
                coordinates with ``S >= 1`` and ``d >= 1``.
            transition_probs: A ``S x A x S``-dim Tensor giving transition
                probabilities ``P(s' | s, a)`` for each state ``s`` and action ``a``.
            horizon: Planning horizon ``H >= 1`` (number of transition steps /
                queries in an episode).
            action_mask: Optional ``S x A``-dim boolean Tensor indicating which
                actions are valid at each state. Every state must have at least
                one valid action. If ``None``, all ``A`` actions are valid.
            atol: Numerical tolerance used to validate probability simplex rows.
        """
        if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1:
            raise ValueError(
                f"`states` must have shape (S, d) with S, d >= 1; got {states.shape}."
            )
        if not states.is_floating_point():
            raise ValueError("`states` must be a floating-point Tensor.")
        num_states = states.shape[0]
        if (
            transition_probs.ndim != 3
            or transition_probs.shape[0] != num_states
            or transition_probs.shape[2] != num_states
            or transition_probs.shape[1] < 1
        ):
            raise ValueError(
                "`transition_probs` must have shape (S, A, S) matching `states`; "
                f"got {transition_probs.shape} for S={num_states}."
            )
        if horizon < 1:
            raise ValueError(f"`horizon` must be >= 1; got {horizon}.")

        num_actions = transition_probs.shape[1]
        transition_probs = transition_probs.to(device=states.device, dtype=states.dtype)

        if action_mask is None:
            mask = torch.ones(
                (num_states, num_actions), dtype=torch.bool, device=states.device
            )
        else:
            if action_mask.shape != (num_states, num_actions):
                raise ValueError(
                    f"`action_mask` must have shape {(num_states, num_actions)}; "
                    f"got {action_mask.shape}."
                )
            mask = action_mask.to(device=states.device, dtype=torch.bool)
            if not mask.any(dim=-1).all():
                raise ValueError(
                    "Every state must have at least one valid action in `action_mask`."
                )

        valid_probs = transition_probs[mask]
        if (valid_probs < -atol).any():
            raise ValueError(
                "`transition_probs` contains negative entries on valid actions."
            )
        row_sums = valid_probs.sum(dim=-1)
        if (row_sums - 1.0).abs().max().item() > atol:
            raise ValueError(
                "Valid `(s, a)` rows of `transition_probs` must sum to 1.0."
            )

        self.states: Tensor = states
        self.transition_probs: Tensor = transition_probs.clamp_min(0.0)
        self.action_mask: Tensor = mask
        self.horizon: int = int(horizon)

    @property
    def num_states(self) -> int:
        r"""Number of discrete states ``S``."""
        return self.states.shape[0]

    @property
    def dim(self) -> int:
        r"""State space feature dimension ``d``."""
        return self.states.shape[1]

    @property
    def num_actions(self) -> int:
        r"""Number of action slots ``A`` per state."""
        return self.transition_probs.shape[1]

    @property
    def device(self) -> torch.device:
        r"""Torch device of the MDP tensors."""
        return self.states.device

    @property
    def dtype(self) -> torch.dtype:
        r"""Torch floating-point dtype of the MDP tensors."""
        return self.states.dtype

    @classmethod
    def from_step_constraints(
        cls,
        states: Tensor,
        max_step: float | Tensor,
        p: float = 2.0,
        metric: Tensor | None = None,
        horizon: int = 10,
    ) -> TransitionMDP:
        r"""Construct a deterministic transition-graph MDP from local step bounds.

        From state ``i``, any state ``j`` satisfying ``d(states[i], states[j]) <=
        max_step`` is reachable in a single transition. Self-loops ``j = i`` are
        always included so every state has at least one valid action.

        Args:
            states: A ``S x d``-dim floating-point Tensor of discrete states.
            max_step: Maximum allowed step size per transition. Either a non-negative
                scalar (``float`` or 0D/1-element ``Tensor``) or, when ``metric`` is
                ``None``, a ``d``-dim Tensor of per-dimension non-negative step limits
                enforcing :math:`\| (x_i - x_j) / \text{max\_step} \|_p \le 1`.
            p: Order ``p > 0`` of the :math:`\ell_p` norm used when ``metric`` is
                ``None`` or 1D (supports ``float("inf")``).
            metric: Optional distance metric specification. Either a ``d``-dim Tensor
                of positive coordinate weights or a ``d x d`` symmetric positive
                semi-definite matrix defining Mahalanobis distance
                :math:`\sqrt{(x_i - x_j)^\top M (x_i - x_j)}`.
            horizon: Planning horizon ``H >= 1``.

        Returns:
            A ``TransitionMDP`` encoding the feasible step-constrained transition graph.
        """
        if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1:
            raise ValueError(
                f"`states` must have shape (S, d) with S, d >= 1; got {states.shape}."
            )
        if p <= 0:
            raise ValueError(f"`p` must be positive; got {p}.")

        num_states, dim = states.shape
        diff = states.unsqueeze(1) - states.unsqueeze(0)  # (S, S, d)

        if metric is not None:
            metric = metric.to(device=states.device, dtype=states.dtype)
            step_val = (
                float(max_step.item())
                if isinstance(max_step, Tensor)
                else float(max_step)
            )
            if step_val < 0.0:
                raise ValueError(f"`max_step` must be >= 0; got {step_val}.")
            if metric.ndim == 1:
                if metric.shape != (dim,) or (metric <= 0).any():
                    raise ValueError(
                        "1D `metric` must have shape (d,) with strictly positive "
                        "entries."
                    )
                if math.isinf(p):
                    dists = (diff.abs() * metric).amax(dim=-1)
                else:
                    dists = ((diff.abs() ** p) * metric).sum(dim=-1).pow(1.0 / p)
            elif metric.ndim == 2:
                if metric.shape != (dim, dim):
                    raise ValueError(
                        f"2D `metric` must have shape {(dim, dim)}; got {metric.shape}."
                    )
                quad = torch.einsum("ijd,de,ije->ij", diff, metric, diff)
                dists = quad.clamp_min(0.0).sqrt()
            else:
                raise ValueError("`metric` must be a 1D or 2D Tensor.")
            feasible = dists <= step_val + 1e-12
        elif isinstance(max_step, Tensor) and max_step.ndim == 1:
            step_vec = max_step.to(device=states.device, dtype=states.dtype)
            if step_vec.shape != (dim,) or (step_vec < 0).any():
                raise ValueError(
                    "1D `max_step` must have shape (d,) with non-negative entries."
                )
            zero_dims = step_vec == 0.0
            exact_ok = (
                (diff[..., zero_dims].abs() <= 1e-12).all(dim=-1)
                if zero_dims.any()
                else torch.ones(
                    (num_states, num_states), dtype=torch.bool, device=states.device
                )
            )
            pos_dims = ~zero_dims
            if pos_dims.any():
                scaled = diff[..., pos_dims] / step_vec[pos_dims]
                norms = torch.linalg.vector_norm(scaled, ord=p, dim=-1)
                feasible = exact_ok & (norms <= 1.0 + 1e-12)
            else:
                feasible = exact_ok
        else:
            step_val = (
                float(max_step.item())
                if isinstance(max_step, Tensor)
                else float(max_step)
            )
            if step_val < 0.0:
                raise ValueError(f"`max_step` must be >= 0; got {step_val}.")
            dists = torch.linalg.vector_norm(diff, ord=p, dim=-1)
            feasible = dists <= step_val + 1e-12

        feasible.fill_diagonal_(True)
        degrees = feasible.sum(dim=-1)
        max_deg = int(degrees.max().item())

        transition_probs = torch.zeros(
            (num_states, max_deg, num_states),
            dtype=states.dtype,
            device=states.device,
        )
        action_mask = torch.zeros(
            (num_states, max_deg), dtype=torch.bool, device=states.device
        )

        for s in range(num_states):
            nbrs = torch.where(feasible[s])[0]
            deg = nbrs.numel()
            action_Idx = torch.arange(deg, device=states.device)
            transition_probs[s, action_Idx, nbrs] = 1.0
            action_mask[s, :deg] = True
            if deg < max_deg:
                transition_probs[s, deg:, s] = 1.0

        return cls(
            states=states,
            transition_probs=transition_probs,
            horizon=horizon,
            action_mask=action_mask,
        )

    def _validate_state_index(self, state_idx: int, name: str = "initial_state") -> int:
        if not (0 <= int(state_idx) < self.num_states):
            raise ValueError(
                f"`{name}` must be in [0, {self.num_states - 1}]; got {state_idx}."
            )
        return int(state_idx)

    def compute_policy_visitation(
        self,
        policy: Tensor,
        initial_state: int,
        horizon: int | None = None,
        normalize: bool = False,
        return_stagewise: bool = False,
    ) -> Tensor:
        r"""Compute exact state-visitation counts via forward Markov propagation.

        Args:
            policy: Either a ``H' x S``-dim integer Tensor of deterministic actions
                or a ``H' x S x A``-dim floating-point Tensor of randomized Markov
                action probabilities.
            initial_state: Starting state index :math:`s_0 \in \{0, \dots, S-1\}`.
            horizon: Number of steps ``H'`` to propagate. Defaults to
                ``policy.shape[0]``.
            normalize: If ``True``, divide expected visit counts by ``H'`` so the
                returned distribution sums to 1.0 instead of ``H'``.
            return_stagewise: If ``True``, return the ``H' x S`` stagewise state
                distributions :math:`(p_1, \dots, p_{H'})` instead of summing over
                stages.

        Returns:
            A ``S``-dim Tensor of expected state-visitation counts (or ``H' x S``
            stagewise distributions if ``return_stagewise=True``).
        """
        s0 = self._validate_state_index(initial_state, "initial_state")
        plan_horizon = int(policy.shape[0]) if horizon is None else int(horizon)
        if plan_horizon < 1 or policy.shape[0] < plan_horizon:
            raise ValueError(
                f"Invalid `horizon`={plan_horizon} for `policy` shape {policy.shape}."
            )

        state_range = torch.arange(self.num_states, device=self.device)
        p_curr = torch.zeros(self.num_states, dtype=self.dtype, device=self.device)
        p_curr[s0] = 1.0
        stage_dists: list[Tensor] = []

        for h in range(plan_horizon):
            if policy.ndim == 2:
                act_h = policy[h].to(device=self.device, dtype=torch.long)
                trans_h = self.transition_probs[state_range, act_h]
            elif policy.ndim == 3:
                probs_h = policy[h].to(device=self.device, dtype=self.dtype)
                trans_h = torch.einsum("sa,sat->st", probs_h, self.transition_probs)
            else:
                raise ValueError(
                    f"`policy` must be 2D (H, S) or 3D (H, S, A); got {policy.shape}."
                )
            p_curr = p_curr @ trans_h
            stage_dists.append(p_curr)

        stacked = torch.stack(stage_dists, dim=0)
        if return_stagewise:
            return stacked
        visitation = stacked.sum(dim=0)
        if normalize:
            visitation = visitation / float(plan_horizon)
        return visitation

    def solve_linear_mdp(
        self,
        state_rewards: Tensor,
        initial_state: int,
        horizon: int | None = None,
        normalize: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Solve a linear-reward finite-horizon MDP via exact backward induction DP.

        Computes the optimal non-stationary deterministic policy maximizing
        :math:`\mathbb{E}^\pi [ \sum_{h=0}^{H-1} R_h(s_{h+1}) \mid s_0 ]` and its
        exact induced state-visitation measure via forward Markov propagation.

        Args:
            state_rewards: State rewards ``R`` of shape ``(S,)`` (stationary across
                stages) or ``(H, S)`` (stage-specific rewards upon entering ``s'`` at
                stage ``h``).
            initial_state: Initial state index :math:`s_0 \in \{0, \dots, S-1\}`.
            horizon: Optional planning horizon override ``H >= 1``. Defaults to
                ``self.horizon``.
            normalize: If ``True``, normalize the returned visitation distribution
                to sum to 1.0 instead of ``H``.

        Returns:
            A 3-tuple ``(values, policy, visitation)`` where:
            - ``values``: Optimal value table of shape ``(H, S)``.
            - ``policy``: Optimal action indices of shape ``(H, S)`` (``torch.long``).
            - ``visitation``: Induced state-visitation measure of shape ``(S,)``
              summing to ``H`` (or ``1.0`` when ``normalize=True``).
        """
        s0 = self._validate_state_index(initial_state, "initial_state")
        plan_horizon = self.horizon if horizon is None else int(horizon)
        if plan_horizon < 1:
            raise ValueError(f"`horizon` must be >= 1; got {plan_horizon}.")

        work_dtype = torch.promote_types(self.dtype, state_rewards.dtype)
        rewards = state_rewards.to(device=self.device, dtype=work_dtype)
        if rewards.ndim == 1:
            if rewards.shape[0] != self.num_states:
                raise ValueError(
                    f"1D `state_rewards` must have length {self.num_states}; "
                    f"got {rewards.shape[0]}."
                )
            rewards = rewards.unsqueeze(0).expand(plan_horizon, self.num_states)
        elif rewards.ndim == 2:
            if rewards.shape != (plan_horizon, self.num_states):
                raise ValueError(
                    "`state_rewards` must have shape "
                    f"{(plan_horizon, self.num_states)}; got {rewards.shape}."
                )
        else:
            raise ValueError(
                f"`state_rewards` must be 1D or 2D; got {rewards.ndim}D Tensor."
            )

        trans = self.transition_probs.to(dtype=work_dtype)
        values = torch.empty(
            (plan_horizon, self.num_states), dtype=work_dtype, device=self.device
        )
        policy = torch.empty(
            (plan_horizon, self.num_states), dtype=torch.long, device=self.device
        )

        v_next = torch.zeros(self.num_states, dtype=work_dtype, device=self.device)
        neg_inf = torch.tensor(float("-inf"), dtype=work_dtype, device=self.device)

        for h in range(plan_horizon - 1, -1, -1):
            continuation = rewards[h] + v_next  # (S,)
            q_vals = torch.matmul(trans, continuation)  # (S, A)
            q_vals = torch.where(self.action_mask, q_vals, neg_inf)
            v_h, pi_h = q_vals.max(dim=-1)
            values[h] = v_h
            policy[h] = pi_h
            v_next = v_h

        state_range = torch.arange(self.num_states, device=self.device)
        p_curr = torch.zeros(self.num_states, dtype=work_dtype, device=self.device)
        p_curr[s0] = 1.0
        visitation = torch.zeros(self.num_states, dtype=work_dtype, device=self.device)

        for h in range(plan_horizon):
            trans_h = trans[state_range, policy[h]]
            p_curr = p_curr @ trans_h
            visitation = visitation + p_curr

        if normalize:
            visitation = visitation / float(plan_horizon)
        return values, policy, visitation


def _extract_cov_blocks_from_posterior(
    posterior: Posterior, num_states: int
) -> list[Tensor]:
    r"""Extract per-output ``S x S`` latent covariance matrices from a posterior."""
    if isinstance(posterior, PosteriorList):
        blocks: list[Tensor] = []
        for sub_post in posterior.posteriors:
            blocks.extend(_extract_cov_blocks_from_posterior(sub_post, num_states))
        return blocks

    if not hasattr(posterior, "distribution"):
        raise ValueError(
            "Model posterior must provide `distribution` (e.g., `GPyTorchPosterior`) "
            "or `posteriors` (`PosteriorList`)."
        )

    mvn = posterior.distribution
    cov = mvn.covariance_matrix
    num_outputs = posterior.mean.shape[-1]
    if num_outputs == 1 and not getattr(posterior, "_is_mt", False):
        if cov.shape != (num_states, num_states):
            raise ValueError(
                f"Expected ({num_states}, {num_states}) covariance, got {cov.shape}."
            )
        return [0.5 * (cov + cov.transpose(-1, -2))]

    interleaved = getattr(mvn, "_interleaved", True)
    if interleaved:
        cov_4d = cov.reshape(num_states, num_outputs, num_states, num_outputs)
        per_out = torch.diagonal(cov_4d, dim1=1, dim2=3).permute(2, 0, 1)
    else:
        cov_4d = cov.reshape(num_outputs, num_states, num_outputs, num_states)
        per_out = torch.diagonal(cov_4d, dim1=0, dim2=2).permute(2, 0, 1)

    per_out = 0.5 * (per_out + per_out.transpose(-1, -2))
    return [per_out[m] for m in range(num_outputs)]


def _extract_posterior_cov_and_noise(
    model: Model,
    states: Tensor,
    observation_noise: float | Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Extract ``(M, S, S)`` latent covariances and ``(M, S)`` noise variances."""
    num_states = states.shape[0]
    min_noise = 1e-12 if states.dtype == torch.double else 1e-7

    with torch.no_grad():
        post_f = model.posterior(states, observation_noise=False)
        cov_blocks = _extract_cov_blocks_from_posterior(post_f, num_states=num_states)
        cov_matrix = torch.stack(cov_blocks, dim=0)  # (M, S, S)
        num_outputs = cov_matrix.shape[0]

        if observation_noise is None:
            post_y = model.posterior(states, observation_noise=True)
            noise_sm = (post_y.variance - post_f.variance).clamp_min(min_noise)
            noise_var = noise_sm.transpose(0, 1).contiguous()  # (M, S)
        elif isinstance(observation_noise, Tensor):
            obs_t = observation_noise.to(
                device=cov_matrix.device, dtype=cov_matrix.dtype
            )
            if obs_t.ndim == 0:
                noise_var = obs_t.expand(num_outputs, num_states)
            elif obs_t.ndim == 1 and obs_t.shape[0] == num_states:
                noise_var = obs_t.unsqueeze(0).expand(num_outputs, num_states)
            elif obs_t.ndim == 1 and obs_t.shape[0] == num_outputs:
                noise_var = obs_t.unsqueeze(-1).expand(num_outputs, num_states)
            elif obs_t.shape == (num_states, num_outputs):
                noise_var = obs_t.transpose(0, 1).contiguous()
            elif obs_t.shape == (num_outputs, num_states):
                noise_var = obs_t
            else:
                raise ValueError(
                    "`observation_noise` Tensor has incompatible shape "
                    f"{obs_t.shape} for S={num_states}, M={num_outputs}."
                )
            if (noise_var <= 0).any():
                raise ValueError("`observation_noise` must be strictly positive.")
        else:
            val = float(observation_noise)
            if val <= 0.0:
                raise ValueError(f"`observation_noise` must be > 0; got {val}.")
            noise_var = torch.full(
                (num_outputs, num_states),
                fill_value=val,
                dtype=cov_matrix.dtype,
                device=cov_matrix.device,
            )

    return cov_matrix.detach(), noise_var.detach()


def compute_potential_maximizers(
    model: Model,
    states: Tensor,
    beta: float = 4.0,
) -> Tensor:
    r"""Compute the confidence-bound set of potential maximizers :math:`\mathcal{Z}_t`.

    Using marginal confidence bounds :math:`l_t(s) = \mu_t(s) - \sqrt{\beta}\sigma_t(s)`
    and :math:`u_t(s) = \mu_t(s) + \sqrt{\beta}\sigma_t(s)`:
    - Single-objective (``M == 1``):
      :math:`\mathcal{Z}_t = \{s : u_t(s) \ge \max_{s'} l_t(s')\}`.
    - Multi-objective (``M > 1``):
      :math:`\mathcal{Z}_t = \{s : \neg \exists s' \text{ s.t. } l_t(s') \succ u_t(s)\}`
      where :math:`a \succ b` denotes strict Pareto dominance (:math:`\forall m,
      a_m \ge b_m` and :math:`\exists m, a_m > b_m`).

    Args:
        model: A fitted BoTorch ``Model`` (single-output or multi-output).
        states: A ``S x d``-dim Tensor of discrete state coordinates.
        beta: Non-negative confidence scaling parameter :math:`\beta \ge 0`.

    Returns:
        A 1D ``torch.long`` Tensor of sorted state indices belonging to
        :math:`\mathcal{Z}_t` (always non-empty).
    """
    if beta < 0.0:
        raise ValueError(f"`beta` must be >= 0; got {beta}.")

    with torch.no_grad():
        post = model.posterior(states, observation_noise=False)
        mean = post.mean  # (S, M)
        std = post.variance.clamp_min(0.0).sqrt()  # (S, M)

    radius = math.sqrt(beta) * std
    lcb = mean - radius
    ucb = mean + radius

    if mean.shape[-1] == 1:
        max_lcb = lcb.squeeze(-1).max()
        mask = ucb.squeeze(-1) >= max_lcb
    else:
        # dominates[s', s] is True iff lcb[s'] strictly Pareto-dominates ucb[s]
        ge_all = (lcb.unsqueeze(1) >= ucb.unsqueeze(0)).all(dim=-1)
        gt_any = (lcb.unsqueeze(1) > ucb.unsqueeze(0)).any(dim=-1)
        dominates = ge_all & gt_any
        mask = ~dominates.any(dim=0)

    if not mask.any():
        best_idx = lcb.sum(dim=-1).argmax()
        mask[best_idx] = True

    return torch.where(mask)[0]


def compute_posterior_covariance_woodbury(
    cov_matrix: Tensor,
    noise_var: Tensor,
    visitation: Tensor,
    target_indices: Tensor | None = None,
) -> Tensor:
    r"""Compute exact posterior covariance :math:`\Sigma_{\mathcal{Z}}(\eta)`.

    Evaluates the posterior covariance on ``target_indices`` :math:`\mathcal{Z}`
    induced by allocating expected visit counts :math:`\eta \in \mathbb{R}^S_{\ge 0}`
    across states using the singularity-free precision push-through identity:

    .. math::
        \Sigma_{\mathcal{Z}}(\eta) = K_{\mathcal{Z},\mathcal{Z}}
        - K_{\mathcal{Z},\mathcal{S}} \left(I_S + W(\eta) K_{\mathcal{S},\mathcal{S}}
        \right)^{-1} W(\eta) K_{\mathcal{S},\mathcal{Z}},
        \quad W(\eta) = \operatorname{diag}(\eta / \sigma_n^2).

    Note that :math:`I_S + W(\eta) K_{\mathcal{S},\mathcal{S}}` has all eigenvalues
    :math:`\ge 1` for every :math:`\eta \ge 0` (including exact zeros
    :math:`\eta_s = 0`), enabling smooth, exact ``autograd`` differentiation at
    boundary allocations without ``0 / 0`` or ``sqrt(0)`` singularities.

    Args:
        cov_matrix: Prior/current latent covariance ``K`` of shape ``(S, S)`` or
            ``(M, S, S)``.
        noise_var: Positive observation noise variances of shape ``(S,)`` or
            ``(M, S)``.
        visitation: Non-negative state-visitation allocation :math:`\eta` of shape
            ``(S,)`` or ``(..., S)``.
        target_indices: Optional 1D integer Tensor of target state indices
            :math:`\mathcal{Z}`. If ``None``, uses all ``S`` states.

    Returns:
        Posterior covariance Tensor of shape ``(Z, Z)`` (when ``cov_matrix`` is 2D
        and ``visitation`` is 1D) or ``(..., M, Z, Z)`` / ``(M, Z, Z)``.
    """
    squeeze_output = cov_matrix.ndim == 2
    if squeeze_output:
        cov = cov_matrix.unsqueeze(0)
    elif cov_matrix.ndim == 3:
        cov = cov_matrix
    else:
        raise ValueError(
            f"`cov_matrix` must be 2D or 3D; got shape {cov_matrix.shape}."
        )

    num_outputs, num_states, _ = cov.shape
    if noise_var.ndim == 1:
        if noise_var.shape[0] != num_states:
            raise ValueError(
                f"1D `noise_var` must have length {num_states}; got {noise_var.shape}."
            )
        n_var = noise_var.unsqueeze(0).expand(num_outputs, num_states)
    elif noise_var.ndim == 2 and noise_var.shape == (num_outputs, num_states):
        n_var = noise_var
    else:
        raise ValueError(
            f"`noise_var` must have shape ({num_states},) or "
            f"{(num_outputs, num_states)}; got {noise_var.shape}."
        )

    if visitation.shape[-1] != num_states:
        raise ValueError(
            f"`visitation` last dim must be {num_states}; got {visitation.shape}."
        )
    if (visitation.detach() < -1e-6).any():
        raise ValueError("`visitation` entries must be non-negative.")

    work_dtype = torch.promote_types(cov.dtype, visitation.dtype)
    cov = cov.to(dtype=work_dtype)
    n_var = n_var.to(device=cov.device, dtype=work_dtype)
    eta = visitation.to(device=cov.device, dtype=work_dtype)

    if target_indices is None:
        cov_zz = cov
        cov_sz = cov
        cov_zs = cov
    else:
        idx = target_indices.to(device=cov.device, dtype=torch.long)
        cov_sz = cov.index_select(dim=-1, index=idx)  # (M, S, Z)
        cov_zs = cov.index_select(dim=-2, index=idx)  # (M, Z, S)
        cov_zz = cov_zs.index_select(dim=-1, index=idx)  # (M, Z, Z)

    batch_shape = eta.shape[:-1]
    # Precision weights w = eta / sigma_n^2 of shape (..., M, S)
    weights = eta.unsqueeze(-2) / n_var
    eye_s = torch.eye(num_states, dtype=work_dtype, device=cov.device)

    expanded_cov = cov.expand(*batch_shape, num_outputs, num_states, num_states)
    expanded_sz = cov_sz.expand(*batch_shape, *cov_sz.shape)
    expanded_zs = cov_zs.expand(*batch_shape, *cov_zs.shape)
    expanded_zz = cov_zz.expand(*batch_shape, *cov_zz.shape)

    lhs = eye_s + weights.unsqueeze(-1) * expanded_cov
    rhs = weights.unsqueeze(-1) * expanded_sz
    solved = torch.linalg.solve(lhs, rhs)
    sigma_z = expanded_zz - torch.matmul(expanded_zs, solved)
    sigma_z = 0.5 * (sigma_z + sigma_z.transpose(-1, -2))

    if squeeze_output:
        sigma_z = sigma_z.squeeze(-3)
    return sigma_z


def mdp_experimental_design_utility(
    cov_matrix: Tensor,
    noise_var: Tensor,
    visitation: Tensor,
    target_indices: Tensor | None = None,
    criterion: str = "a_optimal",
    jitter: float = 1e-6,
) -> Tensor:
    r"""Evaluate concave experimental-design utility :math:`U(\eta)`.

    Supports both criteria studied in [mutny2023markovdesign]_ and
    [folch2024transitionmdp]_:
    - ``"a_optimal"``: :math:`U_A(\eta) = -\sum_{m=1}^{M} \operatorname{tr}(
      \Sigma_{\mathcal{Z}_t}^{(m)}(\eta))`.
    - ``"d_optimal"``: :math:`U_D(\eta) = -\frac{1}{2} \sum_{m=1}^{M}
      \log\det(\Sigma_{\mathcal{Z}_t}^{(m)}(\eta) + \text{jitter} \cdot I)`.

    Args:
        cov_matrix: Latent covariance Tensor of shape ``(S, S)`` or ``(M, S, S)``.
        noise_var: Observation noise variance Tensor of shape ``(S,)`` or ``(M, S)``.
        visitation: Non-negative state visitation counts of shape ``(S,)`` or
            ``(..., S)``.
        target_indices: Optional 1D integer Tensor of target indices
            :math:`\mathcal{Z}_t`.
        criterion: Either ``"a_optimal"`` or ``"d_optimal"``.
        jitter: Positive diagonal jitter added inside ``"d_optimal"`` log-det.

    Returns:
        Utility Tensor of shape ``visitation.shape[:-1]`` (scalar for 1D
        ``visitation``).
    """
    if criterion not in SUPPORTED_CRITERIA:
        raise ValueError(
            f"Unsupported `criterion`={criterion!r}. Must be one of "
            f"{SUPPORTED_CRITERIA}."
        )
    if jitter <= 0.0:
        raise ValueError(f"`jitter` must be > 0; got {jitter}.")

    cov_3d = cov_matrix.unsqueeze(0) if cov_matrix.ndim == 2 else cov_matrix
    sigma_z = compute_posterior_covariance_woodbury(
        cov_matrix=cov_3d,
        noise_var=noise_var,
        visitation=visitation,
        target_indices=target_indices,
    )  # (..., M, Z, Z)

    if criterion == "a_optimal":
        traces = torch.diagonal(sigma_z, dim1=-2, dim2=-1).sum(dim=-1)  # (..., M)
        return -traces.sum(dim=-1)

    num_z = sigma_z.shape[-1]
    eye_z = torch.eye(num_z, dtype=sigma_z.dtype, device=sigma_z.device)
    _, logabsdet = torch.linalg.slogdet(sigma_z + jitter * eye_z)
    return -0.5 * logabsdet.sum(dim=-1)


class PolicyMixture:
    r"""Frank-Wolfe mixture of finite-horizon deterministic Markov policies."""

    def __init__(
        self,
        policies: Tensor,
        weights: Tensor,
        visitation: Tensor,
        utility: float,
        duality_gap: float,
        potential_maximizers: Tensor,
        utilities_history: list[float],
        gaps_history: list[float],
    ) -> None:
        r"""Initialize a ``PolicyMixture`` solution container.

        Args:
            policies: A ``K x H x S``-dim ``torch.long`` Tensor of deterministic
                Markov policies collected by Frank-Wolfe.
            weights: A ``K``-dim Tensor of non-negative mixture weights summing to 1.
            visitation: Expected state-visitation counts :math:`\eta \in
                \mathbb{R}^S_{\ge 0}` induced by the mixture (summing to ``H``).
            utility: Objective value :math:`U(\eta_{\text{past}} + \eta)` achieved
                by the mixture.
            duality_gap: Final Frank-Wolfe linearized duality gap certificate.
            potential_maximizers: 1D ``torch.long`` Tensor of state indices in
                :math:`\mathcal{Z}_t`.
            utilities_history: Sequence of utility values across FW iterations.
            gaps_history: Sequence of FW duality gaps across FW iterations.
        """
        self.policies = policies
        self.weights = weights
        self.visitation = visitation
        self.utility = float(utility)
        self.duality_gap = float(duality_gap)
        self.potential_maximizers = potential_maximizers
        self.utilities_history = utilities_history
        self.gaps_history = gaps_history

    def sample_policy(self, generator: torch.Generator | None = None) -> Tensor:
        r"""Sample a single deterministic policy ``(H, S)`` from the mixture.

        Args:
            generator: Optional PyTorch random number generator.

        Returns:
            A ``H x S``-dim ``torch.long`` Tensor representing the sampled policy.
        """
        idx = torch.multinomial(self.weights, num_samples=1, generator=generator).item()
        return self.policies[int(idx)]

    def to_markov_policy(self, mdp: TransitionMDP, initial_state: int) -> Tensor:
        r"""Convert the episodic policy mixture into an equivalent Markov policy.

        Constructs the non-stationary randomized Markov policy
        :math:`\pi_{\text{markov}}(a \mid h, s)` whose state-action occupancy
        matches :math:`\sum_{k=1}^{K} w_k d_h^{\pi^{(k)}}(s, a)` from
        ``initial_state``, thereby inducing the exact same state-visitation
        distribution ``self.visitation``.

        Args:
            mdp: The ``TransitionMDP`` on which the policies are defined.
            initial_state: Starting state index :math:`s_0`.

        Returns:
            A ``H x S x A``-dim Tensor of Markov action probabilities.
        """
        s0 = mdp._validate_state_index(initial_state, "initial_state")
        num_policies, plan_horizon, num_states = self.policies.shape
        num_actions = mdp.num_actions
        occupancy = torch.zeros(
            (plan_horizon, num_states, num_actions),
            dtype=mdp.dtype,
            device=mdp.device,
        )
        state_range = torch.arange(num_states, device=mdp.device)

        for k in range(num_policies):
            w_k = float(self.weights[k].item())
            if w_k <= 0.0:
                continue
            p_curr = torch.zeros(num_states, dtype=mdp.dtype, device=mdp.device)
            p_curr[s0] = 1.0
            pol_k = self.policies[k]
            for h in range(plan_horizon):
                act_h = pol_k[h]
                occupancy[h, state_range, act_h] += w_k * p_curr
                trans_h = mdp.transition_probs[state_range, act_h]
                p_curr = p_curr @ trans_h

        state_mass = occupancy.sum(dim=-1, keepdim=True)
        # Default fallback for unreachable (h, s): first valid action per state
        first_valid = mdp.action_mask.to(dtype=torch.long).argmax(dim=-1)
        fallback = torch.zeros(
            (num_states, num_actions), dtype=mdp.dtype, device=mdp.device
        )
        fallback[state_range, first_valid] = 1.0

        return torch.where(
            state_mass > 0.0,
            occupancy / state_mass.clamp_min(1e-30),
            fallback.unsqueeze(0).expand_as(occupancy),
        )

    def sample_trajectory(
        self,
        mdp: TransitionMDP,
        initial_state: int,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Sample an episode trajectory from the policy mixture.

        Samples a deterministic policy :math:`\pi^{(k)} \sim \text{Categorical}(w)`
        at the start of the episode and rolls out ``H`` Markov transitions from
        ``initial_state``.

        Args:
            mdp: The ``TransitionMDP`` governing state transitions.
            initial_state: Starting state index :math:`s_0`.
            generator: Optional PyTorch random number generator.

        Returns:
            A 2-tuple ``(trajectory_indices, actions)`` of ``torch.long`` Tensors
            with shapes ``(H + 1,)`` (starting with ``initial_state``) and ``(H,)``.
        """
        s_curr = mdp._validate_state_index(initial_state, "initial_state")
        pol = self.sample_policy(generator=generator)
        plan_horizon = pol.shape[0]
        traj = [s_curr]
        actions = []

        for h in range(plan_horizon):
            act = int(pol[h, s_curr].item())
            probs = mdp.transition_probs[s_curr, act]
            s_curr = int(
                torch.multinomial(probs, num_samples=1, generator=generator).item()
            )
            actions.append(act)
            traj.append(s_curr)

        return (
            torch.tensor(traj, dtype=torch.long, device=mdp.device),
            torch.tensor(actions, dtype=torch.long, device=mdp.device),
        )


class RecedingHorizonResult:
    r"""Trajectory result produced by history-dependent receding-horizon execution."""

    def __init__(
        self,
        initial_state: int,
        trajectory_indices: Tensor,
        visited_indices: Tensor,
        visited_states: Tensor,
        actions: Tensor,
        visitation_counts: Tensor,
        mixtures: list[PolicyMixture],
    ) -> None:
        r"""Initialize a ``RecedingHorizonResult`` container.

        Args:
            initial_state: Starting state index :math:`s_0`.
            trajectory_indices: ``(H + 1,)``-dim ``torch.long`` Tensor
                ``[s_0, s_1, ..., s_H]`` of full state indices along the episode.
            visited_indices: ``(H,)``-dim ``torch.long`` Tensor ``[s_1, ..., s_H]``
                of the ``H`` queried states.
            visited_states: ``(H, d)``-dim Tensor of coordinates for
                ``visited_indices``.
            actions: ``(H,)``-dim ``torch.long`` Tensor of executed actions.
            visitation_counts: ``(S,)``-dim Tensor of empirical visit counts across
                ``visited_indices`` (summing to ``H``).
            mixtures: List of ``H`` ``PolicyMixture`` objects replanned at each step.
        """
        self.initial_state = int(initial_state)
        self.trajectory_indices = trajectory_indices
        self.visited_indices = visited_indices
        self.visited_states = visited_states
        self.actions = actions
        self.visitation_counts = visitation_counts
        self.mixtures = mixtures


class TransitionConstrainedMDPBO(AcquisitionFunction):
    r"""Transition-Constrained BO via MDPs and Frank-Wolfe Policy Design.

    Implements the policy-space acquisition optimization and receding-horizon
    planning framework of [folch2024transitionmdp]_, extending the Markov-chain
    experimental design method of [mutny2023markovdesign]_ to single- and
    multi-objective Bayesian optimization under state transition constraints.
    """

    def __init__(
        self,
        model: Model,
        mdp: TransitionMDP,
        criterion: str = "a_optimal",
        beta: float = 4.0,
        num_fw_iters: int = 25,
        step_size_rule: str = "line_search",
        fw_tol: float = 1e-7,
        jitter: float = 1e-6,
        observation_noise: float | Tensor | None = None,
    ) -> None:
        r"""Initialize ``TransitionConstrainedMDPBO``.

        Args:
            model: A fitted single-output or multi-output BoTorch ``Model`` (e.g.,
                ``SingleTaskGP`` or ``ModelListGP``).
            mdp: A ``TransitionMDP`` specifying discrete states, feasible Markov
                transitions, and planning horizon ``H``.
            criterion: Concave experimental design objective on :math:`\mathcal{Z}_t`,
                either ``"a_optimal"`` or ``"d_optimal"``.
            beta: Confidence bound parameter :math:`\beta \ge 0` defining the
                potential maximizers set :math:`\mathcal{Z}_t`.
            num_fw_iters: Maximum number of Frank-Wolfe conditional gradient
                iterations (>= 0).
            step_size_rule: Step-size strategy for mixture updates: ``"line_search"``
                (exact 1D concave golden-section line search over ``[0, 1]``) or
                ``"classic"`` (:math:`\gamma_t = 2 / (t + 2)`).
            fw_tol: Early-stopping threshold on the Frank-Wolfe duality gap.
            jitter: Positive diagonal jitter for ``"d_optimal"`` log-det stability.
            observation_noise: Optional explicit observation noise variance override.
                If ``None``, extracted automatically from ``model``.
        """
        super().__init__(model=model)
        if criterion not in SUPPORTED_CRITERIA:
            raise ValueError(
                f"Unsupported `criterion`={criterion!r}. Expected one of "
                f"{SUPPORTED_CRITERIA}."
            )
        if step_size_rule not in SUPPORTED_STEP_SIZE_RULES:
            raise ValueError(
                f"Unsupported `step_size_rule`={step_size_rule!r}. Expected one of "
                f"{SUPPORTED_STEP_SIZE_RULES}."
            )
        if beta < 0.0:
            raise ValueError(f"`beta` must be >= 0; got {beta}.")
        if num_fw_iters < 0:
            raise ValueError(f"`num_fw_iters` must be >= 0; got {num_fw_iters}.")
        if fw_tol < 0.0:
            raise ValueError(f"`fw_tol` must be >= 0; got {fw_tol}.")
        if jitter <= 0.0:
            raise ValueError(f"`jitter` must be > 0; got {jitter}.")

        self.mdp = mdp
        self.criterion = criterion
        self.beta = float(beta)
        self.num_fw_iters = int(num_fw_iters)
        self.step_size_rule = step_size_rule
        self.fw_tol = float(fw_tol)
        self.jitter = float(jitter)
        self._observation_noise_override = observation_noise

        self.cov_matrix: Tensor
        self.noise_var: Tensor
        self.potential_maximizers: Tensor
        self.update_model(model=model, observation_noise=observation_noise)

    def update_model(
        self,
        model: Model,
        observation_noise: float | Tensor | None = None,
    ) -> None:
        r"""Refresh cached GP covariances, noise variances, and potential maximizers.

        Args:
            model: Updated fitted BoTorch ``Model``.
            observation_noise: Optional observation noise variance override. If
                omitted, reuses the override supplied at initialization (if any).
        """
        self.model = model
        obs_noise = (
            self._observation_noise_override
            if observation_noise is None
            else observation_noise
        )
        self.cov_matrix, self.noise_var = _extract_posterior_cov_and_noise(
            model=model,
            states=self.mdp.states,
            observation_noise=obs_noise,
        )
        self.potential_maximizers = compute_potential_maximizers(
            model=model,
            states=self.mdp.states,
            beta=self.beta,
        )

    def posterior_covariance(self, visitation: Tensor) -> Tensor:
        r"""Compute posterior covariance :math:`\Sigma_{\mathcal{Z}_t}(\eta)`.

        Args:
            visitation: Non-negative state-visitation Tensor of shape ``(S,)`` or
                ``(..., S)``.

        Returns:
            Posterior covariance Tensor of shape ``(..., M, Z, Z)`` (or
            ``(M, Z, Z)`` for 1D ``visitation``).
        """
        return compute_posterior_covariance_woodbury(
            cov_matrix=self.cov_matrix,
            noise_var=self.noise_var,
            visitation=visitation,
            target_indices=self.potential_maximizers,
        )

    def evaluate_utility(
        self,
        visitation: Tensor,
        past_visitation: Tensor | None = None,
    ) -> Tensor:
        r"""Evaluate concave design utility :math:`U(\eta_{\text{past}} + \eta)`.

        Args:
            visitation: Planned state-visitation allocation of shape ``(S,)`` or
                ``(..., S)``.
            past_visitation: Optional ``(S,)`` Tensor of counts already visited
                earlier in the current episode.

        Returns:
            Utility value Tensor of shape ``visitation.shape[:-1]``.
        """
        total_eta = (
            visitation
            if past_visitation is None
            else visitation
            + past_visitation.to(device=visitation.device, dtype=visitation.dtype)
        )
        return mdp_experimental_design_utility(
            cov_matrix=self.cov_matrix,
            noise_var=self.noise_var,
            visitation=total_eta,
            target_indices=self.potential_maximizers,
            criterion=self.criterion,
            jitter=self.jitter,
        )

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate experimental-design utility on state-visitation allocations ``X``.

        Args:
            X: State-visitation allocation Tensor :math:`\eta` of shape ``(S,)``
                or ``(..., S)``.

        Returns:
            Utility Tensor of shape ``X.shape[:-1]``.
        """
        return self.evaluate_utility(visitation=X)

    def utility_and_gradient(
        self,
        visitation: Tensor,
        past_visitation: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Compute :math:`U(\eta_{\text{past}} + \eta)` and :math:`\nabla_\eta U`.

        Args:
            visitation: ``(S,)``-dim non-negative state-visitation allocation.
            past_visitation: Optional ``(S,)``-dim non-negative past visitation counts.

        Returns:
            A 2-tuple ``(utility, gradient)`` where ``utility`` is a scalar Tensor
            and ``gradient`` has shape ``(S,)``.
        """
        if visitation.ndim != 1 or visitation.shape[0] != self.mdp.num_states:
            raise ValueError(
                f"`visitation` must have shape ({self.mdp.num_states},); "
                f"got {visitation.shape}."
            )
        base = visitation.detach().to(
            device=self.cov_matrix.device, dtype=self.cov_matrix.dtype
        )
        if past_visitation is not None:
            past = past_visitation.detach().to(
                device=self.cov_matrix.device, dtype=self.cov_matrix.dtype
            )
            if past.shape != base.shape or (past < -1e-6).any():
                raise ValueError(
                    "`past_visitation` must have shape (S,) and non-negative entries."
                )
            base = base + past.clamp_min(0.0)

        with torch.enable_grad():
            eta_var = base.clamp_min(0.0).requires_grad_(True)
            val = mdp_experimental_design_utility(
                cov_matrix=self.cov_matrix,
                noise_var=self.noise_var,
                visitation=eta_var,
                target_indices=self.potential_maximizers,
                criterion=self.criterion,
                jitter=self.jitter,
            )
            (grad,) = torch.autograd.grad(val, eta_var)
        return val.detach(), grad.detach()

    def _line_search_step_size(
        self,
        eta_curr: Tensor,
        eta_target: Tensor,
        past_visitation: Tensor | None,
        num_steps: int = 30,
    ) -> float:
        r"""Maximize concave 1D objective along segment ``[eta_curr, eta_target]``."""
        inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
        inv_phi2 = (3.0 - math.sqrt(5.0)) / 2.0
        low = 0.0
        high = 1.0
        c = low + inv_phi2 * (high - low)
        d = low + inv_phi * (high - low)

        with torch.no_grad():
            u0 = float(self.evaluate_utility(eta_curr, past_visitation).item())
            u1 = float(self.evaluate_utility(eta_target, past_visitation).item())
            uc = float(
                self.evaluate_utility(
                    (1.0 - c) * eta_curr + c * eta_target, past_visitation
                ).item()
            )
            ud = float(
                self.evaluate_utility(
                    (1.0 - d) * eta_curr + d * eta_target, past_visitation
                ).item()
            )

            for _ in range(num_steps):
                if uc >= ud:
                    high = d
                    d = c
                    ud = uc
                    c = low + inv_phi2 * (high - low)
                    uc = float(
                        self.evaluate_utility(
                            (1.0 - c) * eta_curr + c * eta_target, past_visitation
                        ).item()
                    )
                else:
                    low = c
                    c = d
                    uc = ud
                    d = low + inv_phi * (high - low)
                    ud = float(
                        self.evaluate_utility(
                            (1.0 - d) * eta_curr + d * eta_target, past_visitation
                        ).item()
                    )

            g_mid = 0.5 * (low + high)
            u_mid = float(
                self.evaluate_utility(
                    (1.0 - g_mid) * eta_curr + g_mid * eta_target, past_visitation
                ).item()
            )

        best_gamma = 0.0
        best_val = u0
        if u_mid > best_val:
            best_val = u_mid
            best_gamma = g_mid
        if u1 > best_val:
            best_gamma = 1.0
        return best_gamma

    def optimize_policy_mixture(
        self,
        current_state: int,
        past_visitation: Tensor | None = None,
        horizon: int | None = None,
    ) -> PolicyMixture:
        r"""Optimize a policy mixture via Frank-Wolfe in MDP state-visitation space.

        Starts from the greedy policy optimal for the initial linearized gradient
        :math:`\nabla_\eta U(\eta_{\text{past}})` and performs ``self.num_fw_iters``
        conditional gradient updates using exact Bellman DP as the linear
        maximization oracle.

        Args:
            current_state: Current state index :math:`s_{\text{curr}} \in
                \{0, \dots, S-1\}`.
            past_visitation: Optional ``(S,)``-dim Tensor of state counts already
                visited in the current episode (for receding-horizon replanning).
            horizon: Optional planning horizon override ``H' >= 1``. Defaults to
                ``self.mdp.horizon``.

        Returns:
            A ``PolicyMixture`` containing the optimized policies, mixture weights,
            induced state-visitation measure, utility, and FW duality gap history.
        """
        s_curr = self.mdp._validate_state_index(current_state, "current_state")
        plan_horizon = self.mdp.horizon if horizon is None else int(horizon)
        if plan_horizon < 1:
            raise ValueError(f"`horizon` must be >= 1; got {plan_horizon}.")

        zero_eta = torch.zeros(
            self.mdp.num_states,
            dtype=self.cov_matrix.dtype,
            device=self.cov_matrix.device,
        )
        _, grad_init = self.utility_and_gradient(
            visitation=zero_eta, past_visitation=past_visitation
        )
        _, pi_0, eta_0 = self.mdp.solve_linear_mdp(
            state_rewards=grad_init,
            initial_state=s_curr,
            horizon=plan_horizon,
            normalize=False,
        )

        policies: list[Tensor] = [pi_0]
        weights: list[float] = [1.0]
        eta_k = eta_0.to(dtype=self.cov_matrix.dtype)

        utilities_history: list[float] = []
        gaps_history: list[float] = []
        final_gap = 0.0

        for k in range(self.num_fw_iters):
            u_k_tensor, r_k = self.utility_and_gradient(
                visitation=eta_k, past_visitation=past_visitation
            )
            u_k = float(u_k_tensor.item())
            _, pi_greedy, eta_greedy_raw = self.mdp.solve_linear_mdp(
                state_rewards=r_k,
                initial_state=s_curr,
                horizon=plan_horizon,
                normalize=False,
            )
            eta_greedy = eta_greedy_raw.to(dtype=self.cov_matrix.dtype)
            gap = float(torch.dot(r_k, eta_greedy - eta_k).item())
            gap = max(gap, 0.0)

            utilities_history.append(u_k)
            gaps_history.append(gap)
            final_gap = gap

            if gap <= self.fw_tol:
                break

            if self.step_size_rule == "classic":
                # Initial LMO policy corresponds to FW step 0 (gamma_0 = 1);
                # subsequent mixture steps use gamma_t = 2 / (t + 2) for t = k + 1.
                gamma = 2.0 / float(k + 3)
            else:
                gamma = self._line_search_step_size(
                    eta_curr=eta_k,
                    eta_target=eta_greedy,
                    past_visitation=past_visitation,
                )

            if gamma <= 0.0:
                break
            if gamma >= 1.0:
                policies = [pi_greedy]
                weights = [1.0]
                eta_k = eta_greedy
            else:
                weights = [(1.0 - gamma) * w for w in weights]
                merged = False
                for idx, existing_pi in enumerate(policies):
                    if torch.equal(existing_pi, pi_greedy):
                        weights[idx] += gamma
                        merged = True
                        break
                if not merged:
                    policies.append(pi_greedy)
                    weights.append(gamma)
                eta_k = (1.0 - gamma) * eta_k + gamma * eta_greedy

        final_u_tensor, final_r = self.utility_and_gradient(
            visitation=eta_k, past_visitation=past_visitation
        )
        final_u = float(final_u_tensor.item())
        _, _, eta_last_raw = self.mdp.solve_linear_mdp(
            state_rewards=final_r,
            initial_state=s_curr,
            horizon=plan_horizon,
            normalize=False,
        )
        final_gap = max(
            0.0,
            float(
                torch.dot(
                    final_r, eta_last_raw.to(dtype=self.cov_matrix.dtype) - eta_k
                ).item()
            ),
        )
        utilities_history.append(final_u)
        gaps_history.append(final_gap)

        # Prune negligible mixture components and normalize weights
        kept_policies: list[Tensor] = []
        kept_weights: list[float] = []
        for pol, w in zip(policies, weights):
            if w > 1e-12:
                kept_policies.append(pol)
                kept_weights.append(w)

        weights_tensor = torch.tensor(
            kept_weights, dtype=self.mdp.dtype, device=self.mdp.device
        )
        weights_tensor = weights_tensor / weights_tensor.sum()
        policies_tensor = torch.stack(kept_policies, dim=0)

        return PolicyMixture(
            policies=policies_tensor,
            weights=weights_tensor,
            visitation=eta_k.to(dtype=self.mdp.dtype),
            utility=final_u,
            duality_gap=final_gap,
            potential_maximizers=self.potential_maximizers,
            utilities_history=utilities_history,
            gaps_history=gaps_history,
        )

    def select_next_state(
        self,
        current_state: int,
        past_visitation: Tensor | None = None,
        remaining_horizon: int | None = None,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[int, int, PolicyMixture]:
        r"""Select the next feasible transition action and state via FW replanning.

        Args:
            current_state: Current state index :math:`s_t`.
            past_visitation: Optional ``(S,)``-dim Tensor of counts already visited
                in the episode.
            remaining_horizon: Remaining steps ``H - t >= 1`` in the episode.
                Defaults to ``self.mdp.horizon``.
            deterministic: If ``True``, execute the highest-weight stage-0 action
                and modal next state; if ``False``, sample from the mixture's
                stage-0 action distribution and Markov transition kernel.
            generator: Optional PyTorch random number generator when
                ``deterministic=False``.

        Returns:
            A 3-tuple ``(next_state, action, mixture)``.
        """
        s_curr = self.mdp._validate_state_index(current_state, "current_state")
        mixture = self.optimize_policy_mixture(
            current_state=s_curr,
            past_visitation=past_visitation,
            horizon=remaining_horizon,
        )

        # Aggregate stage-0 action probabilities at current_state across mixture
        action_probs = torch.zeros(
            self.mdp.num_actions, dtype=self.mdp.dtype, device=self.mdp.device
        )
        stage0_actions = mixture.policies[:, 0, s_curr]
        action_probs.scatter_add_(0, stage0_actions, mixture.weights)

        if deterministic:
            masked_probs = action_probs.masked_fill(~self.mdp.action_mask[s_curr], -1.0)
            action = int(masked_probs.argmax().item())
            next_state = int(self.mdp.transition_probs[s_curr, action].argmax().item())
        else:
            action = int(
                torch.multinomial(
                    action_probs, num_samples=1, generator=generator
                ).item()
            )
            trans_probs = self.mdp.transition_probs[s_curr, action]
            next_state = int(
                torch.multinomial(
                    trans_probs, num_samples=1, generator=generator
                ).item()
            )

        return next_state, action, mixture

    def plan_receding_horizon_trajectory(
        self,
        initial_state: int,
        horizon: int | None = None,
        past_visitation: Tensor | None = None,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> RecedingHorizonResult:
        r"""Execute history-dependent receding-horizon planning over an episode.

        At episode step ``h = 0, ..., H - 1``, replans a Frank-Wolfe policy mixture
        from the realized state :math:`s_h` over remaining horizon ``H - h``
        conditioned on empirical past counts :math:`\eta_{\text{past}}` of states
        already visited in the episode, executes one feasible transition
        :math:`s_h \to s_{h+1}`, and increments :math:`\eta_{\text{past}}(s_{h+1})`.

        Args:
            initial_state: Starting state index :math:`s_0`.
            horizon: Optional episode horizon ``H >= 1``. Defaults to
                ``self.mdp.horizon``.
            past_visitation: Optional ``(S,)``-dim initial past visitation counts
                prior to step 0.
            deterministic: If ``True``, select modal actions/transitions at each
                step; if ``False``, sample stochastically.
            generator: Optional PyTorch random number generator.

        Returns:
            A ``RecedingHorizonResult`` containing the full trajectory, queried
            state coordinates, empirical visit counts, and step-by-step mixtures.
        """
        s_curr = self.mdp._validate_state_index(initial_state, "initial_state")
        total_horizon = self.mdp.horizon if horizon is None else int(horizon)
        if total_horizon < 1:
            raise ValueError(f"`horizon` must be >= 1; got {total_horizon}.")

        if past_visitation is None:
            eta_past = torch.zeros(
                self.mdp.num_states, dtype=self.mdp.dtype, device=self.mdp.device
            )
        else:
            eta_past = (
                past_visitation.detach()
                .clone()
                .to(device=self.mdp.device, dtype=self.mdp.dtype)
            )
            if eta_past.shape != (self.mdp.num_states,) or (eta_past < -1e-6).any():
                raise ValueError(
                    "`past_visitation` must have shape (S,) and non-negative entries."
                )

        episode_counts = torch.zeros(
            self.mdp.num_states, dtype=self.mdp.dtype, device=self.mdp.device
        )
        traj_indices = [s_curr]
        actions = []
        mixtures: list[PolicyMixture] = []

        for step in range(total_horizon):
            rem_horizon = total_horizon - step
            s_next, act, mix = self.select_next_state(
                current_state=s_curr,
                past_visitation=eta_past,
                remaining_horizon=rem_horizon,
                deterministic=deterministic,
                generator=generator,
            )
            traj_indices.append(s_next)
            actions.append(act)
            mixtures.append(mix)
            eta_past[s_next] = eta_past[s_next] + 1.0
            episode_counts[s_next] = episode_counts[s_next] + 1.0
            s_curr = s_next

        traj_tensor = torch.tensor(
            traj_indices, dtype=torch.long, device=self.mdp.device
        )
        visited_tensor = traj_tensor[1:]
        visited_coords = self.mdp.states.index_select(0, visited_tensor)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.mdp.device)

        return RecedingHorizonResult(
            initial_state=initial_state,
            trajectory_indices=traj_tensor,
            visited_indices=visited_tensor,
            visited_states=visited_coords,
            actions=actions_tensor,
            visitation_counts=episode_counts,
            mixtures=mixtures,
        )
