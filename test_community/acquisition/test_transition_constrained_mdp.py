#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from itertools import product

import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.testing import BotorchTestCase
from botorch_community.acquisition.transition_constrained_mdp import (
    _extract_cov_blocks_from_posterior,
    compute_posterior_covariance_woodbury,
    compute_potential_maximizers,
    mdp_experimental_design_utility,
    TransitionConstrainedMDPBO,
    TransitionMDP,
)


class TestTransitionConstrainedMDP(BotorchTestCase):
    def _build_single_task_gp(
        self,
        dtype: torch.dtype,
        dim: int = 1,
        num_outputs: int = 1,
        standardize: bool = True,
    ) -> SingleTaskGP:
        torch.manual_seed(42)
        tkwargs = {"device": self.device, "dtype": dtype}
        train_X = torch.linspace(0.0, 1.0, 6, **tkwargs).unsqueeze(-1)
        if dim > 1:
            extra = torch.rand(6, dim - 1, **tkwargs)
            train_X = torch.cat([train_X, extra], dim=-1)
        y1 = torch.sin(3.0 * train_X[:, :1])
        if num_outputs == 1:
            train_Y = y1
        else:
            y2 = torch.cos(3.0 * train_X[:, :1])
            train_Y = torch.cat([y1, y2], dim=-1)
        outcome_transform = Standardize(m=num_outputs) if standardize else None
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=outcome_transform,
        )
        model.eval()
        return model

    def _build_model_list_gp(self, dtype: torch.dtype, dim: int = 2) -> ModelListGP:
        torch.manual_seed(123)
        tkwargs = {"device": self.device, "dtype": dtype}
        train_X1 = torch.rand(6, dim, **tkwargs)
        train_Y1 = torch.sin(2.5 * train_X1[:, :1]) + 0.5 * train_X1[:, 1:2]
        train_X2 = torch.rand(7, dim, **tkwargs)
        train_Y2 = torch.cos(2.5 * train_X2[:, :1]) - 0.3 * train_X2[:, 1:2]
        m1 = SingleTaskGP(train_X1, train_Y1)
        m2 = SingleTaskGP(train_X2, train_Y2)
        model = ModelListGP(m1, m2)
        model.eval()
        return model

    def test_transition_mdp_step_constraints_and_validation(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            states_1d = torch.linspace(0.0, 1.0, 5, **tkwargs).unsqueeze(-1)
            mdp = TransitionMDP.from_step_constraints(
                states=states_1d, max_step=0.26, p=2.0, horizon=4
            )
            self.assertEqual(mdp.num_states, 5)
            self.assertEqual(mdp.dim, 1)
            self.assertEqual(mdp.horizon, 4)
            self.assertEqual(mdp.device, self.device)
            self.assertEqual(mdp.dtype, dtype)

            # Endpoints have 2 neighbors (self + 1 neighbor); interior states have 3
            self.assertEqual(mdp.num_actions, 3)
            self.assertEqual(int(mdp.action_mask[0].sum().item()), 2)
            self.assertEqual(int(mdp.action_mask[2].sum().item()), 3)
            self.assertAllClose(
                mdp.transition_probs.sum(dim=-1),
                torch.ones(5, 3, **tkwargs),
            )

            # Zero max_step still includes self-loops for every state
            mdp_zero = TransitionMDP.from_step_constraints(
                states=states_1d, max_step=0.0, horizon=3
            )
            self.assertEqual(mdp_zero.num_actions, 1)
            _, _, eta_zero = mdp_zero.solve_linear_mdp(
                state_rewards=torch.arange(5, **tkwargs), initial_state=2
            )
            expected_zero = torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0], **tkwargs)
            self.assertAllClose(eta_zero, expected_zero)

            # Test per-dimension 1D max_step vector (including a frozen 0-step coord)
            states_2d = torch.tensor(
                [[0.0, 0.0], [0.2, 0.0], [0.0, 0.2], [0.2, 0.2]], **tkwargs
            )
            mdp_vec = TransitionMDP.from_step_constraints(
                states=states_2d,
                max_step=torch.tensor([0.25, 0.0], **tkwargs),
                p=float("inf"),
                horizon=2,
            )
            # Only transitions preserving x_2 == 0 or x_2 == 0.2 are allowed
            self.assertEqual(int(mdp_vec.action_mask[0].sum().item()), 2)

            # Test 1D weighted metric (p=2 and p=inf) and 2D Mahalanobis metric
            w_metric = torch.tensor([1.0, 4.0], **tkwargs)
            mdp_w2 = TransitionMDP.from_step_constraints(
                states=states_2d, max_step=0.25, p=2.0, metric=w_metric, horizon=2
            )
            self.assertEqual(int(mdp_w2.action_mask[0].sum().item()), 2)
            mdp_winf = TransitionMDP.from_step_constraints(
                states=states_2d,
                max_step=0.25,
                p=float("inf"),
                metric=w_metric,
                horizon=2,
            )
            self.assertEqual(int(mdp_winf.action_mask[0].sum().item()), 2)

            m_mat = torch.diag(torch.tensor([1.0, 25.0], **tkwargs))
            mdp_maha = TransitionMDP.from_step_constraints(
                states=states_2d, max_step=0.25, metric=m_mat, horizon=2
            )
            self.assertEqual(int(mdp_maha.action_mask[0].sum().item()), 2)

        # Validation errors
        s_valid = torch.zeros(3, 2, device=self.device, dtype=torch.double)
        p_valid = torch.eye(3, device=self.device, dtype=torch.double).unsqueeze(1)
        with self.assertRaises(ValueError):
            TransitionMDP(
                states=torch.zeros(3, device=self.device), transition_probs=p_valid
            )
        with self.assertRaises(ValueError):
            TransitionMDP(
                states=torch.zeros(3, 2, device=self.device, dtype=torch.long),
                transition_probs=p_valid,
            )
        with self.assertRaises(ValueError):
            TransitionMDP(
                states=s_valid,
                transition_probs=torch.zeros(
                    3, 1, 4, device=self.device, dtype=torch.double
                ),
            )
        with self.assertRaises(ValueError):
            TransitionMDP(states=s_valid, transition_probs=p_valid, horizon=0)
        with self.assertRaises(ValueError):
            TransitionMDP(
                states=s_valid,
                transition_probs=p_valid,
                action_mask=torch.zeros(3, 1, dtype=torch.bool, device=self.device),
            )
        with self.assertRaises(ValueError):
            TransitionMDP(states=s_valid, transition_probs=2.0 * p_valid)
        with self.assertRaises(ValueError):
            bad_p = torch.tensor(
                [[[-0.5, 1.5, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]],
                device=self.device,
                dtype=torch.double,
            )
            TransitionMDP(states=s_valid, transition_probs=bad_p)
        with self.assertRaises(ValueError):
            TransitionMDP.from_step_constraints(states=s_valid, max_step=-0.1)
        with self.assertRaises(ValueError):
            TransitionMDP.from_step_constraints(states=s_valid, max_step=0.1, p=0.0)

    def test_solve_linear_mdp_exact_against_brute_force(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            atol = 1e-5 if dtype == torch.float else 1e-12

            # 1. Deterministic 4-state chain MDP: compare DP vs all feasible paths
            states = torch.linspace(0.0, 3.0, 4, **tkwargs).unsqueeze(-1)
            mdp_det = TransitionMDP.from_step_constraints(
                states=states, max_step=1.05, horizon=4
            )
            rewards_1d = torch.tensor([0.2, -0.4, 1.1, 2.5], **tkwargs)
            values_det, pi_det, eta_det = mdp_det.solve_linear_mdp(
                state_rewards=rewards_1d, initial_state=0
            )
            self.assertEqual(values_det.shape, (4, 4))
            self.assertEqual(pi_det.shape, (4, 4))
            self.assertEqual(eta_det.shape, (4,))
            self.assertAllClose(eta_det.sum(), torch.tensor(4.0, **tkwargs), atol=atol)
            # Bellman-visitation duality: V[0, s0] == <R, eta>
            self.assertAllClose(
                values_det[0, 0], torch.dot(rewards_1d, eta_det), atol=atol
            )

            # Brute-force all deterministic paths of length H=4 starting from s0=0
            best_det = float("-inf")
            for s1 in range(4):
                if abs(s1 - 0) > 1:
                    continue
                for s2 in range(4):
                    if abs(s2 - s1) > 1:
                        continue
                    for s3 in range(4):
                        if abs(s3 - s2) > 1:
                            continue
                        for s4 in range(4):
                            if abs(s4 - s3) > 1:
                                continue
                            total_r = float(
                                (
                                    rewards_1d[s1]
                                    + rewards_1d[s2]
                                    + rewards_1d[s3]
                                    + rewards_1d[s4]
                                ).item()
                            )
                            best_det = max(best_det, total_r)
            self.assertAllClose(
                values_det[0, 0], torch.tensor(best_det, **tkwargs), atol=atol
            )

            # 2. Stochastic 4-state, 2-action MDP with stagewise rewards (H=3)
            # Brute-force all (2^4)^3 = 4,096 non-stationary Markov policies!
            trans_stoch = torch.tensor(
                [
                    [[0.7, 0.3, 0.0, 0.0], [0.1, 0.6, 0.3, 0.0]],
                    [[0.2, 0.5, 0.3, 0.0], [0.0, 0.2, 0.5, 0.3]],
                    [[0.0, 0.3, 0.5, 0.2], [0.0, 0.0, 0.3, 0.7]],
                    [[0.0, 0.0, 0.4, 0.6], [0.1, 0.0, 0.1, 0.8]],
                ],
                **tkwargs,
            )
            mdp_stoch = TransitionMDP(
                states=states, transition_probs=trans_stoch, horizon=3
            )
            rewards_2d = torch.tensor(
                [
                    [0.5, 1.2, -0.3, 0.8],
                    [-0.2, 0.4, 1.7, 0.9],
                    [0.1, -0.5, 0.6, 2.3],
                ],
                **tkwargs,
            )
            values_st, pi_st, eta_st = mdp_stoch.solve_linear_mdp(
                state_rewards=rewards_2d, initial_state=1
            )
            self.assertAllClose(eta_st.sum(), torch.tensor(3.0, **tkwargs), atol=atol)
            _, _, d_norm = mdp_stoch.solve_linear_mdp(
                state_rewards=rewards_2d, initial_state=1, normalize=True
            )
            self.assertAllClose(d_norm.sum(), torch.tensor(1.0, **tkwargs), atol=atol)

            # Enumerate all 16 stage policies (each is a 4-tuple of {0, 1})
            stage_policies = list(product((0, 1), repeat=4))
            state_idx = torch.arange(4, device=self.device)
            stage_mats = [
                trans_stoch[state_idx, torch.tensor(sp, device=self.device)]
                for sp in stage_policies
            ]
            p0 = torch.tensor([0.0, 1.0, 0.0, 0.0], **tkwargs)
            best_stoch = float("-inf")
            for m0 in stage_mats:
                p1 = p0 @ m0
                r0 = torch.dot(p1, rewards_2d[0])
                for m1 in stage_mats:
                    p2 = p1 @ m1
                    r1 = r0 + torch.dot(p2, rewards_2d[1])
                    for m2 in stage_mats:
                        p3 = p2 @ m2
                        val = float((r1 + torch.dot(p3, rewards_2d[2])).item())
                        if val > best_stoch:
                            best_stoch = val

            self.assertAllClose(
                values_st[0, 1], torch.tensor(best_stoch, **tkwargs), atol=atol
            )

            # Verify stagewise visitation helper matches summed visitation
            stagewise = mdp_stoch.compute_policy_visitation(
                policy=pi_st, initial_state=1, return_stagewise=True
            )
            self.assertEqual(stagewise.shape, (3, 4))
            self.assertAllClose(stagewise.sum(dim=0), eta_st, atol=atol)

    def test_woodbury_covariance_matches_condition_on_observations(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            atol = 1e-4 if dtype == torch.float else 1e-11
            rtol = 1e-4 if dtype == torch.float else 1e-10

            states = torch.linspace(0.05, 0.95, 6, **tkwargs).unsqueeze(-1)
            mdp = TransitionMDP.from_step_constraints(
                states=states, max_step=0.25, horizon=5
            )
            # Integer allocation including 0s, 1s, and repeated 2s (sum = 5)
            counts = torch.tensor([2, 0, 1, 0, 2, 0], device=self.device)
            eta = counts.to(dtype=dtype)

            # Construct explicit repeated observation tensor X_obs of shape (5, d)
            obs_indices = torch.repeat_interleave(
                torch.arange(6, device=self.device), counts
            )

            # Test 1: Single-output SingleTaskGP (with and without Standardize)
            for standardize in (False, True):
                model_1d = self._build_single_task_gp(
                    dtype=dtype, dim=1, num_outputs=1, standardize=standardize
                )
                planner_1d = TransitionConstrainedMDPBO(
                    model=model_1d, mdp=mdp, beta=4.0
                )
                z_idx = planner_1d.potential_maximizers
                sigma_woodbury = planner_1d.posterior_covariance(eta)  # (1, Z, Z)

                X_obs = states.index_select(0, obs_indices)
                Y_obs = torch.zeros(X_obs.shape[0], 1, **tkwargs)
                cond_model = model_1d.condition_on_observations(X=X_obs, Y=Y_obs)
                cond_post = cond_model.posterior(
                    states.index_select(0, z_idx), observation_noise=False
                )
                sigma_exact = cond_post.distribution.covariance_matrix.unsqueeze(0)
                self.assertAllClose(sigma_woodbury, sigma_exact, atol=atol, rtol=rtol)

            # Test 2: Batched Multi-output SingleTaskGP (M = 2)
            model_mo = self._build_single_task_gp(
                dtype=dtype, dim=1, num_outputs=2, standardize=True
            )
            planner_mo = TransitionConstrainedMDPBO(model=model_mo, mdp=mdp, beta=4.0)
            z_mo = planner_mo.potential_maximizers
            sigma_woodbury_mo = planner_mo.posterior_covariance(eta)  # (2, Z, Z)

            X_obs = states.index_select(0, obs_indices)
            Y_obs_mo = torch.zeros(X_obs.shape[0], 2, **tkwargs)
            cond_mo = model_mo.condition_on_observations(X=X_obs, Y=Y_obs_mo)
            cond_post_mo = cond_mo.posterior(
                states.index_select(0, z_mo), observation_noise=False
            )
            exact_blocks_mo = torch.stack(
                _extract_cov_blocks_from_posterior(
                    cond_post_mo, num_states=z_mo.numel()
                ),
                dim=0,
            )
            self.assertAllClose(
                sigma_woodbury_mo, exact_blocks_mo, atol=atol, rtol=rtol
            )

            # Test 3: ModelListGP (M = 2 independent GPs in 2D)
            states_2d = torch.rand(6, 2, **tkwargs)
            mdp_2d = TransitionMDP.from_step_constraints(
                states=states_2d, max_step=0.5, horizon=5
            )
            model_list = self._build_model_list_gp(dtype=dtype, dim=2)
            planner_list = TransitionConstrainedMDPBO(
                model=model_list, mdp=mdp_2d, beta=4.0
            )
            z_list = planner_list.potential_maximizers
            sigma_woodbury_list = planner_list.posterior_covariance(eta)

            X_obs_2d = states_2d.index_select(0, obs_indices)
            Y_obs_list = torch.zeros(X_obs_2d.shape[0], 2, **tkwargs)
            cond_list = model_list.condition_on_observations(
                X=[X_obs_2d, X_obs_2d], Y=Y_obs_list
            )
            cond_post_list = cond_list.posterior(
                states_2d.index_select(0, z_list), observation_noise=False
            )
            exact_blocks_list = torch.stack(
                _extract_cov_blocks_from_posterior(
                    cond_post_list, num_states=z_list.numel()
                ),
                dim=0,
            )
            self.assertAllClose(
                sigma_woodbury_list, exact_blocks_list, atol=atol, rtol=rtol
            )

    def test_autograd_gradients_with_zeros_in_visitation(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        states = torch.linspace(0.0, 1.0, 5, **tkwargs).unsqueeze(-1)
        mdp = TransitionMDP.from_step_constraints(
            states=states, max_step=0.3, horizon=4
        )
        model = self._build_single_task_gp(dtype=torch.double, dim=1, num_outputs=1)

        # Allocation with exact zeros at indices 1, 3, 4
        eta = torch.tensor([2.5, 0.0, 1.5, 0.0, 0.0], **tkwargs)
        eps = 1e-7

        for criterion in ("a_optimal", "d_optimal"):
            planner = TransitionConstrainedMDPBO(
                model=model, mdp=mdp, criterion=criterion, jitter=1e-6
            )
            val, grad = planner.utility_and_gradient(eta)
            self.assertFalse(torch.isnan(val).any())
            self.assertFalse(torch.isnan(grad).any())
            self.assertTrue((grad > 0.0).all())

            # Compare against forward finite differences (valid at eta_s == 0 too!)
            fd_grad = torch.zeros_like(eta)
            for s in range(5):
                e_s = torch.zeros_like(eta)
                e_s[s] = eps
                u_plus = planner.evaluate_utility(eta + e_s)
                fd_grad[s] = (u_plus - val) / eps
            self.assertAllClose(grad, fd_grad, atol=1e-5, rtol=1e-5)

            # Test batched forward evaluation shape
            batch_eta = torch.stack([eta, 0.5 * eta], dim=0)
            batch_u = planner(batch_eta)
            self.assertEqual(batch_u.shape, (2,))
            self.assertAllClose(batch_u[0], val)

    def test_potential_maximizers_single_and_multi_objective(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            states = torch.linspace(0.0, 1.0, 8, **tkwargs).unsqueeze(-1)

            # Single-objective monotonicity in beta & exactness at beta=0
            model_so = self._build_single_task_gp(dtype=dtype, dim=1, num_outputs=1)
            z_b0 = compute_potential_maximizers(model=model_so, states=states, beta=0.0)
            z_b1 = compute_potential_maximizers(model=model_so, states=states, beta=1.0)
            z_b9 = compute_potential_maximizers(model=model_so, states=states, beta=9.0)

            self.assertTrue(z_b0.numel() >= 1)
            self.assertTrue(set(z_b0.tolist()).issubset(set(z_b1.tolist())))
            self.assertTrue(set(z_b1.tolist()).issubset(set(z_b9.tolist())))

            with torch.no_grad():
                mean_so = model_so.posterior(states).mean.squeeze(-1)
            self.assertIn(int(mean_so.argmax().item()), z_b0.tolist())

            # Multi-objective Pareto maximizers: at beta=0 equals exact Pareto front
            model_mo = self._build_single_task_gp(dtype=dtype, dim=1, num_outputs=2)
            z_mo_0 = compute_potential_maximizers(
                model=model_mo, states=states, beta=0.0
            )
            z_mo_4 = compute_potential_maximizers(
                model=model_mo, states=states, beta=4.0
            )
            with torch.no_grad():
                mean_mo = model_mo.posterior(states).mean
            expected_pareto = torch.where(is_non_dominated(mean_mo, deduplicate=False))[
                0
            ]
            self.assertTrue(torch.equal(z_mo_0, expected_pareto))
            self.assertTrue(set(z_mo_0.tolist()).issubset(set(z_mo_4.tolist())))

    def test_frank_wolfe_monotonicity_and_markov_occupancy(self) -> None:
        for dtype, criterion in product(
            (torch.float, torch.double), ("a_optimal", "d_optimal")
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            tol = 1e-5 if dtype == torch.float else 1e-11
            states = torch.linspace(0.0, 1.0, 7, **tkwargs).unsqueeze(-1)
            mdp = TransitionMDP.from_step_constraints(
                states=states, max_step=0.2, horizon=6
            )

            # Single-objective GP + line_search Frank-Wolfe
            model = self._build_single_task_gp(dtype=dtype, dim=1, num_outputs=1)
            planner = TransitionConstrainedMDPBO(
                model=model,
                mdp=mdp,
                criterion=criterion,
                beta=4.0,
                num_fw_iters=15,
                step_size_rule="line_search",
                fw_tol=1e-9,
            )
            mixture = planner.optimize_policy_mixture(current_state=0)

            self.assertAllClose(
                mixture.weights.sum(), torch.tensor(1.0, **tkwargs), atol=tol
            )
            self.assertTrue((mixture.weights >= 0.0).all())
            self.assertAllClose(
                mixture.visitation.sum(), torch.tensor(6.0, **tkwargs), atol=tol
            )

            # Utility history must be monotonically non-decreasing under line_search
            hist = mixture.utilities_history
            self.assertTrue(len(hist) >= 2)
            for idx in range(len(hist) - 1):
                self.assertGreaterEqual(hist[idx + 1], hist[idx] - tol)

            # Equivalent randomized Markov policy must match mixture visitation!
            markov_pi = mixture.to_markov_policy(mdp=mdp, initial_state=0)
            self.assertEqual(markov_pi.shape, (6, 7, mdp.num_actions))
            self.assertAllClose(
                markov_pi.sum(dim=-1), torch.ones(6, 7, **tkwargs), atol=tol
            )
            eta_markov = mdp.compute_policy_visitation(
                policy=markov_pi, initial_state=0
            )
            self.assertAllClose(eta_markov, mixture.visitation, atol=tol)

            # Sampled trajectory from mixture obeys step constraints
            gen = torch.Generator(device=self.device).manual_seed(7)
            traj_idx, acts = mixture.sample_trajectory(
                mdp=mdp, initial_state=0, generator=gen
            )
            self.assertEqual(traj_idx.shape, (7,))
            self.assertEqual(acts.shape, (6,))
            self.assertEqual(int(traj_idx[0].item()), 0)
            traj_coords = states.index_select(0, traj_idx)
            step_dists = torch.linalg.vector_norm(
                traj_coords[1:] - traj_coords[:-1], ord=2.0, dim=-1
            )
            self.assertTrue((step_dists <= 0.2 + 1e-5).all())

            # Classic step-size rule + Multi-objective ModelListGP
            states_2d = torch.stack(
                torch.meshgrid(
                    torch.linspace(0.0, 1.0, 3, **tkwargs),
                    torch.linspace(0.0, 1.0, 3, **tkwargs),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 2)
            mdp_2d = TransitionMDP.from_step_constraints(
                states=states_2d, max_step=0.55, horizon=5
            )
            model_list = self._build_model_list_gp(dtype=dtype, dim=2)
            planner_classic = TransitionConstrainedMDPBO(
                model=model_list,
                mdp=mdp_2d,
                criterion=criterion,
                beta=3.0,
                num_fw_iters=10,
                step_size_rule="classic",
            )
            mix_classic = planner_classic.optimize_policy_mixture(current_state=4)
            self.assertAllClose(
                mix_classic.visitation.sum(), torch.tensor(5.0, **tkwargs), atol=tol
            )
            self.assertGreaterEqual(
                mix_classic.utilities_history[-1],
                mix_classic.utilities_history[0] - 1e-2,
            )

    def test_receding_horizon_trajectory_execution(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            # 3x3 2D grid -> 9 states, Manhattan/Euclidean nearest-neighbor step = 0.51
            grid_1d = torch.linspace(0.0, 1.0, 3, **tkwargs)
            states = torch.stack(
                torch.meshgrid(grid_1d, grid_1d, indexing="ij"), dim=-1
            ).reshape(-1, 2)
            max_step = 0.51
            horizon = 6
            mdp = TransitionMDP.from_step_constraints(
                states=states, max_step=max_step, p=2.0, horizon=horizon
            )
            model = self._build_single_task_gp(dtype=dtype, dim=2, num_outputs=2)

            for deterministic in (True, False):
                planner = TransitionConstrainedMDPBO(
                    model=model,
                    mdp=mdp,
                    criterion="d_optimal",
                    beta=4.0,
                    num_fw_iters=12,
                    step_size_rule="line_search",
                )
                gen = torch.Generator(device=self.device).manual_seed(99)
                res = planner.plan_receding_horizon_trajectory(
                    initial_state=0,
                    deterministic=deterministic,
                    generator=gen,
                )

                self.assertEqual(res.initial_state, 0)
                self.assertEqual(res.trajectory_indices.shape, (horizon + 1,))
                self.assertEqual(res.visited_indices.shape, (horizon,))
                self.assertEqual(res.visited_states.shape, (horizon, 2))
                self.assertEqual(res.actions.shape, (horizon,))
                self.assertEqual(len(res.mixtures), horizon)
                self.assertAllClose(
                    res.visitation_counts.sum(),
                    torch.tensor(float(horizon), **tkwargs),
                )

                # Every consecutive transition obeys d(x_{t+1}, x_t) <= max_step
                full_coords = states.index_select(0, res.trajectory_indices)
                step_norms = torch.linalg.vector_norm(
                    full_coords[1:] - full_coords[:-1], ord=2.0, dim=-1
                )
                self.assertTrue((step_norms <= max_step + 1e-5).all())

            # Verify deterministic receding-horizon explores multiple distinct states
            # due to conditioning on `past_visitation`!
            planner_det = TransitionConstrainedMDPBO(
                model=model, mdp=mdp, criterion="a_optimal", num_fw_iters=12
            )
            res_det = planner_det.plan_receding_horizon_trajectory(
                initial_state=0, deterministic=True
            )
            num_unique = int((res_det.visitation_counts > 0).sum().item())
            self.assertGreaterEqual(num_unique, 2)

            # Test online `update_model` and explicit `observation_noise` override
            X_new = res_det.visited_states[:2]
            Y_new = torch.zeros(2, 2, **tkwargs)
            updated_model = model.condition_on_observations(X=X_new, Y=Y_new)
            planner_det.update_model(model=updated_model, observation_noise=0.05)
            self.assertAllClose(
                planner_det.noise_var, torch.full((2, 9), 0.05, **tkwargs)
            )

    def test_utility_and_planner_validation_errors(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        states = torch.linspace(0.0, 1.0, 4, **tkwargs).unsqueeze(-1)
        mdp = TransitionMDP.from_step_constraints(
            states=states, max_step=0.4, horizon=3
        )
        model = self._build_single_task_gp(dtype=torch.double, dim=1, num_outputs=1)

        with self.assertRaises(ValueError):
            TransitionConstrainedMDPBO(model=model, mdp=mdp, criterion="invalid")
        with self.assertRaises(ValueError):
            TransitionConstrainedMDPBO(model=model, mdp=mdp, step_size_rule="invalid")
        with self.assertRaises(ValueError):
            TransitionConstrainedMDPBO(model=model, mdp=mdp, beta=-1.0)
        with self.assertRaises(ValueError):
            TransitionConstrainedMDPBO(model=model, mdp=mdp, num_fw_iters=-1)
        with self.assertRaises(ValueError):
            TransitionConstrainedMDPBO(model=model, mdp=mdp, jitter=0.0)
        with self.assertRaises(ValueError):
            TransitionConstrainedMDPBO(model=model, mdp=mdp, observation_noise=-0.1)

        cov = torch.eye(4, **tkwargs)
        nvar = torch.ones(4, **tkwargs)
        eta = torch.ones(4, **tkwargs)
        with self.assertRaises(ValueError):
            compute_posterior_covariance_woodbury(
                cov_matrix=cov, noise_var=nvar, visitation=-eta
            )
        with self.assertRaises(ValueError):
            mdp_experimental_design_utility(
                cov_matrix=cov, noise_var=nvar, visitation=eta, criterion="bad"
            )
