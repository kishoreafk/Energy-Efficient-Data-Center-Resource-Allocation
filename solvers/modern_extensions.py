"""
solvers/modern_extensions.py
============================

Modern ADP/RL extensions that scale beyond exact tabular DP.

* RTDP — Real-Time Dynamic Programming (Barto et al., 1995) on the exact tabular
  MDP: greedy-in-likelihood rollouts with Bellman backups applied *only* on the
  visited (relevant) part of the state space, sampling s' from the true row of P.

* LinearADP — RTDP-style control with a *factored* linear value function
  approximation  V(s) ~ theta . phi(s)  over hand-engineered features. This is
  the scalability path for N > 5 clusters: for N = 8 the tabular space would be
  4^8 * 5 * 3 = 983,040 states (~1.5e9 sparse transition entries), which is
  infeasible to build, while the simulator + features stay tiny.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from env.datacenter_mdp import (
    DataCenterMDP,
    POWER_FRAC,
)
from .dynamic_programming import SolverResult, greedy_policy_from_V  # noqa: F401

FEATURE_NAMES = [
    "bias",
    "active_frac",      # share of clusters that are ON (u >= 1)
    "mean_power_frac",  # mean POWER_FRAC[u_i] in [0, 1]
    "busy_frac",        # share of clusters with u >= 2
    "q_norm",           # queue length / Q_MAX
    "tau_norm",         # thermal state / 2
    "q_x_idle_cap",     # queue pressure x idle capacity
    "tau_x_busy",       # heat x load interaction
    "util_sum_norm",    # sum of u levels / (3N)
    "q_sq",             # queue squared (superlinear congestion)
]


# --------------------------------------------------------------------------------------
# Tabular RTDP
# --------------------------------------------------------------------------------------
class RTDP:
    """Real-Time Dynamic Programming on the exact tabular MDP.

    V is updated only on states the greedy rollout actually visits, so the
    value function concentrates on the relevant (reachable, high-load) region.
    """

    def __init__(
        self,
        env: DataCenterMDP,
        n_trials: int = 2500,
        max_steps: int = 80,
        theta: float = 1e-5,
        patience: int = 20,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.env = env
        self.n_trials = int(n_trials)
        self.max_steps = int(max_steps)
        self.theta = float(theta)
        self.patience = int(patience)
        self.seed = int(seed)
        self.verbose = verbose

    def _sample_start(self, rng: np.random.Generator) -> int:
        """Half canonical cold-start, half busy-state coverage start."""
        env = self.env
        if rng.random() < 0.5:
            return env.start_idx
        u = tuple(int(rng.choice(3, p=[0.45, 0.35, 0.2]) + 1) for _ in range(env.n_clusters))
        q_weights = np.power(0.55, np.arange(env.q_max + 1))  # decaying: low queues likelier
        q = int(rng.choice(env.q_max + 1, p=q_weights / q_weights.sum()))
        tau = int(rng.integers(0, 3))
        return env.state_to_idx(u, q, tau)

    def solve(self) -> SolverResult:
        env = self.env
        rng = np.random.default_rng(self.seed)
        V = np.zeros(env.n_states)
        visited = np.zeros(env.n_states, dtype=bool)
        visit_counts = np.zeros(env.n_states, dtype=np.int64)
        curve: List[float] = []
        t0 = time.perf_counter()
        streak = 0
        used_trials = 0

        for trial in range(self.n_trials):
            s = self._sample_start(rng)
            max_delta = 0.0
            for _ in range(self.max_steps):
                row = env.R_bar[s] + env.gamma * (
                    env.PROB[s] * V[env.NEXT[s]]
                ).sum(axis=-1)
                row[~env.mask[s]] = -np.inf
                a = int(row.argmax())
                v_new = float(row[a])
                max_delta = max(max_delta, abs(v_new - V[s]))
                V[s] = v_new
                visited[s] = True
                visit_counts[s] += 1
                # sample s' ~ P(s'|s,a) exactly from the (normalized) row
                k = int(rng.choice(env.PROB.shape[-1], p=env.PROB[s, a]))
                s = int(env.NEXT[s, a, k])
            curve.append(max_delta)
            used_trials = trial + 1
            streak = streak + 1 if max_delta < self.theta else 0
            if self.verbose and trial % 100 == 0:
                print(f"  [RTDP] trial {trial:4d}  max_backup_delta={max_delta:.3e}")
            if streak >= self.patience:
                break

        pi = greedy_policy_from_V(env, V)
        return SolverResult(
            V=V,
            pi=pi,
            n_sweeps=used_trials,
            residuals=curve,
            time_s=time.perf_counter() - t0,
            converged=streak >= self.patience,
            algorithm="RTDP",
            extra={"visited_states": int(visited.sum()), "visited_mask": visited,
                   "visit_counts": visit_counts},
        )


# --------------------------------------------------------------------------------------
# Linear value-function approximation for N > 5 (factored / approximate DP)
# --------------------------------------------------------------------------------------
class LinearADP:
    """Semi-gradient TD(0) control with linear VFA, driven by simulator rollouts.

    Q_hat(s, a) = r_bar(s, a) + gamma * theta . phi(s'_expected(s, a))
    where r_bar and the expected next state use the exact Poisson expectations,
    so the greedy policy is a one-step-lookahead actor over the fitted value.
    """

    def __init__(
        self,
        env: DataCenterMDP,
        alpha0: float = 0.05,
        n_steps: int = 25000,
        episode_len: int = 200,
        seed: int = 0,
        lam_sampler: Optional[Callable[[np.random.Generator], float]] = None,
        verbose: bool = False,
    ) -> None:
        self.env = env
        self.alpha0 = float(alpha0)
        self.n_steps = int(n_steps)
        self.episode_len = int(episode_len)
        self.seed = int(seed)
        self.lam_sampler = lam_sampler
        self.verbose = verbose

    # ---------------- features ----------------
    def features(self, u: Sequence[int], q: float, tau: int) -> np.ndarray:
        env = self.env
        u = np.asarray(u, dtype=float)
        active = (u >= 1).mean()
        busy = (u >= 2).mean()
        mean_power = float(np.mean([POWER_FRAC[int(min(max(x, 0), 3))] for x in u]))
        q_n = float(q) / env.q_max
        tau_n = float(tau) / 2.0
        return np.array(
            [
                1.0,
                active,
                mean_power,
                busy,
                q_n,
                tau_n,
                q_n * (1.0 - active),
                tau_n * busy,
                float(u.sum()) / (3.0 * env.n_clusters),
                q_n * q_n,
            ],
            dtype=np.float64,
        )

    # ---------------- greedy one-step-lookahead policy ----------------
    def greedy_action(self, u: Tuple[int, ...], q: int, tau: int, lam: float):
        env = self.env
        theta = self.theta
        best_a, best_q = None, -np.inf
        for a in range(env.n_actions):
            a_alloc, a_power, a_cool = env.idx_to_action(a)
            if not env.action_is_valid(u, q, a_alloc, a_power):
                continue
            u2, e_q, tau2, r_bar, _drop = env.expected_outcome(
                u, q, tau, a_alloc, a_power, a_cool, lam=lam
            )
            q_val = r_bar + env.gamma * float(theta @ self.features(u2, e_q, tau2))
            if q_val > best_q:
                best_q, best_a = q_val, a
        return best_a, best_q

    def act(self, u, q, tau, t: int = 0, lam: float = 3.0) -> int:
        a, _ = self.greedy_action(tuple(u), int(q), int(tau), lam)
        return int(a) if a is not None else self.env.action_to_idx(self.env.n_clusters, 0, 0)

    # ---------------- training loop ----------------
    def solve(self):
        env = self.env
        rng = np.random.default_rng(self.seed)
        self.theta = np.zeros(len(FEATURE_NAMES))
        td_curve: List[float] = []
        window: List[float] = []
        t0 = time.perf_counter()
        u, q, tau = tuple([1] * env.n_clusters), 0, 0
        lam = 3.0
        episode_t = 0

        for step in range(self.n_steps):
            if episode_t % self.episode_len == 0:  # new episode: resample load regime
                lam = (
                    self.lam_sampler(rng)
                    if self.lam_sampler is not None
                    else float(rng.uniform(0.3, 7.0))
                )
                episode_t = 0
            a = self.greedy_action(u, q, tau, lam)[0]
            if a is None:  # dead-end guard (cannot happen with the mask)
                u, q, tau = tuple([1] * env.n_clusters), 0, 0
                continue
            a_alloc, a_power, a_cool = env.idx_to_action(a)
            phi_s = self.features(u, q, tau)
            u2, q2, tau2, r, _info = env.step_dynamics(
                u, q, tau, a_alloc, a_power, a_cool, lam=lam, rng=rng
            )
            phi_s2 = self.features(u2, q2, tau2)
            td_err = r + env.gamma * float(self.theta @ phi_s2) - float(self.theta @ phi_s)
            alpha = self.alpha0 / (1.0 + step / 4000.0)
            self.theta += alpha * td_err * phi_s
            window.append(abs(td_err))
            if len(window) >= 100:
                td_curve.append(float(np.mean(window)))
                window.clear()
            u, q, tau = u2, q2, tau2
            episode_t += 1
            if self.verbose and step % 2500 == 0:
                print(f"  [ADP] step {step:6d}  |TD|={np.mean(window) if window else 0:.3f}"
                      f"  lam={lam:.2f}")

        return {
            "theta": self.theta.copy(),
            "td_curve": td_curve,
            "time_s": time.perf_counter() - t0,
            "n_steps": self.n_steps,
            "feature_names": list(FEATURE_NAMES),
        }
