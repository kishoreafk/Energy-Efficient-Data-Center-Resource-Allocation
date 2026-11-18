"""
solvers/dynamic_programming.py
==============================

Exact tabular Dynamic Programming for DataCenterMDP.

* ValueIteration  — synchronous (Jacobi) and asynchronous (Gauss-Seidel) Bellman
  optimality backups with configurable convergence threshold theta (default 1e-5).
* PolicyIteration — exact policy evaluation via the linear system
  (I - gamma * P_pi) V = r_pi solved with np.linalg.solve, or iterative backups;
  policy improvement with strict epsilon-improvement tie-breaking (1e-8) and
  policy-hash oscillation detection (guardrail Issue 4).

All solvers respect the environment's action mask: masked actions are set to
-inf before the max, so illegal/degenerate moves can never be selected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse

from env.datacenter_mdp import DataCenterMDP


# --------------------------------------------------------------------------------------
# Shared Bellman operators
# --------------------------------------------------------------------------------------
def q_values(env: DataCenterMDP, V: np.ndarray) -> np.ndarray:
    """Q(s, a) = R_bar(s, a) + gamma * sum_k P(s'|s,a) V(s')   ->  [S, A]."""
    return env.R_bar + env.gamma * (env.PROB * V[env.NEXT]).sum(axis=-1)


def greedy_policy_from_V(env: DataCenterMDP, V: np.ndarray) -> np.ndarray:
    """pi(s) = argmax_a Q(s, a) over mask-valid actions only."""
    Q = np.where(env.mask, q_values(env, V), -np.inf)
    return Q.argmax(axis=1).astype(np.int64)


@dataclass
class SolverResult:
    V: np.ndarray
    pi: np.ndarray
    n_sweeps: int
    residuals: List[float]
    time_s: float
    converged: bool
    algorithm: str
    extra: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Value Iteration
# --------------------------------------------------------------------------------------
class ValueIteration:
    """Bellman optimality sweeps, synchronous or asynchronous (Gauss-Seidel)."""

    def __init__(
        self,
        env: DataCenterMDP,
        theta: float = 1e-5,
        max_sweeps: int = 5000,
        mode: str = "synchronous",  # "synchronous" | "gauss_seidel"
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        if mode not in ("synchronous", "gauss_seidel"):
            raise ValueError("mode must be 'synchronous' or 'gauss_seidel'")
        self.env = env
        self.theta = float(theta)
        self.max_sweeps = int(max_sweeps)
        self.mode = mode
        self.seed = int(seed)
        self.verbose = verbose

    def _backup_row(self, s: int, V: np.ndarray) -> np.ndarray:
        row = self.env.R_bar[s] + self.env.gamma * (
            self.env.PROB[s] * V[self.env.NEXT[s]]
        ).sum(axis=-1)
        row[~self.env.mask[s]] = -np.inf
        return row

    def solve(self) -> SolverResult:
        env = self.env
        S = env.n_states
        V = np.zeros(S, dtype=np.float64)
        residuals: List[float] = []
        converged = False
        t0 = time.perf_counter()

        if self.mode == "synchronous":
            for k in range(self.max_sweeps):
                V_new = np.where(env.mask, q_values(env, V), -np.inf).max(axis=1)
                res = float(np.max(np.abs(V_new - V)))
                V = V_new
                residuals.append(res)
                if self.verbose and (k % 25 == 0 or res < self.theta):
                    print(f"  [VI-sync] sweep {k:4d}  residual={res:.3e}")
                if res < self.theta:
                    converged = True
                    break
        else:  # Gauss-Seidel: in-place updates along a fixed shuffled order
            order = np.random.default_rng(self.seed).permutation(S)
            for k in range(self.max_sweeps):
                res = 0.0
                for s in order:
                    v_new = self._backup_row(int(s), V).max()
                    res = max(res, abs(v_new - V[s]))
                    V[s] = v_new
                residuals.append(res)
                if self.verbose and (k % 25 == 0 or res < self.theta):
                    print(f"  [VI-GS ] sweep {k:4d}  residual={res:.3e}")
                if res < self.theta:
                    converged = True
                    break

        pi = greedy_policy_from_V(env, V)
        return SolverResult(
            V=V,
            pi=pi,
            n_sweeps=len(residuals),
            residuals=residuals,
            time_s=time.perf_counter() - t0,
            converged=converged,
            algorithm=f"ValueIteration({self.mode})",
            extra={"theta": self.theta, "mode": self.mode},
        )


# --------------------------------------------------------------------------------------
# Policy Iteration
# --------------------------------------------------------------------------------------
class PolicyIteration:
    """Exact policy evaluation (linear solve or iterative backups) + improvement."""

    def __init__(
        self,
        env: DataCenterMDP,
        theta: float = 1e-5,
        eval_method: str = "linear",  # "linear" | "iterative"
        improvement_eps: float = 1e-8,
        max_iters: int = 200,
        verbose: bool = False,
    ) -> None:
        if eval_method not in ("linear", "iterative"):
            raise ValueError("eval_method must be 'linear' or 'iterative'")
        self.env = env
        self.theta = float(theta)
        self.eval_method = eval_method
        self.improvement_eps = float(improvement_eps)
        self.max_iters = int(max_iters)
        self.verbose = verbose

    # ---------------- policy evaluation ----------------
    def _build_pi_mdp(self, pi: np.ndarray):
        env = self.env
        S, K = env.n_states, env.PROB.shape[-1]
        idx = np.arange(S)
        prob_pi = env.PROB[idx, pi]                     # [S, K]
        next_pi = env.NEXT[idx, pi]                     # [S, K]
        rew_pi = env.REW[idx, pi]                       # [S, K]
        r_bar = (prob_pi * rew_pi).sum(axis=-1)
        rows = np.repeat(idx, K)
        P_pi = sparse.csr_matrix(
            (prob_pi.ravel(), (rows, next_pi.ravel())), shape=(S, S)
        )
        P_pi.sum_duplicates()
        return P_pi, r_bar

    def evaluate_policy(self, pi: np.ndarray):
        """Returns (V, eval_residual_supnorm)."""
        env = self.env
        P_pi, r_bar = self._build_pi_mdp(pi)
        if self.eval_method == "linear":
            A_dense = np.eye(env.n_states) - env.gamma * P_pi.toarray()
            V = np.linalg.solve(A_dense, r_bar)
            eval_res = float(np.max(np.abs(A_dense @ V - r_bar)))
        else:  # iterative backups
            V = np.zeros(env.n_states)
            while True:
                V_new = r_bar + env.gamma * P_pi @ V
                res = float(np.max(np.abs(V_new - V)))
                V = V_new
                if res < self.theta:
                    break
            eval_res = res
        return V, eval_res

    # ---------------- policy improvement ----------------
    def improve_policy(self, V: np.ndarray, pi_old: np.ndarray):
        """Strict epsilon-improvement: keep the old action on (near-)ties.

        Guardrail Issue 4: without the +1e-8 margin, floating-point noise makes
        the policy chatter between identical Q-values forever.
        """
        env = self.env
        Q = np.where(env.mask, q_values(env, V), -np.inf)
        best = Q.argmax(axis=1)
        best_q = Q[np.arange(env.n_states), best]
        old_q = Q[np.arange(env.n_states), pi_old]
        pi_new = np.where(best_q > old_q + self.improvement_eps, best, pi_old)
        changes = int((pi_new != pi_old).sum())
        return pi_new.astype(np.int64), changes, Q

    # ---------------- main loop ----------------
    def solve(self) -> SolverResult:
        env = self.env
        t0 = time.perf_counter()
        # deterministic initial policy: greedy on the immediate expected reward
        pi = np.where(env.mask, env.R_bar, -np.inf).argmax(axis=1).astype(np.int64)

        seen_policies = {pi.tobytes(): 0}
        history: List[Dict] = []
        oscillation = False
        V = np.zeros(env.n_states)
        eval_res = 0.0

        for it in range(self.max_iters):
            V, eval_res = self.evaluate_policy(pi)
            pi_new, changes, _Q = self.improve_policy(V, pi)
            history.append(
                {"iteration": it, "policy_changes": changes, "eval_residual": eval_res}
            )
            if self.verbose:
                print(f"  [PI] iter {it:2d}  changes={changes:4d}  eval_res={eval_res:.2e}")
            if changes == 0:
                break
            key = pi_new.tobytes()
            if key in seen_policies:  # policy cycle -> stop and flag
                oscillation = True
                if self.verbose:
                    print("  [PI] oscillation detected (repeated policy) - stopping")
                break
            seen_policies[key] = it + 1
            pi = pi_new

        converged = (not oscillation) and history[-1]["policy_changes"] == 0
        return SolverResult(
            V=V,
            pi=pi,
            n_sweeps=len(history),
            residuals=[h["eval_residual"] for h in history],
            time_s=time.perf_counter() - t0,
            converged=converged,
            algorithm=f"PolicyIteration({self.eval_method})",
            extra={
                "history": history,
                "policy_changes": [h["policy_changes"] for h in history],
                "oscillation_detected": oscillation,
                "improvement_eps": self.improvement_eps,
            },
        )


def policy_agreement(env: DataCenterMDP, pi_a: np.ndarray, pi_b: np.ndarray) -> float:
    """Fraction of states where two policies pick the same action."""
    return float((pi_a == pi_b).mean())
