"""
env/datacenter_mdp.py
=====================

Exact tabular MDP for energy-efficient data-center resource allocation.

State  s = (u_1, ..., u_N, q, tau)
    u_i in {0=Sleep, 1=Idle, 2=Medium, 3=High}   discretized utilization / power state
    q   in {0, ..., Q_MAX}                       job queue length
    tau in {0=Cool, 1=Warm, 2=Hot}               ambient thermal state (drives PUE)

Action a = (a_alloc, a_power, a_cool)
    a_alloc in {0..N-1}  route the queued tasks to cluster a_alloc; N = hold
    a_power in {0 noop, 1..N wake cluster i, N+1..2N sleep cluster i}
    a_cool  in {0 low, 1 high}  HVAC / chiller intensity

Dynamics per step (dt = 5 minutes)
    1. power switch is applied (wake: Sleep->Idle and pay boot spike E_switch;
       sleep: Idle->Sleep, only idle servers may be put to sleep)
    2. routing (priority + spill): the target cluster is served first
       (min(q, cap_i(u_target))); leftover backlog spills to the other ACTIVE
       clusters in tier order (efficient-first). Fleet throughput is therefore
       the sum of the caps of all engaged clusters (up to 12/step vs peak
       lambda 7), and no single cluster (max cap 5) can drain the 6-slot buffer
       alone - scale-out is a genuine energy/SLA decision.
    3. utilization: the serving cluster rises to min(3, max(u,1) + ceil(served/2));
       every other active cluster decays one level toward Idle; Sleep stays Sleep
    4. thermal (deterministic): pressure = n_busy + (0 if cool=high else 1) - 2,
       tau' = clip(tau + sign(pressure), 0, 2)
    5. arrivals n ~ Poisson(lambda) discretized with the tail folded at Q_MAX
       q' = min(Q_MAX, q - served + n),  dropped = max(0, q - served + n - Q_MAX)

Reward (negative total cost, maximized)
    R(s,a,s') = -( w1*E_IT(s,a)*PUE(tau,a_cool) + w2*E_switch(s,a)
                   + w3*(dropped * P_drop + I(tau'==Hot) * P_hot) )

Engineering guardrails implemented here
    * |S| kept between 300 and 1500 (1344 states for N=3, Q_MAX=6)
    * P(s'|s,a) stored as scipy.sparse CSR matrices, one per action
      (never a dense |S| x |A| x |S| tensor)
    * tail probability mass is accumulated into the boundary state q' = Q_MAX
    * action mask: if q > 0 and every cluster sleeps, the policy MUST wake one
    * anti "lazy shutdown": w3 * P_drop > 5 * max(E_IT + E_switch) per step
    * anti "thrashing": E_switch >= 5 x idle energy per step (hysteresis)

Gymnasium-compatible: if the `gymnasium` package is available the class
subclasses gym.Env and exposes Discrete observation/action spaces.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

try:  # Gymnasium is optional; the tabular MDP works standalone as well.
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
except Exception:  # pragma: no cover
    _HAS_GYM = False
    gym = None
    spaces = None

# --------------------------------------------------------------------------------------
# Physical / cost model constants (canonical defaults; the env stores its own copies)
# --------------------------------------------------------------------------------------
STEP_MINUTES = 5
STEP_HOURS = STEP_MINUTES / 60.0

U_LABELS = ("Sleep", "Idle", "Medium", "High")
TAU_LABELS = ("Cool", "Warm", "Hot")

# Tiered hardware: cluster 0 is energy-efficient but small, cluster 2 is powerful
# but thirsty. No single cluster can absorb the peak arrival rate (cap_max = 7 <
# lambda_peak = 7 at surge steps), so scale-out / wake decisions genuinely trade
# energy against SLA violations.
PEAK_POWER_KW = (0.8, 1.0, 1.3)          # IT peak power per cluster [kW]
# Every single cap is STRICTLY below the queue buffer Q_MAX=6: one cluster alone
# cannot drain the shared backlog at peak (lambda up to 7/step), so scale-out
# (waking + spilling into bigger tiers) genuinely trades energy against SLA.
CAP_TABLE_N3 = ((1, 2, 3), (1, 3, 4), (2, 4, 5))   # cap[tier][u-1], u = 1..3
CAP_LO, CAP_HI = (1, 2, 3), (2, 4, 5)              # interpolation anchors for N != 3
POWER_FRAC = (0.05, 0.60, 0.80, 1.00)    # Sleep / Idle / Medium / High share of peak
SWITCH_IDLE_STEPS = 5.0                  # boot spike = 5 x idle energy of one step

# PUE(tau, a_cool): rows = Cool/Warm/Hot, cols = cooling low/high.
# High cooling wastes fan power in a cool room but saves energy when Hot.
PUE_TABLE = ((1.30, 1.45), (1.50, 1.55), (1.80, 1.65))

# Cost weights (reward = negative cost)
W_ENERGY = 10.0      # cost units per kWh of IT energy (times PUE)
W_SWITCH = 10.0      # cost units per kWh of boot-spike energy
W_SLA = 1.0          # scales the SLA block
PENALTY_DROP = 50.0  # cost units per dropped task   (w3 * P_drop guardrail)
PENALTY_HOT = 8.0    # cost units per step spent Hot  (thermal throttling)

GAMMA = 0.95          # discount factor, inside the required [0.90, 0.99] band
LAMBDA_SOLVE = 3.0    # mean Poisson arrivals per step used to build P
SEED = 42

State = Tuple[Tuple[int, ...], int, int]
Action = Tuple[int, int, int]


class DataCenterMDP(gym.Env if _HAS_GYM else object):  # type: ignore[misc,valid-type]
    """Tabular, Gymnasium-compatible data-center MDP.

    Parameters
    ----------
    n_clusters : number of server clusters (N). Use 3 for exact tabular DP.
    build_exact : build the full transition tensors (NEXT/PROB/REW + sparse P, R).
        Set to False for large N (e.g. 8) where |S| makes exact DP infeasible;
        the step simulator keeps working either way.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_clusters: int = 3,
        q_max: int = 6,
        lambda_arrivals: float = LAMBDA_SOLVE,
        gamma: float = GAMMA,
        w_energy: float = W_ENERGY,
        w_switch: float = W_SWITCH,
        w_sla: float = W_SLA,
        penalty_drop: float = PENALTY_DROP,
        penalty_hot: float = PENALTY_HOT,
        build_exact: bool = True,
        seed: int = SEED,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        self.n_clusters = int(n_clusters)
        self.q_max = int(q_max)
        self.lambda_arrivals = float(lambda_arrivals)
        self.gamma = float(gamma)
        self.w_energy = float(w_energy)
        self.w_switch = float(w_switch)
        self.w_sla = float(w_sla)
        self.penalty_drop = float(penalty_drop)
        self.penalty_hot = float(penalty_hot)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # ---------------- tiered hardware model ----------------
        if self.n_clusters == 3:
            self.peak_kw = np.array(PEAK_POWER_KW, dtype=float)
        else:  # generic tiers for N != 3 (simulator mode)
            denom = max(1, self.n_clusters - 1)
            self.peak_kw = np.array(
                [0.7 + 0.7 * i / denom for i in range(self.n_clusters)], dtype=float
            )
        # service capacity of cluster i at utilization level u (u >= 1)
        self.cap_table = np.zeros((self.n_clusters, 4), dtype=int)
        for i in range(self.n_clusters):
            frac = i / max(1, self.n_clusters - 1)
            for u in (1, 2, 3):
                cap = CAP_LO[u - 1] + (CAP_HI[u - 1] - CAP_LO[u - 1]) * frac
                self.cap_table[i, u] = max(1, int(cap + 0.5))
        # energies per 5-minute step [kWh]
        self.idle_kwh = POWER_FRAC[1] * self.peak_kw * STEP_HOURS
        self.e_switch_kwh = SWITCH_IDLE_STEPS * self.idle_kwh

        # ---------------- action space geometry ----------------
        self.n_alloc = self.n_clusters + 1            # +1 = hold
        self.n_power = 2 * self.n_clusters + 1        # noop, wakes, sleeps
        self.n_cool = 2
        self.n_actions = self.n_alloc * self.n_power * self.n_cool

        # ---------------- state space geometry ----------------
        self.q_levels = self.q_max + 1
        self.n_states = (4 ** self.n_clusters) * self.q_levels * 3
        # mixed-radix strides for idx = (...((u0*a + u1)*a + u2)*qlevels + q)*3 + tau
        self._u_strides = np.array(
            [4 ** (self.n_clusters - 1 - i) * self.q_levels * 3 for i in range(self.n_clusters)],
            dtype=np.int64,
        )
        self._q_stride = 3
        self._tau_stride = 1

        self._poisson_cache: Dict[float, np.ndarray] = {}
        self._states: Optional[np.ndarray] = None

        # Gymnasium spaces (when available)
        if _HAS_GYM:
            self.observation_space = spaces.Discrete(self.n_states)
            self.action_space = spaces.Discrete(self.n_actions)

        # exact-mode tensors (filled by _build_exact)
        self.NEXT: Optional[np.ndarray] = None      # [S, A, K] int next-state ids
        self.PROB: Optional[np.ndarray] = None      # [S, A, K] transition probabilities
        self.REW: Optional[np.ndarray] = None       # [S, A, K] rewards r(s,a,s')
        self.R_bar: Optional[np.ndarray] = None     # [S, A] expected reward
        self.mask: Optional[np.ndarray] = None      # [S, A] action validity
        self.P: Optional[List[sparse.csr_matrix]] = None
        self.R: Optional[List[sparse.csr_matrix]] = None

        self._s = self.start_idx
        if build_exact and self.n_states <= 100_000:
            self._build_exact()

    # ==================================================================================
    # Encoding / decoding
    # ==================================================================================
    def state_to_idx(self, u: Sequence[int], q: int, tau: int) -> int:
        """Mixed-radix encode (u tuple, q, tau) -> flat state id."""
        idx = int(q) * self._q_stride + int(tau)
        for i in range(self.n_clusters):
            idx += int(u[i]) * int(self._u_strides[i])
        return idx

    def idx_to_state(self, idx: int) -> Tuple[Tuple[int, ...], int, int]:
        """Flat state id -> (u tuple, q, tau)."""
        idx = int(idx)
        tau = idx % 3
        idx //= 3
        q = idx % self.q_levels
        idx //= self.q_levels
        u = []
        for i in range(self.n_clusters):
            u.append(idx % 4)
            idx //= 4
        u.reverse()
        return tuple(u), q, tau

    def action_to_idx(self, a_alloc: int, a_power: int, a_cool: int) -> int:
        return (int(a_alloc) * self.n_power + int(a_power)) * self.n_cool + int(a_cool)

    def idx_to_action(self, idx: int) -> Action:
        cool = idx % self.n_cool
        rest = idx // self.n_cool
        power = rest % self.n_power
        alloc = rest // self.n_power
        return (alloc, power, cool)

    # Convenience aliases requested in the spec
    encode_state = state_to_idx
    decode_state = idx_to_state

    @property
    def S(self) -> List[State]:
        """All states as decoded tuples (exact mode only)."""
        return [self.idx_to_state(s) for s in range(self.n_states)]

    @property
    def A(self) -> List[Action]:
        return [self.idx_to_action(a) for a in range(self.n_actions)]

    @property
    def start_idx(self) -> int:
        """Canonical start state: all clusters Idle, empty queue, Cool."""
        return self.state_to_idx(tuple([1] * self.n_clusters), 0, 0)

    # ==================================================================================
    # Poisson arrival model (tail folded at Q_MAX)
    # ==================================================================================
    def poisson_probs(self, lam: float) -> np.ndarray:
        """P(n=k) for k = 0..Q_MAX-1 plus the folded tail at k = Q_MAX."""
        key = round(float(lam), 6)
        probs = self._poisson_cache.get(key)
        if probs is None:
            lam = float(lam)
            ks = np.arange(self.q_max, dtype=np.int64)
            pk = np.exp(-lam) * np.power(lam, ks) / np.array(
                [math.factorial(int(k)) for k in ks], dtype=float
            )
            tail = max(0.0, 1.0 - pk.sum())
            probs = np.append(pk, tail)
            self._poisson_cache[key] = probs
        return probs

    # ==================================================================================
    # Core dynamics (single source of truth for P, the simulator and the expectations)
    # ==================================================================================
    def action_is_valid(self, u: Sequence[int], q: int, a_alloc: int, a_power: int) -> bool:
        """Action mask rules (guardrail against the lazy-shutdown degenerate policy)."""
        u = list(u)
        if a_power == 0:  # noop
            if q > 0 and all(x == 0 for x in u):
                return False  # forced wake: queue waiting, nobody alive
        elif 1 <= a_power <= self.n_clusters:  # wake cluster i
            if u[a_power - 1] != 0:
                return False
        else:  # sleep cluster i (only an Idle server may sleep)
            if u[a_power - 1 - self.n_clusters] != 1:
                return False
        if a_alloc < self.n_clusters:  # routing target must be active or woken now
            i = a_alloc
            woken_now = u[i] == 0 and a_power == i + 1
            if u[i] == 0 and not woken_now:
                return False
        return True

    def valid_actions(self, s_idx: int) -> np.ndarray:
        u, q, _ = self.idx_to_state(s_idx)
        out = [
            a
            for a in range(self.n_actions)
            if self.action_is_valid(u, q, *self.idx_to_action(a)[:2])
        ]
        return np.asarray(out, dtype=np.int64)

    def _deterministic_part(
        self, u: Sequence[int], q: int, tau: int, a_alloc: int, a_power: int, a_cool: int
    ) -> Dict:
        u = list(int(x) for x in u)
        u_post = list(u)
        switch_kwh = 0.0
        woke = -1
        if 1 <= a_power <= self.n_clusters:
            i = a_power - 1
            if u_post[i] == 0:
                u_post[i] = 1
                switch_kwh = float(self.e_switch_kwh[i])
                woke = i
        elif a_power > self.n_clusters:
            i = a_power - 1 - self.n_clusters
            if u_post[i] == 1:
                u_post[i] = 0

        target = a_alloc if a_alloc < self.n_clusters else -1
        served_per_cluster = [0] * self.n_clusters
        if q > 0:
            # engagement order: target first (if servable), then other active
            # clusters in tier order (efficient-first spill)
            order = []
            if target >= 0 and (u_post[target] >= 1 or target == woke):
                order.append(target)
            order += [i for i in range(self.n_clusters)
                      if i not in order and u_post[i] >= 1]
            remaining = q
            for i in order:
                if remaining <= 0:
                    break
                cap = int(self.cap_table[i, max(u_post[i], 1)])
                take = int(min(remaining, cap))
                served_per_cluster[i] = take
                remaining -= take
        served = int(sum(served_per_cluster))
        q_mid = q - served

        u_next = []
        for i in range(self.n_clusters):
            if served_per_cluster[i] > 0:
                bump = -(-served_per_cluster[i] // 2)  # ceil(served_i / 2)
                u_next.append(min(3, max(u_post[i], 1) + bump))
            elif u_post[i] >= 2:
                u_next.append(u_post[i] - 1)  # backlog drains toward Idle
            else:
                u_next.append(u_post[i])      # Idle stays Idle, Sleep stays Sleep

        n_busy = sum(1 for x in u_next if x >= 2)
        pressure = n_busy + (0 if a_cool == 1 else 1) - 2
        d_tau = 1 if pressure >= 1 else (-1 if pressure <= -1 else 0)
        tau_next = int(min(2, max(0, tau + d_tau)))

        e_it_kwh = float(np.sum(np.array([POWER_FRAC[x] for x in u_post]) * self.peak_kw)
                         * STEP_HOURS)
        return {
            "u_post": u_post,
            "u_next": tuple(u_next),
            "q_mid": q_mid,
            "served": served,
            "served_per_cluster": served_per_cluster,
            "target": target,
            "woke": woke,
            "switch_kwh": switch_kwh,
            "tau_next": tau_next,
            "e_it_kwh": e_it_kwh,
            "pue": PUE_TABLE[tau][a_cool],
        }

    def _reward(self, det: Dict, dropped: int) -> float:
        sla = self.w_sla * (dropped * self.penalty_drop
                            + (self.penalty_hot if det["tau_next"] == 2 else 0.0))
        return -(
            self.w_energy * det["e_it_kwh"] * det["pue"]
            + self.w_switch * det["switch_kwh"]
            + sla
        )

    def _apply_arrival(self, det: Dict, n: int) -> Tuple[Tuple[int, ...], int, int, float, Dict]:
        total = det["q_mid"] + int(n)
        q_next = int(min(self.q_max, total))
        dropped = int(max(0, total - self.q_max))
        reward = self._reward(det, dropped)
        info = {
            "served": det["served"],
            "dropped": dropped,
            "arrived": int(n),
            "switched": det["switch_kwh"] > 0.0,
            "switch_kwh": det["switch_kwh"],
            "e_it_kwh": det["e_it_kwh"],
            "pue": det["pue"],
            "e_total_kwh": (det["e_it_kwh"] + det["switch_kwh"]) * det["pue"],
            "tau_next": det["tau_next"],
            "u_next": det["u_next"],
        }
        return det["u_next"], q_next, det["tau_next"], float(reward), info

    def outcomes(
        self, u: Sequence[int], q: int, tau: int, a_alloc: int, a_power: int, a_cool: int,
        lam: Optional[float] = None,
    ) -> List[Tuple[float, Tuple[int, ...], int, int, float, int]]:
        """All stochastic outcomes of (s, a): (prob, u', q', tau', reward, dropped).

        The Poisson tail is folded at q_max so every (s, a) row sums to exactly 1.
        When clamping merges two arrival counts into the same s', the stored reward
        is the probability-weighted mean (exact for expected-reward DP).
        """
        lam = self.lambda_arrivals if lam is None else lam
        det = self._deterministic_part(u, q, tau, a_alloc, a_power, a_cool)
        probs = self.poisson_probs(lam)
        merged: Dict[Tuple[Tuple[int, ...], int, int], List[float]] = {}
        for n in range(len(probs)):
            u2, q2, tau2, r, _ = self._apply_arrival(det, n)
            key = (u2, q2, tau2)
            if key in merged:
                merged[key][0] += float(probs[n])
                merged[key][1] += float(probs[n]) * r
            else:
                merged[key] = [float(probs[n]), float(probs[n]) * r]
        return [
            (p, u2, q2, tau2, rw / p, 0)
            for (u2, q2, tau2), (p, rw) in merged.items()
        ]

    def step_dynamics(
        self,
        u: Sequence[int],
        q: int,
        tau: int,
        a_alloc: int,
        a_power: int,
        a_cool: int,
        lam: Optional[float] = None,
        forced_arrival: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[Tuple[int, ...], int, int, float, Dict]:
        """Sample one transition from the true dynamics (simulator mode & traces)."""
        lam = self.lambda_arrivals if lam is None else lam
        rng = self.rng if rng is None else rng
        det = self._deterministic_part(u, q, tau, a_alloc, a_power, a_cool)
        if forced_arrival is not None:
            n = int(forced_arrival)
        else:
            probs = self.poisson_probs(lam)
            n = int(rng.choice(len(probs), p=probs))
        return self._apply_arrival(det, n)

    def expected_outcome(
        self, u: Sequence[int], q: int, tau: int, a_alloc: int, a_power: int, a_cool: int,
        lam: Optional[float] = None,
    ) -> Tuple[Tuple[int, ...], float, int, float, float]:
        """Exact expectation of the next state and reward (used by the ADP solver)."""
        lam = self.lambda_arrivals if lam is None else lam
        det = self._deterministic_part(u, q, tau, a_alloc, a_power, a_cool)
        probs = self.poisson_probs(lam)
        e_q = 0.0
        e_drop = 0.0
        for n in range(len(probs)):
            total = det["q_mid"] + n
            e_q += float(probs[n]) * min(self.q_max, total)
            e_drop += float(probs[n]) * max(0, total - self.q_max)
        # expected reward: linear in dropped tasks + hot-state indicator
        base = -(self.w_energy * det["e_it_kwh"] * det["pue"] + self.w_switch * det["switch_kwh"])
        sla = self.w_sla * (e_drop * self.penalty_drop
                            + (self.penalty_hot if det["tau_next"] == 2 else 0.0))
        e_reward = base - sla
        return det["u_next"], float(e_q), det["tau_next"], float(e_reward), float(e_drop)

    # ==================================================================================
    # Exact tabular construction (scipy.sparse, guardrail Issue 1)
    # ==================================================================================
    def _build_exact(self) -> None:
        S, A, K = self.n_states, self.n_actions, self.q_max + 1
        NEXT = np.zeros((S, A, K), dtype=np.int64)
        PROB = np.zeros((S, A, K), dtype=np.float64)
        REW = np.zeros((S, A, K), dtype=np.float64)
        for s in range(S):
            u, q, tau = self.idx_to_state(s)
            for a in range(A):
                a_alloc, a_power, a_cool = self.idx_to_action(a)
                for k, (p, u2, q2, tau2, r, _d) in enumerate(
                    self.outcomes(u, q, tau, a_alloc, a_power, a_cool)
                ):
                    NEXT[s, a, k] = self.state_to_idx(u2, q2, tau2)
                    PROB[s, a, k] = p
                    REW[s, a, k] = r
        self.NEXT, self.PROB, self.REW = NEXT, PROB, REW
        self.R_bar = (PROB * REW).sum(axis=-1)

        rows = np.repeat(np.arange(S, dtype=np.int64), K)
        self.P, self.R = [], []
        for a in range(A):
            cols = NEXT[:, a, :].ravel()
            pr = PROB[:, a, :].ravel()
            rw = REW[:, a, :].ravel()
            keep = pr > 0.0
            P_a = sparse.csr_matrix(
                (pr[keep], (rows[keep], cols[keep])), shape=(S, S)
            )
            P_a.sum_duplicates()
            self.P.append(P_a)
            R_a = sparse.csr_matrix(
                (rw[keep], (rows[keep], cols[keep])), shape=(S, S)
            )
            R_a.sum_duplicates()
            self.R.append(R_a)

        mask = np.zeros((S, A), dtype=bool)
        for s in range(S):
            u, q, _ = self.idx_to_state(s)
            for a in range(A):
                a_alloc, a_power, _c = self.idx_to_action(a)
                mask[s, a] = self.action_is_valid(u, q, a_alloc, a_power)
        self.mask = mask

    # ==================================================================================
    # Simulator / Gymnasium API
    # ==================================================================================
    def simulate_step(
        self,
        s: int,
        a: int,
        lam: Optional[float] = None,
        forced_arrival: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, float, Dict]:
        """Pure step function on flat ids (used by the evaluation harness)."""
        if self.NEXT is None and forced_arrival is None and s >= self.n_states:
            raise IndexError("state id out of range")
        u, q, tau = self.idx_to_state(s)
        a_alloc, a_power, a_cool = self.idx_to_action(a)
        u2, q2, tau2, r, info = self.step_dynamics(
            u, q, tau, a_alloc, a_power, a_cool, lam=lam,
            forced_arrival=forced_arrival, rng=rng,
        )
        return self.state_to_idx(u2, q2, tau2), r, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._s = self.start_idx
        return self._s, {"state": self.idx_to_state(self._s)}

    def step(self, action: int):
        """Gymnasium step: continuing task (never terminates)."""
        s2, r, info = self.simulate_step(self._s, int(action), rng=self.rng)
        self._s = s2
        return s2, float(r), False, False, info

    # ==================================================================================
    # Reporting helpers
    # ==================================================================================
    def max_step_cost(self) -> float:
        """Upper bound on the per-step energy + switching cost (guardrail Issue 2)."""
        max_it = float(np.sum(POWER_FRAC[3] * self.peak_kw) * STEP_HOURS)
        max_pue = max(max(row) for row in PUE_TABLE)
        return self.w_energy * max_it * max_pue + self.w_switch * float(self.e_switch_kwh.max())

    def describe(self) -> Dict:
        return {
            "n_clusters": self.n_clusters,
            "u_levels": 4,
            "u_labels": list(U_LABELS),
            "tau_levels": 3,
            "tau_labels": list(TAU_LABELS),
            "q_max": self.q_max,
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "lambda_solve": self.lambda_arrivals,
            "step_minutes": STEP_MINUTES,
            "peak_kw": self.peak_kw.tolist(),
            "cap_table": self.cap_table.tolist(),
            "power_frac": list(POWER_FRAC),
            "pue_table": [list(r) for r in PUE_TABLE],
            "e_switch_kwh": self.e_switch_kwh.tolist(),
            "w_energy": self.w_energy,
            "w_switch": self.w_switch,
            "w_sla": self.w_sla,
            "penalty_drop": self.penalty_drop,
            "penalty_hot": self.penalty_hot,
            "service_rule": "served = min(q, cap_i(u_target)); other clusters drain 1 level/step",
            "thermal_rule": "pressure = n_busy + (0 if cool=high else 1) - 2; tau += sign(pressure)",
        }


# --------------------------------------------------------------------------------------
# Automated pre-solver validation (guardrail Issue 5)
# --------------------------------------------------------------------------------------
def assert_valid_mdp(env: DataCenterMDP) -> Dict[str, float]:
    """Hard checks that must pass before any solver is allowed to run.

    Returns a dict of measured quantities for logging/reporting.
    Raises AssertionError on the first violated check.
    """
    assert env.PROB is not None, "exact tensors not built (build_exact=True required)"

    # 1. no negative probabilities
    min_p = float(env.PROB.min())
    assert min_p >= 0.0, f"negative transition probability found: {min_p}"

    # 2. every (s, a) row sums to 1.0 +/- 1e-6 (both in the padded arrays and
    #    in the exposed sparse matrices)
    dev = float(np.abs(env.PROB.sum(axis=-1) - 1.0).max())
    assert dev <= 1e-6, f"transition rows deviate from 1.0 by {dev:.3e}"
    for a, P_a in enumerate(env.P):
        row_sums = np.asarray(P_a.sum(axis=1)).ravel()
        d = float(np.abs(row_sums - 1.0).max())
        assert d <= 1e-6, f"sparse P[{a}] rows deviate by {d:.3e}"

    # 3. discount factor inside the required band
    assert 0.90 <= env.gamma <= 0.99, f"gamma={env.gamma} outside [0.90, 0.99]"

    # 4. finite rewards everywhere (values cannot diverge to +/-inf)
    assert bool(np.isfinite(env.REW).all()) and bool(np.isfinite(env.R_bar).all()), \
        "non-finite reward detected"

    # 5. every state keeps at least one valid action (mask can never dead-end)
    n_valid = env.mask.sum(axis=1)
    assert int(n_valid.min()) >= 1, "dead-end state with empty action mask"

    # 6. anti lazy-shutdown (reward hacking): w3 * P_drop > 5 * max step cost
    max_cost = env.max_step_cost()
    drop_side = env.w_sla * env.penalty_drop
    assert drop_side > 5.0 * max_cost, (
        f"SLA guardrail violated: w3*P_drop={drop_side:.2f} "
        f"<= 5*max_step_cost={5.0 * max_cost:.2f}"
    )

    # 7. anti thrashing: E_switch >= 3 time steps of idle power (spec asks 3-5)
    for es, idle in zip(env.e_switch_kwh, env.idle_kwh):
        assert es >= 3.0 * idle, f"E_switch {es:.4f} < 3 x idle {3 * idle:.4f}"

    # 8. boundary clamping: the folded tail never pushes q' beyond Q_MAX
    u_next = np.array([env.idx_to_state(int(i))[1] for i in env.NEXT.ravel()])
    assert int(u_next.max()) <= env.q_max, "queue exceeded Q_MAX in transition tensor"

    return {
        "min_probability": min_p,
        "max_row_sum_deviation": dev,
        "gamma": env.gamma,
        "min_valid_actions_per_state": int(n_valid.min()),
        "max_valid_actions_per_state": int(n_valid.max()),
        "max_step_cost_units": max_cost,
        "w3_times_penalty_drop": drop_side,
        "guardrail_ratio": drop_side / (5.0 * max_cost),
        "min_switch_to_idle_ratio": float(
            min(es / idle for es, idle in zip(env.e_switch_kwh, env.idle_kwh))
        ),
        "nonzeros_P": int(sum(P.nnz for P in env.P)),
    }
