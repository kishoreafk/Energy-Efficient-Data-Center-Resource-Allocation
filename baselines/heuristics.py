"""
baselines/heuristics.py
=======================

Hand-crafted operating policies used as baselines:

* RoundRobinAlwaysOn            — never sleeps a server; routes jobs round-robin
                                  across all active clusters.
* GreedyThresholdConsolidation  — packs tasks onto the lowest-index (most
                                  energy-efficient) cluster until it saturates,
                                  wakes the next tier only above a queue
                                  threshold, and sleeps idle servers after a
                                  patience period (built-in hysteresis).

Both expose the same `act(u, q, tau, t) -> action_index` interface as the DP
policies so the evaluation harness treats them uniformly.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from env.datacenter_mdp import DataCenterMDP


class BaseHeuristic:
    name = "base"

    def __init__(self, env: DataCenterMDP) -> None:
        self.env = env

    def act(self, u: Sequence[int], q: int, tau: int, t: int) -> int:  # pragma: no cover
        raise NotImplementedError

    def _index(self, a_alloc: int, a_power: int, a_cool: int) -> int:
        return self.env.action_to_idx(a_alloc, a_power, a_cool)

    def _safe(self, a_alloc: int, a_power: int, a_cool: int, u, q) -> int:
        """Return the action index, falling back to a legal conservative action
        if the heuristic proposal is masked out in this state."""
        if self.env.action_is_valid(u, q, a_alloc, a_power):
            return self._index(a_alloc, a_power, a_cool)
        if self.env.action_is_valid(u, q, self.env.n_clusters, 0):
            return self._index(self.env.n_clusters, 0, a_cool)  # hold + noop
        # dead-end state (q>0, all asleep): wake the first sleeping cluster
        for i in range(self.env.n_clusters):
            if u[i] == 0 and self.env.action_is_valid(u, q, a_alloc, i + 1):
                return self._index(self.env.n_clusters, i + 1, a_cool)
        # ultimate fallback: first valid action
        s = self.env.state_to_idx(u, q, 0)
        return int(self.env.valid_actions(s)[0])

    @staticmethod
    def _cool(tau: int, hot_only: bool = False) -> int:
        return 1 if tau >= (2 if hot_only else 1) else 0


class RoundRobinAlwaysOn(BaseHeuristic):
    """All servers stay active forever; tasks are distributed evenly across the
    pool: the queue goes to the active cluster with the highest current service
    capacity (ties broken round-robin) - the standard load-balancer abstraction."""

    name = "Always-On (Round-Robin)"

    def __init__(self, env: DataCenterMDP) -> None:
        super().__init__(env)
        self.rr = 0

    def act(self, u, q, tau, t) -> int:
        u = list(u)
        N = self.env.n_clusters
        # 1. power: keep everything alive (wake one sleeping cluster per step)
        a_power = 0
        woke = -1
        for i in range(N):
            if u[i] == 0:
                a_power, woke = i + 1, i
                break
        # 2. routing: even distribution -> most-capable active cluster, RR tie-break
        a_alloc = N  # hold
        if q > 0:
            active = [i for i in range(N) if u[i] >= 1]
            if active:
                caps = [self.env.cap_table[i, u[i]] for i in active]
                best = max(caps)
                pool = [i for i, c in zip(active, caps) if c == best]
                a_alloc = pool[self.rr % len(pool)]
                self.rr += 1
            elif woke >= 0:
                a_alloc = woke  # serve with the freshly-woken cluster
        return self._safe(a_alloc, a_power, self._cool(tau), u, q)


class GreedyThresholdConsolidation(BaseHeuristic):
    """Pack onto the efficient tiers first; scale out only under pressure.

    - wake the next (lowest-index) sleeping cluster when the queue reaches
      `wake_q_threshold` or the primary cluster is saturated (u == 3) with
      work pending;
    - put a cluster to sleep only after it has been Idle for `sleep_patience`
      consecutive steps and at least one other cluster remains active
      (hysteresis against thrashing).
    """

    name = "Greedy Consolidation"

    def __init__(self, env: DataCenterMDP, wake_q_threshold: int = 2,
                 sleep_patience: int = 2) -> None:
        super().__init__(env)
        self.wake_q_threshold = int(wake_q_threshold)
        self.sleep_patience = int(sleep_patience)
        self.idle_steps = np.zeros(env.n_clusters, dtype=int)

    def act(self, u, q, tau, t) -> int:
        u = list(u)
        N = self.env.n_clusters
        for i in range(N):  # hysteresis counters
            self.idle_steps[i] = self.idle_steps[i] + 1 if u[i] == 1 else 0

        active = [i for i in range(N) if u[i] >= 1]
        sleeping = [i for i in range(N) if u[i] == 0]

        # 1. power decision
        a_power = 0
        woke = -1
        primary_full = bool(active) and u[active[0]] == 3
        if q >= self.wake_q_threshold or (q > 0 and primary_full):
            if sleeping:
                a_power, woke = sleeping[0] + 1, sleeping[0]
        elif len(active) > 1:
            idle_long = [i for i in active
                         if u[i] == 1 and self.idle_steps[i] >= self.sleep_patience]
            if idle_long:
                a_power = N + max(idle_long) + 1  # sleep the highest idle tier

        # 2. routing: first active cluster with headroom, else the busiest tail
        a_alloc = N
        if q > 0:
            if woke >= 0:
                a_alloc = woke
            else:
                with_headroom = [i for i in active if u[i] < 3]
                a_alloc = with_headroom[0] if with_headroom else (active[-1] if active else N)
        # 3. cooling: high only when Hot, or Warm under heavy load
        busy = sum(1 for x in u if x >= 2)
        a_cool = 1 if (tau >= 2 or (tau == 1 and busy >= 2)) else 0
        return self._safe(a_alloc, a_power, a_cool, u, q)
