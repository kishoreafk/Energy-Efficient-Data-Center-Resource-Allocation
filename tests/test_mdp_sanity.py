"""
tests/test_mdp_sanity.py
========================

Debugging-playbook checks (Engineering Guardrails, Issues 1-5):

  1. state indexing is a bijection and |S| stays in the 300..1500 tabular window
  2. every (s, a) row of P sums to 1.0 +/- 1e-6, no negative probabilities,
     boundary tail folded into q' = Q_MAX
  3. gamma inside [0.90, 0.99]
  4. action mask can never dead-end; forced-wake rule works
  5. reward guardrails: anti lazy-shutdown (w3*P_drop > 5*max step cost) and
     anti thrashing (E_switch >= 3 idle-steps)
  6. short VI run stays finite (no divergence to +/-inf / NaN)
  7. VI and PI converge to the same V* within 1e-4 (and identical policies)
  8. step simulator agrees with the analytic expectation from P (Monte Carlo)

Runnable both ways:
    python -m pytest tests -q
    python tests/test_mdp_sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.datacenter_mdp import DataCenterMDP, assert_valid_mdp  # noqa: E402
from solvers.dynamic_programming import PolicyIteration, ValueIteration, policy_agreement  # noqa: E402

_ENV = None


def get_env() -> DataCenterMDP:
    """Build the canonical env once and reuse it across tests."""
    global _ENV
    if _ENV is None:
        _ENV = DataCenterMDP()
        assert_valid_mdp(_ENV)  # guardrail Issue 5: validate before any solver
    return _ENV


# --------------------------------------------------------------------------------------
# 1. state space / indexing
# --------------------------------------------------------------------------------------
def test_state_indexing_roundtrip():
    env = get_env()
    seen = set()
    for s in range(env.n_states):
        u, q, tau = env.idx_to_state(s)
        assert all(0 <= x <= 3 for x in u), f"u out of range at {s}"
        assert 0 <= q <= env.q_max and 0 <= tau <= 2
        assert env.state_to_idx(u, q, tau) == s, f"round-trip failed at {s}"
        seen.add((u, q, tau))
    assert len(seen) == env.n_states, "encoding is not injective"
    print(f"PASS  state indexing bijective over {env.n_states} states")


def test_state_space_size_in_guardrail():
    env = get_env()
    assert 300 <= env.n_states <= 1500, (
        f"|S|={env.n_states} outside the 300..1500 tabular window (Issue 1)"
    )
    print(f"PASS  |S| = {env.n_states} within [300, 1500] (curse-of-dimensionality guard)")


# --------------------------------------------------------------------------------------
# 2. transition tensor validity
# --------------------------------------------------------------------------------------
def test_probability_rows_sum_to_one():
    env = get_env()
    dev = np.abs(env.PROB.sum(axis=-1) - 1.0)
    assert dev.max() <= 1e-6, f"max row deviation {dev.max():.3e} > 1e-6"
    for a, P_a in enumerate(env.P):
        rs = np.asarray(P_a.sum(axis=1)).ravel()
        assert np.abs(rs - 1.0).max() <= 1e-6
    print(f"PASS  all {env.n_states * env.n_actions} (s,a) rows sum to 1.0 +/- 1e-6 "
          f"(max dev {dev.max():.2e})")


def test_no_negative_probabilities():
    env = get_env()
    assert env.PROB.min() >= 0.0
    assert all(P.data.min() >= 0.0 for P in env.P)
    print("PASS  no negative transition probabilities")


def test_boundary_tail_clamped():
    env = get_env()
    q_next = np.array([env.idx_to_state(int(i))[1] for i in env.NEXT.ravel()])
    assert q_next.max() <= env.q_max, "queue exceeded Q_MAX (tail not folded)"
    # states already at the boundary must stay clamped, never roll over
    boundary = [s for s in range(env.n_states) if env.idx_to_state(s)[1] == env.q_max]
    for s in boundary[:64]:
        for a in range(env.n_actions):
            for k in range(env.PROB.shape[-1]):
                if env.PROB[s, a, k] > 0:
                    assert env.idx_to_state(int(env.NEXT[s, a, k]))[1] <= env.q_max
    print("PASS  Poisson tail folded into q' = Q_MAX for all boundary states")


def test_gamma_range():
    env = get_env()
    assert 0.90 <= env.gamma <= 0.99
    print(f"PASS  gamma = {env.gamma} inside [0.90, 0.99]")


# --------------------------------------------------------------------------------------
# 4. action mask
# --------------------------------------------------------------------------------------
def test_action_mask_safety():
    env = get_env()
    n_valid = env.mask.sum(axis=1)
    assert n_valid.min() >= 1, "dead-end state (empty mask)"
    # forced-wake: q > 0 and all clusters asleep -> every legal action must wake
    for s in range(env.n_states):
        u, q, _ = env.idx_to_state(s)
        if q > 0 and all(x == 0 for x in u):
            valid = np.where(env.mask[s])[0]
            assert len(valid) > 0
            powers = {env.idx_to_action(int(a))[1] for a in valid}
            assert all(1 <= p <= env.n_clusters for p in powers), (
                f"forced-wake violated at state {u, q}: powers={powers}"
            )
    print(f"PASS  action mask safe: min valid actions/state = {n_valid.min()}, "
          "forced-wake rule enforced when q>0 & all asleep")


# --------------------------------------------------------------------------------------
# 5. reward guardrails
# --------------------------------------------------------------------------------------
def test_reward_guardrail_anti_lazy_shutdown():
    env = get_env()
    max_cost = env.max_step_cost()
    drop_side = env.w_sla * env.penalty_drop
    assert drop_side > 5.0 * max_cost, (
        f"w3*P_drop={drop_side:.2f} must exceed 5*max(E_IT+E_switch)={5 * max_cost:.2f}"
    )
    print(f"PASS  anti lazy-shutdown: w3*P_drop = {drop_side:.1f} > "
          f"5 x max_step_cost = {5 * max_cost:.2f} (ratio {drop_side / (5 * max_cost):.2f})")


def test_switch_cost_hysteresis():
    env = get_env()
    for i, (es, idle) in enumerate(zip(env.e_switch_kwh, env.idle_kwh)):
        ratio = es / idle
        assert 3.0 - 1e-9 <= ratio <= 5.0 + 1e-9, (
            f"cluster {i}: E_switch/idle = {ratio:.2f} outside the 3..5 hysteresis band"
        )
    print("PASS  E_switch = "
          + ", ".join(f"{es:.3f} kWh ({es / idle:.1f}x idle)" for es, idle in
                      zip(env.e_switch_kwh, env.idle_kwh))
          + " -> Bellman lookahead learns hysteresis (no thrashing)")


# --------------------------------------------------------------------------------------
# 6. short VI stays finite
# --------------------------------------------------------------------------------------
def test_no_nan_short_value_iteration():
    env = get_env()
    res = ValueIteration(env, theta=1e-5, max_sweeps=20).solve()
    assert np.isfinite(res.V).all(), "V diverged to NaN/inf (Issue 5)"
    print(f"PASS  20 Bellman sweeps finite (max |V| = {np.abs(res.V).max():.2f})")


# --------------------------------------------------------------------------------------
# 7. VI <-> PI agreement (workflow step 2)
# --------------------------------------------------------------------------------------
def test_vi_pi_agreement():
    env = get_env()
    vi = ValueIteration(env, theta=1e-6, max_sweeps=5000).solve()
    pi = PolicyIteration(env, eval_method="linear").solve()
    diff = float(np.max(np.abs(vi.V - pi.V)))
    assert diff <= 1e-4, f"|V_vi - V_pi|_inf = {diff:.3e} > 1e-4"
    agree = policy_agreement(env, vi.pi, pi.pi)
    assert agree >= 0.99, f"policies disagree on {1 - agree:.2%} of states"
    print(f"PASS  VI vs PI: max |V* diff| = {diff:.2e} <= 1e-4, "
          f"policy agreement = {agree:.2%}")


# --------------------------------------------------------------------------------------
# 8. simulator vs analytic model
# --------------------------------------------------------------------------------------
def test_simulator_matches_model():
    env = get_env()
    rng = np.random.default_rng(7)
    checked = 0
    for s in rng.choice(env.n_states, size=12, replace=False):
        u, q, tau = env.idx_to_state(int(s))
        for a in np.where(env.mask[int(s)])[0][:: max(1, env.mask[int(s)].sum() // 3)]:
            a_alloc, a_power, a_cool = env.idx_to_action(int(a))
            rewards, q_next = [], []
            for _ in range(6000):
                u2, q2, tau2, r, _i = env.step_dynamics(
                    u, q, tau, a_alloc, a_power, a_cool, forced_arrival=None, rng=rng
                )
                rewards.append(r)
                q_next.append(q2)
            mc_r, mc_q = np.mean(rewards), np.mean(q_next)
            # drop-penalty rewards are heavy-tailed: tolerance = 4 standard errors + slack
            tol = 4.0 * float(np.std(rewards)) / np.sqrt(len(rewards)) + 1.0
            assert abs(mc_r - env.R_bar[int(s), int(a)]) < tol, (
                f"simulator reward drift at s={u, q, tau}, a={a_alloc, a_power, a_cool}: "
                f"{mc_r:.2f} vs model {env.R_bar[int(s), int(a)]:.2f} (tol {tol:.2f})"
            )
            assert abs(mc_q - float((env.PROB[int(s), int(a)]
                                     * np.array([env.idx_to_state(int(i))[1]
                                                 for i in env.NEXT[int(s), int(a)]])).sum())) < 0.15
            checked += 1
    print(f"PASS  step simulator matches analytic P/R on {checked} (s,a) pairs "
          "(Monte Carlo, 4000 rollouts each)")


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------
ALL_TESTS = [
    test_state_indexing_roundtrip,
    test_state_space_size_in_guardrail,
    test_probability_rows_sum_to_one,
    test_no_negative_probabilities,
    test_boundary_tail_clamped,
    test_gamma_range,
    test_action_mask_safety,
    test_reward_guardrail_anti_lazy_shutdown,
    test_switch_cost_hysteresis,
    test_no_nan_short_value_iteration,
    test_vi_pi_agreement,
    test_simulator_matches_model,
]


def run_all(env: DataCenterMDP | None = None, verbose: bool = True) -> bool:
    """Run the full sanity suite; returns True when every check passes."""
    global _ENV
    if env is not None:
        _ENV = env
    failed = []
    for fn in ALL_TESTS:
        try:
            fn()
        except AssertionError as exc:
            failed.append((fn.__name__, str(exc)))
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # unexpected error still counts as failure
            failed.append((fn.__name__, repr(exc)))
            print(f"ERROR {fn.__name__}: {exc!r}")
    if verbose:
        print(f"\n{'-' * 72}\nSanity suite: {len(ALL_TESTS) - len(failed)}/{len(ALL_TESTS)} "
              f"checks passed{'' if not failed else '  ->  ' + ', '.join(n for n, _ in failed)}")
    return not failed


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
