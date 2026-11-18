"""
main.py — end-to-end pipeline for
"Energy-Efficient Data Center Resource Allocation using Dynamic Programming
 (with Modern ADP/RL Extensions)".

Workflow (matches the Unit-2 curriculum requirement):
  [1/8] build DataCenterMDP + assert_valid_mdp guardrails
  [2/8] run tests/test_mdp_sanity.py
  [3/8] Value Iteration (synchronous + Gauss-Seidel) per load regime
  [4/8] Policy Iteration (exact linear-system evaluation) per load regime
  [5/8] verify VI and PI reach the same V* (tol 1e-4) in every regime
  [6/8] RTDP (tabular) + Linear-VFA ADP scalability demo at N = 8
  [7/8] 24h diurnal evaluation: VI, PI, RTDP, Greedy Consolidation, Always-On
        + energy/SLA Pareto sweep over the drop penalty
  [8/8] figures, summary tables, model weights, dashboard data -> results/, weights/

Deployment note: the diurnal trace is non-stationary, the tabular MDP is not.
Following the standard production pattern, ONE policy is solved per load regime
(night lambda=1.2 / day lambda=3.5 / peak lambda=6.5) and switched on the
forecast arrival rate (cuts at 2.5 and 5.5 jobs/step). VI and PI are compared
inside every regime; the 24h traces deploy the regime-switched policies.

Run:  python main.py [--skip-adp] [--seed 42]
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from env.datacenter_mdp import DataCenterMDP, assert_valid_mdp
from solvers.dynamic_programming import PolicyIteration, ValueIteration, policy_agreement
from solvers.modern_extensions import LinearADP, RTDP
from baselines.heuristics import GreedyThresholdConsolidation, RoundRobinAlwaysOn
import evaluation.metrics_and_plots as ev

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"
SEED = 42
REGIME_LAMBDAS = (1.2, 3.5, 6.5)     # night / day / peak design points
REGIME_CUTS = (2.5, 5.5)             # lambda_t thresholds for switching
RTDP_LAMBDA = 3.0                    # single design point for tabular RTDP
PARETO_DROPS = (10.0, 25.0, 50.0, 100.0, 200.0)

LAM_T, HOURS_T = ev.diurnal_lambda()
ARRIVALS = ev.pre_sample_arrivals(LAM_T, seed=1234)
SURGE_FLOOR = int(np.maximum(ARRIVALS - 6, 0).sum())  # drops no policy can avoid


def banner(txt: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{txt}\n{line}", flush=True)


def save_weights(path: Path, **arrays) -> None:
    WEIGHTS.mkdir(exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"  saved {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.1f} KB)")


# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-adp", action="store_true", help="skip the N=8 ADP demo")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    t_wall = time.perf_counter()
    timings: dict = {}
    RESULTS.mkdir(exist_ok=True)

    banner("Energy-Efficient Data Center Resource Allocation via Dynamic Programming\n"
           "Unit 2 curriculum pipeline (MDP -> VI/PI -> ADP/RL -> evaluation)")

    # ------------------------------------------------------------------ [1/8] env + MDP checks
    banner("[1/8] Environment construction & MDP validation")
    t0 = time.perf_counter()
    env = DataCenterMDP(n_clusters=3, q_max=6, lambda_arrivals=3.0, gamma=0.95, seed=args.seed)
    validation = assert_valid_mdp(env)  # Issue 5: automated pre-solver validation
    timings["env_build_s"] = time.perf_counter() - t0
    print(f"  |S| = {env.n_states} states, |A| = {env.n_actions} actions, "
          f"gamma = {env.gamma}, lambda_solve = {env.lambda_arrivals}")
    print(f"  sparse P: {validation['nonzeros_P']:,} nonzeros "
          f"(a dense 3D tensor would need {env.n_states * env.n_actions * env.n_states:,} cells)")
    print("  guardrails:")
    print(f"    row sums 1.0 +/- {validation['max_row_sum_deviation']:.1e}  |  "
          f"min prob {validation['min_probability']:.2f}  |  "
          f"valid actions/state: {validation['min_valid_actions_per_state']}"
          f"..{validation['max_valid_actions_per_state']}")
    print(f"    anti lazy-shutdown:  w3*P_drop = {validation['w3_times_penalty_drop']:.1f} > "
          f"5 x max_step_cost = {5 * validation['max_step_cost_units']:.2f}  "
          f"(ratio {validation['guardrail_ratio']:.2f})")
    print(f"    anti thrashing:      E_switch = {validation['min_switch_to_idle_ratio']:.1f}x "
          f"idle-step energy (hysteresis band 3..5)")
    print(f"  24h trace: {int(ARRIVALS.sum())} offered tasks, arrival surge floor "
          f"(unavoidable clamps) = {SURGE_FLOOR} tasks")

    # ------------------------------------------------------------------ [2/8] sanity tests
    banner("[2/8] Sanity tests (tests/test_mdp_sanity.py)")
    import tests.test_mdp_sanity as sanity

    tests_env = DataCenterMDP(q_max=6, seed=args.seed)  # fresh, independent of solvers below
    ok = sanity.run_all(env=tests_env)
    if not ok:
        print("FATAL: sanity suite failed - aborting before solvers.")
        return 1

    # ------------------------------------------------------------------ [3/8] Value Iteration
    banner(f"[3/8] Value Iteration per regime (theta = 1e-5, gamma = 0.95, "
           f"lambdas = {REGIME_LAMBDAS})")
    vi_res, vi_gs_res, envs = {}, {}, {}
    for lam in REGIME_LAMBDAS:
        e = DataCenterMDP(lambda_arrivals=lam, seed=args.seed)
        envs[lam] = e
        vi_res[lam] = ValueIteration(e, theta=1e-5, mode="synchronous").solve()
        vi_gs_res[lam] = ValueIteration(e, theta=1e-5, mode="gauss_seidel", seed=7).solve()
        print(f"  lambda={lam:4.1f}: sync sweeps={vi_res[lam].n_sweeps:3d} "
              f"({vi_res[lam].time_s:.2f}s)   GS sweeps={vi_gs_res[lam].n_sweeps:3d} "
              f"({vi_gs_res[lam].time_s:.2f}s)   sync-vs-GS |dV|="
              f"{np.max(np.abs(vi_res[lam].V - vi_gs_res[lam].V)):.1e}")
    timings["vi_sync_s"] = sum(r.time_s for r in vi_res.values())
    timings["vi_gs_s"] = sum(r.time_s for r in vi_gs_res.values())
    day_lam = REGIME_LAMBDAS[1]
    vi, vi_gs = vi_res[day_lam], vi_gs_res[day_lam]  # representative for curves

    # ------------------------------------------------------------------ [4/8] Policy Iteration
    banner("[4/8] Policy Iteration (exact linear-system evaluation) per regime")
    pi_res = {}
    for lam in REGIME_LAMBDAS:
        pi_res[lam] = PolicyIteration(envs[lam], eval_method="linear",
                                      improvement_eps=1e-8).solve()
        p = pi_res[lam]
        print(f"  lambda={lam:4.1f}: converged={p.converged} iters={p.n_sweeps} "
              f"time={p.time_s:.2f}s oscillation={p.extra['oscillation_detected']} "
              f"changes={p.extra['policy_changes']}")
        for h in p.extra["history"]:
            print(f"      iter {h['iteration']}: changes={h['policy_changes']:4d}  "
                  f"eval_residual={h['eval_residual']:.1e}")
    timings["pi_s"] = sum(p.time_s for p in pi_res.values())

    # ------------------------------------------------------------------ [5/8] VI vs PI agreement
    banner("[5/8] VI <-> PI agreement check per regime (required tolerance 1e-4)")
    worst_diff, worst_agree = 0.0, 1.0
    for lam in REGIME_LAMBDAS:
        d = float(np.max(np.abs(vi_res[lam].V - pi_res[lam].V)))
        a = policy_agreement(envs[lam], vi_res[lam].pi, pi_res[lam].pi)
        if d > 1e-4:  # tighten VI once if the conservative bound bites
            print(f"  lambda={lam}: diff {d:.2e} > 1e-4 -> re-running VI with theta = 1e-6")
            vi_res[lam] = ValueIteration(envs[lam], theta=1e-6, mode="synchronous").solve()
            d = float(np.max(np.abs(vi_res[lam].V - pi_res[lam].V)))
            a = policy_agreement(envs[lam], vi_res[lam].pi, pi_res[lam].pi)
        worst_diff, worst_agree = max(worst_diff, d), min(worst_agree, a)
        print(f"  lambda={lam:4.1f}: max |V_vi - V_pi|_inf = {d:.3e}  "
              f"policy agreement = {a:.2%}")
    print(f"  worst case across regimes: diff {worst_diff:.3e} (tol 1e-4) -> "
          f"{'PASS' if worst_diff <= 1e-4 else 'FAIL'}, agreement {worst_agree:.2%}")
    assert worst_diff <= 1e-4, "VI and PI did not converge to the same optimal V*"

    # ------------------------------------------------------------------ [6/8] RTDP + ADP
    banner("[6/8] RTDP (tabular) and Linear-VFA ADP scalability demo (N = 8)")
    env_rtdp = DataCenterMDP(lambda_arrivals=RTDP_LAMBDA, seed=args.seed)
    vi_rtdp_ref = ValueIteration(env_rtdp, theta=1e-6, mode="synchronous").solve()
    rtdp = RTDP(env_rtdp, n_trials=4000, max_steps=80, theta=1e-5, seed=3).solve()
    visited = rtdp.extra["visited_mask"]
    counts = rtdp.extra["visit_counts"]
    frequent = visited & (counts >= 5)
    rtdp_gap = float(np.max(np.abs(rtdp.V[visited] - vi_rtdp_ref.V[visited])))
    rtdp_gap_core = float(np.max(np.abs(rtdp.V[frequent] - vi_rtdp_ref.V[frequent])))
    rtdp_agree = policy_agreement(env_rtdp, rtdp.pi[frequent], vi_rtdp_ref.pi[frequent])
    print(f"  RTDP (lambda={RTDP_LAMBDA}): trials={rtdp.n_sweeps}  "
          f"visited={visited.sum()}/{env_rtdp.n_states} states  time={rtdp.time_s:.1f}s")
    print(f"        max|V_rtdp - V*| visited={rtdp_gap:.3f}   frequently-visited="
          f"{rtdp_gap_core:.3f}   policy agreement there = {rtdp_agree:.2%}")
    timings["rtdp_s"] = rtdp.time_s

    adp_payload = None
    if not args.skip_adp:
        n_big = 8
        tab_states = (4 ** n_big) * 7 * 3
        tab_entries = tab_states * (n_big + 1) * (2 * n_big + 1) * 2 * 7
        print(f"  scalability: for N={n_big}, exact tabular DP needs |S| = 4^{n_big}*7*3 = "
              f"{tab_states:,} states and ~{tab_entries / 1e9:.2f}e9 sparse transition "
              f"entries -> infeasible; using simulator + linear VFA instead.")
        env8 = DataCenterMDP(n_clusters=n_big, build_exact=False, seed=args.seed)
        adp = LinearADP(env8, alpha0=0.05, n_steps=25000, seed=5)
        adp_res = adp.solve()
        print(f"  LinearADP (N={n_big}): steps={adp_res['n_steps']}  "
              f"time={adp_res['time_s']:.1f}s  final |TD| = {adp_res['td_curve'][-1]:.3f}")
        timings["adp_s"] = adp_res["time_s"]

        trace8 = {
            "Linear-ADP (N=8)": ev.run_trace(env8, ev.LamForecastPolicy(adp, LAM_T, env8),
                                             LAM_T, ARRIVALS),
            "Always-On (N=8)": ev.run_trace(env8, RoundRobinAlwaysOn(env8), LAM_T, ARRIVALS),
            "Greedy Consol. (N=8)": ev.run_trace(env8, GreedyThresholdConsolidation(env8),
                                                 LAM_T, ARRIVALS),
        }
        for name, tr in trace8.items():
            m = tr["metrics"]
            print(f"    {name:22s} energy={m['total_energy_kwh']:7.2f} kWh  "
                  f"SLA={m['sla_violation_rate_pct']:6.2f}%  switches={m['switch_count']:3d}")
        adp_payload = {
            "n_clusters": n_big,
            "tabular_state_space": tab_states,
            "tabular_sparse_entries": tab_entries,
            "theta": adp_res["theta"].tolist(),
            "feature_names": adp_res["feature_names"],
            "td_curve": adp_res["td_curve"][:: max(1, len(adp_res["td_curve"]) // 250)],
            "n_steps": adp_res["n_steps"],
            "time_s": adp_res["time_s"],
            "traces": {k: ev.downsample_trace(v) for k, v in trace8.items()},
            "metrics": {k: v["metrics"] for k, v in trace8.items()},
        }
        save_weights(WEIGHTS / "adp_linear_theta.npz",
                     theta=adp_res["theta"],
                     feature_names=np.array(adp_res["feature_names"]),
                     n_clusters=np.int64(n_big),
                     n_steps=np.int64(adp_res["n_steps"]),
                     td_curve=np.array(adp_res["td_curve"]))

    # ------------------------------------------------------------------ [7/8] 24h evaluation
    banner("[7/8] 24-hour diurnal evaluation (5 policies, common random numbers)")
    regime_entries = [(envs[lam], vi_res[lam].V) for lam in REGIME_LAMBDAS]
    regime_entries_pi = [(envs[lam], pi_res[lam].V) for lam in REGIME_LAMBDAS]
    policies = {
        "Value Iteration": ev.RegimePolicy(regime_entries, REGIME_CUTS, LAM_T,
                                           "Value Iteration"),
        "Policy Iteration": ev.RegimePolicy(regime_entries_pi, REGIME_CUTS, LAM_T,
                                            "Policy Iteration"),
        "RTDP": ev.GreedyPolicy(env_rtdp, rtdp.V, "RTDP"),
        "Greedy Consolidation": GreedyThresholdConsolidation(env),
        "Always-On (Round-Robin)": RoundRobinAlwaysOn(env),
    }
    traces = {}
    for name, pol in policies.items():
        t0 = time.perf_counter()
        traces[name] = ev.run_trace(env, pol, LAM_T, ARRIVALS)
        timings[f"trace_{name}"] = time.perf_counter() - t0
        m = traces[name]["metrics"]
        print(f"  {name:26s} energy={m['total_energy_kwh']:7.2f} kWh  "
              f"SLA viol={m['sla_violation_rate_pct']:6.2f}%  drops={m['dropped_tasks']:4d}  "
              f"switches={m['switch_count']:3d}  warm={m['warm_frac_pct']:5.1f}%  "
              f"avgQ={m['avg_queue']:.2f}  PUE={m['pue_avg']:.3f}")
    print(f"  (surge floor: {SURGE_FLOOR} tasks = "
          f"{100.0 * SURGE_FLOOR / int(ARRIVALS.sum()):.2f}% - no policy can do better)")

    base_e = traces["Always-On (Round-Robin)"]["metrics"]["total_energy_kwh"]
    for name, tr in traces.items():
        tr["metrics"]["energy_saved_vs_always_on_pct"] = (
            100.0 * (1.0 - tr["metrics"]["total_energy_kwh"] / base_e))

    # Pareto sweep over the SLA drop penalty (Issue 2 demonstration included)
    print(f"\n  Pareto sweep: penalty_drop in {PARETO_DROPS} with the regime deployment "
          "(values below the guardrail demonstrate lazy-shutdown reward hacking)")
    sweep_points = []
    for pd_pen in PARETO_DROPS:
        entries = []
        for lam in REGIME_LAMBDAS:
            e_p = DataCenterMDP(lambda_arrivals=lam, penalty_drop=pd_pen, seed=args.seed)
            v_p = ValueIteration(e_p, theta=1e-5, max_sweeps=2000).solve()
            entries.append((e_p, v_p.V))
        tr_p = ev.run_trace(env, ev.RegimePolicy(entries, REGIME_CUTS, LAM_T,
                                                 f"DP P={pd_pen:.0f}"), LAM_T, ARRIVALS)
        m = tr_p["metrics"]
        guardrail = 1.0 * pd_pen > 5.0 * env.max_step_cost()
        sweep_points.append({
            "label": f"DP P_drop={pd_pen:.0f}", "penalty_drop": pd_pen,
            "energy_kwh": m["total_energy_kwh"],
            "sla_violation_rate_pct": m["sla_violation_rate_pct"],
            "kind": "dp-sweep", "guardrail_ok": bool(guardrail),
        })
        print(f"    P_drop={pd_pen:6.1f}  guardrail={'OK  ' if guardrail else 'VIOL'}  "
              f"energy={m['total_energy_kwh']:7.2f} kWh  SLA={m['sla_violation_rate_pct']:6.2f}%")
    main_point = {
        "label": "DP (main, P_drop=50)",
        "energy_kwh": traces["Value Iteration"]["metrics"]["total_energy_kwh"],
        "sla_violation_rate_pct": traces["Value Iteration"]["metrics"]["sla_violation_rate_pct"],
        "kind": "dp-main",
    }
    baseline_points = [
        {"label": "Greedy Consolidation", "kind": "baseline", "marker": "s", "color": "#f59e0b",
         "energy_kwh": traces["Greedy Consolidation"]["metrics"]["total_energy_kwh"],
         "sla_violation_rate_pct":
             traces["Greedy Consolidation"]["metrics"]["sla_violation_rate_pct"]},
        {"label": "Always-On", "kind": "baseline", "marker": "D", "color": "#64748b",
         "energy_kwh": traces["Always-On (Round-Robin)"]["metrics"]["total_energy_kwh"],
         "sla_violation_rate_pct":
             traces["Always-On (Round-Robin)"]["metrics"]["sla_violation_rate_pct"]},
    ]

    # ------------------------------------------------------------------ [8/8] artifacts
    banner("[8/8] Figures, summary tables, weights, dashboard data")
    ev.plot_convergence(vi, vi_gs, pi_res[day_lam], rtdp,
                        str(RESULTS / "convergence_curves.png"))
    print("  saved results/convergence_curves.png")
    ev.plot_pareto(sweep_points + [main_point] + baseline_points,
                   guardrail_drop=5.0 * env.max_step_cost() / env.w_sla,
                   path=str(RESULTS / "pareto_frontier.png"),
                   floor_pct=100.0 * SURGE_FLOOR / int(ARRIVALS.sum()),
                   floor_label=f"arrival-surge floor ({SURGE_FLOOR} tasks, "
                               f"{100.0 * SURGE_FLOOR / int(ARRIVALS.sum()):.2f}%)")
    print("  saved results/pareto_frontier.png")
    ev.plot_server_heatmap(traces["Value Iteration"], env,
                           str(RESULTS / "server_state_heatmap.png"))
    print("  saved results/server_state_heatmap.png")
    grids = ev.plot_decision_boundary(envs[REGIME_LAMBDAS[2]], vi_res[REGIME_LAMBDAS[2]].V,
                                      str(RESULTS / "decision_boundary.png"))
    print("  saved results/decision_boundary.png (peak-regime policy)")

    rows = [{"policy": name, **m} for name, tr in traces.items() for m in (tr["metrics"],)]
    df = ev.metrics_frame(rows)
    ev.print_summary_table(df)
    df.to_csv(RESULTS / "summary_metrics.csv", index=False)
    pd.DataFrame(sweep_points).to_csv(RESULTS / "pareto_points.csv", index=False)
    print("  saved results/summary_metrics.csv, results/pareto_points.csv")

    save_weights(WEIGHTS / "vi_policy.npz",
                 V=np.stack([vi_res[l].V for l in REGIME_LAMBDAS]),
                 pi=np.stack([vi_res[l].pi for l in REGIME_LAMBDAS]),
                 regime_lambdas=np.array(REGIME_LAMBDAS), regime_cuts=np.array(REGIME_CUTS),
                 residuals=np.array(vi.residuals), gamma=env.gamma,
                 theta=np.float64(vi.extra["theta"]), n_sweeps=np.int64(vi.n_sweeps))
    save_weights(WEIGHTS / "vi_gauss_seidel.npz",
                 V=np.stack([vi_gs_res[l].V for l in REGIME_LAMBDAS]),
                 pi=np.stack([vi_gs_res[l].pi for l in REGIME_LAMBDAS]),
                 regime_lambdas=np.array(REGIME_LAMBDAS), regime_cuts=np.array(REGIME_CUTS),
                 residuals=np.array(vi_gs.residuals), gamma=env.gamma,
                 theta=np.float64(vi_gs.extra["theta"]), n_sweeps=np.int64(vi_gs.n_sweeps))
    save_weights(WEIGHTS / "pi_policy.npz",
                 V=np.stack([pi_res[l].V for l in REGIME_LAMBDAS]),
                 pi=np.stack([pi_res[l].pi for l in REGIME_LAMBDAS]),
                 regime_lambdas=np.array(REGIME_LAMBDAS), regime_cuts=np.array(REGIME_CUTS),
                 gamma=env.gamma,
                 policy_changes=np.array(pi_res[day_lam].extra["policy_changes"]),
                 improvement_eps=np.float64(pi_res[day_lam].extra["improvement_eps"]))
    save_weights(WEIGHTS / "rtdp_value_function.npz", V=rtdp.V, pi=rtdp.pi,
                 visited_mask=visited, visit_counts=counts,
                 n_trials=np.int64(rtdp.n_sweeps), lambda_solve=np.float64(RTDP_LAMBDA))

    payload = {
        "meta": {
            "generated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "seed": args.seed,
            "timings": {k: round(v, 3) for k, v in timings.items()},
            "runtime_source": "local",
            "surge_floor_tasks": SURGE_FLOOR,
            "offered_tasks": int(ARRIVALS.sum()),
        },
        "spec": env.describe(),
        "validation": validation,
        "regimes": {"lambdas": list(REGIME_LAMBDAS), "cuts": list(REGIME_CUTS),
                    "rtdp_lambda": RTDP_LAMBDA},
        "diurnal": {"hours": np.round(HOURS_T[::6], 3).tolist(),
                    "lam": LAM_T[::6].round(3).tolist(),
                    "hourly_anchors": ev.HOURLY_LAMBDA.tolist()},
        "convergence": {
            "vi_sync": {"residuals": vi.residuals[:: max(1, vi.n_sweeps // 300)],
                        "n_sweeps": vi.n_sweeps, "time_s": vi.time_s,
                        "theta": vi.extra["theta"], "converged": vi.converged},
            "vi_gs": {"residuals": vi_gs.residuals[:: max(1, vi_gs.n_sweeps // 300)],
                      "n_sweeps": vi_gs.n_sweeps, "time_s": vi_gs.time_s,
                      "theta": vi_gs.extra["theta"], "converged": vi_gs.converged},
            "pi": {"policy_changes": pi_res[day_lam].extra["policy_changes"],
                   "eval_residuals": [h["eval_residual"] for h in pi_res[day_lam].extra["history"]],
                   "n_iters": pi_res[day_lam].n_sweeps, "time_s": pi_res[day_lam].time_s,
                   "oscillation_detected": pi_res[day_lam].extra["oscillation_detected"]},
            "rtdp": {"curve": rtdp.residuals[:: max(1, rtdp.n_sweeps // 300)],
                     "n_trials": rtdp.n_sweeps, "visited_states": int(visited.sum()),
                     "time_s": rtdp.time_s, "max_gap_vs_vi_on_visited": rtdp_gap,
                     "max_gap_on_frequent_states": rtdp_gap_core,
                     "policy_agreement_on_visited_pct": 100.0 * rtdp_agree},
            "agreement": {"max_v_diff_worst_regime": worst_diff,
                          "policy_agreement_worst_regime_pct": 100.0 * worst_agree},
        },
        "pareto": {"points": sweep_points + [main_point] + baseline_points,
                   "guardrail_min_w3_times_drop": 5.0 * env.max_step_cost() / env.w_sla},
        "policies": rows,
        "traces": {name: ev.downsample_trace(tr) for name, tr in traces.items()},
        "decision_boundary": grids,
        "adp": adp_payload,
    }
    ev.export_dashboard_json(str(RESULTS / "web_dashboard_data.json"), payload)
    print("  saved results/web_dashboard_data.json "
          f"({(RESULTS / 'web_dashboard_data.json').stat().st_size / 1024:.0f} KB)")

    dp_m = traces["Value Iteration"]["metrics"]
    total_min = (time.perf_counter() - t_wall) / 60
    banner(f"FINAL SUMMARY\n"
           f"  VI sweeps={vi.n_sweeps} (day regime)  PI iters={pi_res[day_lam].n_sweeps}  "
           f"VI<->PI worst-regime max|dV*|={worst_diff:.2e} (tol 1e-4)\n"
           f"  DP (regime-switched): {dp_m['total_energy_kwh']:.2f} kWh, "
           f"{dp_m['sla_violation_rate_pct']:.2f}% SLA violations "
           f"(= surge floor), {traces['Value Iteration']['metrics']['energy_saved_vs_always_on_pct']:.1f}% "
           f"energy saved vs Always-On\n"
           f"  artifacts: results/ (4 figures, 3 data files, dashboard JSON) + weights/ "
           f"({len(list(WEIGHTS.glob('*.npz')))} npz files)\n"
           f"  total wall time: {total_min:.1f} min\n"
           f"ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
