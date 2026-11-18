"""
evaluation/metrics_and_plots.py
===============================

24-hour diurnal evaluation harness + publication-quality figures + exports.

Figures (saved to results/)
    1. convergence_curves.png   - Bellman residual sweeps (VI sync / Gauss-Seidel),
                                  policy-change history (PI), RTDP backup deltas
    2. pareto_frontier.png      - Total energy (kWh) vs SLA violation rate (%),
                                  DP penalty sweep + heuristic baselines
    3. server_state_heatmap.png - cluster utilization states over the 24h trace,
                                  thermal strip and queue/arrival timeline
    4. decision_boundary.png    - policy decision maps on three state slices

Exports: summary_metrics.csv, pareto_points.csv, web_dashboard_data.json
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from env.datacenter_mdp import U_LABELS, TAU_LABELS, DataCenterMDP  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "font.size": 9.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    }
)

HOURLY_LAMBDA = np.array(
    [0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.9, 1.5, 2.2, 3.0, 3.8, 4.6,
     5.4, 6.2, 6.8, 7.0, 6.6, 5.8, 4.6, 3.4, 2.4, 1.5, 0.8, 0.4]
)


# --------------------------------------------------------------------------------------
# Diurnal workload trace
# --------------------------------------------------------------------------------------
def diurnal_lambda(n_steps: int = 288, step_minutes: int = 5):
    """Low night traffic, morning ramp-up, afternoon peak, evening decline."""
    hours = (np.arange(n_steps) + 0.5) * step_minutes / 60.0
    lam = np.interp(hours, np.arange(24) + 0.5, HOURLY_LAMBDA)
    return lam, hours


def pre_sample_arrivals(lam_t: np.ndarray, seed: int = 1234) -> np.ndarray:
    """Common random numbers: every policy sees the identical arrival stream."""
    rng = np.random.default_rng(seed)
    return rng.poisson(lam_t)


# --------------------------------------------------------------------------------------
# Policy wrappers (uniform act(u, q, tau, t) interface)
# --------------------------------------------------------------------------------------
class GreedyPolicy:
    """Greedy w.r.t. a value function, restricted to mask-valid actions."""

    def __init__(self, env: DataCenterMDP, V: np.ndarray, name: str) -> None:
        self.env, self.V, self.name = env, V, name

    def act(self, u, q, tau, t) -> int:
        s = self.env.state_to_idx(u, q, tau)
        valid = self.env.valid_actions(s)
        row = self.env.R_bar[s, valid] + self.env.gamma * (
            self.env.PROB[s][valid] * self.V[self.env.NEXT[s][valid]]
        ).sum(axis=-1)
        return int(valid[int(np.argmax(row))])


class FixedPolicy:
    """Execute a stored policy array (e.g. the PI output)."""

    def __init__(self, env: DataCenterMDP, pi: np.ndarray, name: str) -> None:
        self.env, self.pi, self.name = env, pi, name

    def act(self, u, q, tau, t) -> int:
        return int(self.pi[self.env.state_to_idx(u, q, tau)])


class RegimePolicy:
    """Piecewise-stationary deployment: switch between per-regime value
    functions according to the forecast arrival rate lambda_t.

    The diurnal trace is non-stationary while the tabular MDP is stationary;
    solving one MDP per load regime (night / day / peak) and switching on the
    (forecastable) arrival rate is the standard production pattern.
    """

    def __init__(self, entries, cuts, lam_t, name="DP (regime-switched)") -> None:
        # entries: list of (env, V) ascending in lambda; cuts: len-2 thresholds
        self.entries, self.cuts, self.lam_t, self.name = entries, cuts, lam_t, name
        self._inner = [GreedyPolicy(e, V, name) for e, V in entries]

    def act(self, u, q, tau, t) -> int:
        lam = self.lam_t[t]
        k = 0 if lam < self.cuts[0] else (1 if lam < self.cuts[1] else 2)
        return self._inner[k].act(u, q, tau, t)


class LamForecastPolicy:
    """Adapter for approximate policies whose one-step lookahead uses the
    (forecastable) arrival rate lambda_t, e.g. the linear-VFA ADP controller."""

    def __init__(self, adp, lam_t: np.ndarray, env: DataCenterMDP,
                 name: str = "Linear-ADP") -> None:
        self.adp, self.lam_t, self.env, self.name = adp, lam_t, env, name

    def act(self, u, q, tau, t) -> int:
        return self.adp.act(u, q, tau, t, lam=float(self.lam_t[t]))


# --------------------------------------------------------------------------------------
# Trace runner
# --------------------------------------------------------------------------------------
def run_trace(
    env: DataCenterMDP,
    policy,
    lam_t: np.ndarray,
    arrivals: np.ndarray,
    start_idx: Optional[int] = None,
) -> Dict:
    """Roll out one policy over the diurnal trace with common random numbers."""
    T = len(lam_t)
    s = env.start_idx if start_idx is None else start_idx
    rec = {
        "hour": np.arange(T) * (5 / 60.0),
        "lam": lam_t.astype(float),
        "u": np.zeros((T, env.n_clusters), dtype=int),
        "q": np.zeros(T, dtype=int),
        "tau": np.zeros(T, dtype=int),
        "served": np.zeros(T, dtype=int),
        "dropped": np.zeros(T, dtype=int),
        "arrived": np.zeros(T, dtype=int),
        "e_it_kwh": np.zeros(T),
        "e_total_kwh": np.zeros(T),
        "switch_kwh": np.zeros(T),
        "switch": np.zeros(T, dtype=int),
        "reward": np.zeros(T),
        "a_alloc": np.zeros(T, dtype=int),
        "a_power": np.zeros(T, dtype=int),
        "a_cool": np.zeros(T, dtype=int),
    }
    for t in range(T):
        u, q, tau = env.idx_to_state(s)
        rec["u"][t], rec["q"][t], rec["tau"][t] = u, q, tau
        a = int(policy.act(u, q, tau, t))
        a_alloc, a_power, a_cool = env.idx_to_action(a)
        rec["a_alloc"][t], rec["a_power"][t], rec["a_cool"][t] = a_alloc, a_power, a_cool
        s, r, info = env.simulate_step(s, a, lam=float(lam_t[t]),
                                       forced_arrival=int(arrivals[t]))
        rec["served"][t] = info["served"]
        rec["dropped"][t] = info["dropped"]
        rec["arrived"][t] = info["arrived"]
        rec["e_it_kwh"][t] = info["e_it_kwh"]
        rec["e_total_kwh"][t] = info["e_total_kwh"]
        rec["switch_kwh"][t] = info["switch_kwh"]
        rec["switch"][t] = int(info["switched"])
        rec["tau"][t] = info["tau_next"]
        rec["reward"][t] = r

    it_with_switch = rec["e_it_kwh"] + rec["switch_kwh"]
    arrived = int(rec["arrived"].sum())
    dropped = int(rec["dropped"].sum())
    metrics = {
        "total_energy_kwh": float(rec["e_total_kwh"].sum()),
        "total_it_kwh_incl_switch": float(it_with_switch.sum()),
        "pue_avg": float(rec["e_total_kwh"].sum() / max(it_with_switch.sum(), 1e-9)),
        "arrived_tasks": arrived,
        "served_tasks": int(rec["served"].sum()),
        "dropped_tasks": dropped,
        "sla_violation_rate_pct": 100.0 * dropped / max(arrived, 1),
        "drop_events": int((rec["dropped"] > 0).sum()),
        "hot_steps": int((rec["tau"] == 2).sum()),
        "hot_frac_pct": 100.0 * float((rec["tau"] == 2).mean()),
        "warm_frac_pct": 100.0 * float((rec["tau"] == 1).mean()),
        "cool_frac_pct": 100.0 * float((rec["tau"] == 0).mean()),
        "switch_count": int(rec["switch"].sum()),
        "avg_queue": float(rec["q"].mean()),
        "max_queue": int(rec["q"].max()),
        "total_reward": float(rec["reward"].sum()),
    }
    return {"series": rec, "metrics": metrics}


# --------------------------------------------------------------------------------------
# Figure 1: convergence curves
# --------------------------------------------------------------------------------------
def plot_convergence(vi_sync, vi_gs, pi_res, rtdp_res, path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.semilogy(vi_sync.residuals, lw=1.6, color="#0d9488", label="VI synchronous")
    ax.semilogy(vi_gs.residuals, lw=1.4, ls="--", color="#f59e0b", label="VI Gauss-Seidel")
    ax.axhline(vi_sync.extra["theta"], color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Bellman sweep $k$")
    ax.set_ylabel(r"residual  $\|V_{k+1}-V_k\|_\infty$")
    ax.set_title("(a) Value Iteration convergence " + r"($\theta$ threshold)")
    ax.legend(loc="upper right")

    ax = axes[1]
    changes = pi_res.extra["policy_changes"]
    ax.bar(range(len(changes)), changes, color="#0d9488", alpha=0.85, width=0.6)
    ax.set_xlabel("Policy Iteration step")
    ax.set_ylabel(r"policy changes  $|\pi_{k+1}\neq\pi_k|$")
    ax.set_title("(b) Policy Iteration improvements\n(0 changes = converged)")
    ax.set_ylim(0, max(max(changes), 1) * 1.15 + 1)

    ax = axes[2]
    ax.semilogy(rtdp_res.residuals, lw=1.4, color="#7c3aed")
    ax.set_xlabel("RTDP trial")
    ax.set_ylabel("max Bellman backup delta")
    ax.set_title(f"(c) RTDP ({rtdp_res.extra['visited_states']} states visited)")
    ax.set_xlim(0, len(rtdp_res.residuals))

    fig.suptitle("Bellman residual convergence - VI (sync/async), PI, RTDP",
                 fontsize=11, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Figure 2: energy vs SLA Pareto frontier
# --------------------------------------------------------------------------------------
def plot_pareto(points: List[Dict], guardrail_drop: float, path: str,
                floor_pct: float = 0.0, floor_label: str = "") -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    if floor_pct > 0:
        ax.axhline(floor_pct, color="#0d9488", lw=1.0, ls=":", zorder=1)
    dp = sorted([p for p in points if p["kind"] == "dp-sweep"], key=lambda p: p["penalty_drop"])
    bad = [p for p in dp if p["penalty_drop"] < guardrail_drop]
    good = [p for p in dp if p["penalty_drop"] >= guardrail_drop]
    for gi, (grp, color, label) in enumerate(
            ((good, "#0d9488", "DP (w3$\\cdot$P$_{drop}$ $\\geq$ guardrail)"),
             (bad, "#dc2626", "DP (guardrail violated, Issue 2)"))):
        if not grp:
            continue
        ax.plot([p["energy_kwh"] for p in grp], [p["sla_violation_rate_pct"] for p in grp],
                "-o", color=color, ms=6, lw=1.4, label=label, zorder=3)
        seen = {}  # fan out labels that share the same (x, y) location
        for p in grp:
            key = (round(p["energy_kwh"], 2), round(p["sla_violation_rate_pct"], 3))
            idx = seen.get(key, 0)
            seen[key] = idx + 1
            ax.annotate(f"P={p['penalty_drop']:.0f}",
                        (p["energy_kwh"], p["sla_violation_rate_pct"]),
                        textcoords="offset points", xytext=(7 + 26 * idx, 8 if idx % 2 == 0 else -13),
                        fontsize=7.5, color=color)
    for p in [q for q in points if q["kind"] == "dp-main"]:
        ax.scatter([p["energy_kwh"]], [p["sla_violation_rate_pct"]], marker="*", s=260,
                   color="#0f766e", edgecolor="k", linewidth=0.5, zorder=4, label=p["label"])
    for p in [q for q in points if q["kind"] == "baseline"]:
        ax.scatter([p["energy_kwh"]], [p["sla_violation_rate_pct"]], marker=p.get("marker", "s"),
                   s=90, color=p.get("color", "#64748b"), edgecolor="k", linewidth=0.5,
                   zorder=4, label=p["label"])
    if floor_pct > 0:
        ax.annotate(floor_label or f"arrival-surge floor ({floor_pct:.2f}%)",
                    (0.99, floor_pct), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, 5), ha="right",
                    fontsize=7.5, color="#0d9488")
        ax.set_ylim(top=max(floor_pct + 0.55, max(p["sla_violation_rate_pct"] for p in points) + 0.12))
    ax.set_xlabel("Total energy over the 24h trace  [kWh]")
    ax.set_ylabel("SLA violation rate  [% of arrived tasks dropped]")
    ax.set_title("Energy vs. SLA Pareto frontier\n(DP penalty sweep + heuristic baselines)")
    ax.legend(loc="center right", fontsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Figure 3: 24h server-state heatmap
# --------------------------------------------------------------------------------------
def plot_server_heatmap(trace: Dict, env: DataCenterMDP, path: str,
                        title_suffix: str = "") -> None:
    ser = trace["series"]
    T = len(ser["hour"])
    hours = ser["hour"]
    U = ser["u"].T  # [N, T]

    fig = plt.figure(figsize=(12.5, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 0.55, 1.5])
    ax0 = fig.add_subplot(gs[0])
    cmap_u = ListedColormap(["#1e293b", "#fbbf24", "#f97316", "#dc2626"])
    im0 = ax0.imshow(U, aspect="auto", interpolation="nearest", cmap=cmap_u,
                     vmin=-0.5, vmax=3.5, extent=[hours[0], hours[-1], env.n_clusters - 0.5, -0.5])
    ax0.set_yticks(range(env.n_clusters))
    ax0.set_yticklabels([f"Cluster {i}\n({env.peak_kw[i]:.1f} kW)" for i in range(env.n_clusters)])
    ax0.set_ylabel("Server cluster\nstate")
    ax0.set_title("Server utilization states under the optimal (VI) policy - 24h diurnal trace"
                  + title_suffix)
    ax0.grid(False)
    cbar = fig.colorbar(im0, ax=ax0, ticks=[0, 1, 2, 3], fraction=0.025, pad=0.02)
    cbar.ax.set_yticklabels(U_LABELS)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    cmap_t = ListedColormap(["#bae6fd", "#fb923c", "#b91c1c"])
    im1 = ax1.imshow(ser["tau"][None, :], aspect="auto", interpolation="nearest",
                     cmap=cmap_t, vmin=-0.5, vmax=2.5,
                     extent=[hours[0], hours[-1], 0.5, -0.5])
    ax1.set_yticks([])
    ax1.set_ylabel("thermal")
    ax1.grid(False)
    cbar1 = fig.colorbar(im1, ax=ax1, ticks=[0, 1, 2], fraction=0.025, pad=0.02)
    cbar1.ax.set_yticklabels(TAU_LABELS)

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax2.step(hours, ser["q"], where="mid", color="#0f766e", lw=1.3, label="queue length $q_t$")
    ax2.step(hours, ser["dropped"], where="mid", color="#dc2626", lw=0.9, alpha=0.75,
             label="drops / step")
    ax2.set_xlabel("time of day  [h]")
    ax2.set_ylabel("jobs")
    ax2.legend(loc="upper left")
    sec = ax2.twinx()
    sec.plot(hours, ser["lam"], color="#64748b", lw=1.2, ls="--", label=r"arrival rate $\lambda_t$")
    sec.set_ylabel(r"$\lambda_t$  [jobs/step]", color="#64748b")
    sec.grid(False)
    sec.spines["right"].set_visible(True)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = sec.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Figure 4: policy decision boundaries
# --------------------------------------------------------------------------------------
def _action_class(env: DataCenterMDP, a: int) -> int:
    a_alloc, a_power, _ = env.idx_to_action(a)
    N = env.n_clusters
    if 1 <= a_power <= N:
        return a_power              # 1..N  wake c_i
    if a_power > N:
        return N + (a_power - N)    # N+1..2N sleep c_i
    if a_alloc < N:
        return 2 * N + 1 + a_alloc  # route c_i
    return 0                        # hold


def plot_decision_boundary(env: DataCenterMDP, V: np.ndarray, path: str) -> Dict:
    N = env.n_clusters
    Q = env.R_bar + env.gamma * (env.PROB * V[env.NEXT]).sum(axis=-1)
    pi = np.where(env.mask, Q, -np.inf).argmax(axis=1)

    class_names = (["hold"] + [f"wake c{i}" for i in range(N)] + [f"sleep c{i}" for i in range(N)]
                   + [f"route c{i}" for i in range(N)])
    n_cls = len(class_names)
    # hold=gray, wake=amber shades (dark = bigger tier), sleep=purple shades, route=teal shades
    palette = (["#94a3b8"]
               + ["#fbbf24", "#d97706", "#92400e"][:N]
               + ["#c084fc", "#7e22ce", "#3b0764"][:N]
               + ["#5eead4", "#14b8a6", "#0f766e"][:N])
    cmap = ListedColormap(palette)

    slices = [
        ((0, 0, 0), "(i) cold pool: u = (0,0,0)"),
        ((3, 1, 0), "(ii) primary saturated: u = (3,1,0)"),
        ((3, 3, 0), "(iii) two tiers saturated: u = (3,3,0)"),
    ]
    grids = compute_decision_grids(env, V)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), constrained_layout=True)
    for ax, sl in zip(axes, grids["slices"]):
        grid = np.array(sl["grid"])
        cool = np.array(sl["cooling"])
        ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=n_cls - 0.5, origin="lower",
                  interpolation="nearest")
        for r in range(3):
            for c in range(env.q_max + 1):
                ax.text(c, r, ("cool:Hi" if cool[r, c] else "cool:Lo"),
                        ha="center", va="center", fontsize=6.5, color="#111827")
        ax.set_xticks(range(env.q_max + 1))
        ax.set_xticklabels([str(q) for q in range(env.q_max + 1)])
        ax.set_yticks(range(3))
        ax.set_yticklabels(TAU_LABELS)
        ax.set_xlabel("queue length $q$")
        ax.set_title(sl["title"])
        ax.grid(False)
    handles = [Patch(facecolor=palette[i], label=class_names[i]) for i in range(n_cls)]
    fig.legend(handles=handles, loc="lower center", ncol=n_cls, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Policy decision boundaries (VI-optimal actions on state slices)",
                 fontsize=11, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return grids


def compute_decision_grids(env: DataCenterMDP, V: np.ndarray) -> Dict:
    """Serialize the decision-map slices for the web dashboard."""
    N = env.n_clusters
    Q = env.R_bar + env.gamma * (env.PROB * V[env.NEXT]).sum(axis=-1)
    pi = np.where(env.mask, Q, -np.inf).argmax(axis=1)
    class_names = (["hold"] + [f"wake c{i}" for i in range(N)]
                   + [f"sleep c{i}" for i in range(N)] + [f"route c{i}" for i in range(N)])
    slices = [
        ((0, 0, 0), "(i) cold pool: u = (0,0,0)"),
        ((3, 1, 0), "(ii) primary saturated: u = (3,1,0)"),
        ((3, 3, 0), "(iii) two tiers saturated: u = (3,3,0)"),
    ]
    out = []
    for u_slice, title in slices:
        grid, cool = [], []
        for tau in range(3):
            row, crow = [], []
            for q in range(env.q_max + 1):
                s = env.state_to_idx(u_slice, q, tau)
                a = int(pi[s])
                row.append(_action_class(env, a))
                crow.append(env.idx_to_action(a)[2])
            grid.append(row)
            cool.append(crow)
        out.append({"u": list(u_slice), "title": title, "grid": grid, "cooling": cool})
    return {"class_names": class_names, "slices": out}


# --------------------------------------------------------------------------------------
# Console + CSV + dashboard JSON exports
# --------------------------------------------------------------------------------------
def metrics_frame(rows: List[Dict]) -> pd.DataFrame:
    cols = ["policy", "total_energy_kwh", "sla_violation_rate_pct", "dropped_tasks",
            "pue_avg", "switch_count", "hot_frac_pct", "warm_frac_pct", "avg_queue",
            "drop_events", "total_reward"]
    return pd.DataFrame(rows)[cols]


def print_summary_table(df: pd.DataFrame) -> None:
    with pd.option_context("display.width", 200, "display.float_format",
                           "{:,.3f}".format):
        print(df.to_string(index=False))


def downsample_trace(trace: Dict, bins: int = 96) -> Dict:
    """288 five-minute steps -> 96 fifteen-minute bins for the web dashboard."""
    ser = trace["series"]
    T = len(ser["hour"])
    factor = T // bins
    N = ser["u"].shape[1]

    def mode_bin(a):
        return np.array([np.bincount(a[i * factor:(i + 1) * factor]).argmax()
                         for i in range(bins)])

    out = {
        "hours": np.round(np.arange(bins) * factor * (5 / 60.0) + 2.5 / 60, 4).tolist(),
        "lam": ser["lam"].reshape(bins, factor).mean(axis=1).round(3).tolist(),
        "u": [mode_bin(ser["u"][:, i]).tolist() for i in range(N)],
        "q": ser["q"].reshape(bins, factor).mean(axis=1).round(2).tolist(),
        "tau": mode_bin(ser["tau"]).tolist(),
        "energy_kwh": ser["e_total_kwh"].reshape(bins, factor).sum(axis=1).round(5).tolist(),
        "dropped": ser["dropped"].reshape(bins, factor).sum(axis=1).tolist(),
        "arrived": ser["arrived"].reshape(bins, factor).sum(axis=1).tolist(),
    }
    return out


def export_dashboard_json(path: str, payload: Dict) -> None:
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        return o

    with open(path, "w") as fh:
        json.dump(clean(payload), fh, indent=1)
