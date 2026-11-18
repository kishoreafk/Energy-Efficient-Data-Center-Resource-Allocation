# Energy-Efficient Data Center Resource Allocation — Dynamic Programming with Modern ADP/RL Extensions

A complete, modular Python project implementing an **exact tabular MDP** for
data-center power/cooling/workload management, solved with **Value Iteration**
(synchronous + Gauss–Seidel) and **Policy Iteration** (exact linear-system
evaluation), extended with **RTDP** and **linear value-function approximation**
for scalability beyond the tabular regime, and benchmarked against heuristic
baselines on a 24-hour diurnal trace.

| | |
|---|---|
| State space | `s = (u_1, u_2, u_3, q, τ)`, \|S\| = **1344** (4³ utilization × 7 queue × 3 thermal) |
| Action space | `a = (a_alloc, a_power, a_cool)`, \|A\| = **56** |
| Transitions | Explicit stochastic tensor `P[s,a,s']` — **507,360 scipy.sparse nonzeros** (dense would need 101.2M cells), every row sums to 1.0 ± 0.0 |
| Reward | `R = −(w₁·E_IT·PUE(τ,a_cool) + w₂·E_switch + w₃·(dropped·P_drop + I(τ'=Hot)·P_hot))`, maximized |
| Discount | γ = 0.95 (inside the required [0.90, 0.99] band) |
| Runtime | full pipeline ≈ **2 minutes** on CPU (no GPU required) |

---

## 1. Quick start

```bash
pip install -r requirements.txt      # numpy scipy matplotlib pandas gymnasium pytest
python tests/test_mdp_sanity.py      # 12 guardrail checks (also runs inside main)
python main.py                       # full pipeline: solve -> evaluate -> figures -> weights
# optional: python main.py --skip-adp   (skip the N=8 scalability demo)
```

Outputs land in `results/` (figures, CSV tables, `web_dashboard_data.json`) and
`weights/` (model weights, see §6). A **web dashboard** lives in
`web-dashboard/` (Next.js) — see §8.

---

## 2. MDP formulation

### State  `s = (u_1, u_2, u_3, q, τ)`
* `u_i ∈ {0=Sleep, 1=Idle, 2=Medium, 3=High}` — discretized CPU/power state of
  the three **tiered** clusters (peak IT power 0.8 / 1.0 / 1.3 kW).
* `q ∈ {0..6}` — shared job-queue length (Poisson arrivals, tail folded at `Q_MAX`).
* `τ ∈ {0=Cool, 1=Warm, 2=Hot}` — ambient thermal state driving cooling PUE.

### Action  `a = (a_alloc, a_power, a_cool)`
* `a_alloc ∈ {0,1,2,hold}` — routing priority: the target cluster is served
  first (`min(q, cap_i(u_target))`), leftover backlog **spills to the other
  active clusters** in tier order (efficient-first). Fleet throughput is the
  sum of the engaged clusters' service caps.
* `a_power ∈ {noop, wake_i, sleep_i}` — waking pays a boot spike
  `E_switch = 5 × idle-step energy`; only an **Idle** server may sleep.
* `a_cool ∈ {low, high}` — HVAC intensity, entering `PUE(τ, a_cool)`:

| PUE | Cool | Warm | Hot |
|-----|------|------|-----|
| low cooling  | 1.30 | 1.50 | 1.80 |
| high cooling | 1.45 | 1.55 | 1.65 |

### Dynamics per 5-minute step
1. power switch applied → `u_post`;
2. priority + spill routing → `served_i`, `q_mid = q − Σ served_i`;
3. utilization: serving clusters rise `u' = min(3, max(u,1) + ⌈served_i/2⌉)`,
   non-serving active clusters decay one level toward Idle;
4. thermal (deterministic): `pressure = n_busy + (0 if cool=high else 1) − 2`,
   `τ' = clip(τ + sign(pressure), 0, 2)`;
5. arrivals `n ~ Poisson(λ)` discretized with the tail folded at `Q_MAX`
   (boundary mass accumulated at `q' = Q_MAX`), drops = `max(0, q_mid + n − Q_MAX)`.

### Capacity tiers (why scale-out matters)
`cap_i(u)`: c0 = (1, 2, 3), c1 = (1, 3, 4), c2 = (2, 4, 5) for u = Idle/Med/High.
Every single cap is **strictly below the 6-slot buffer** and λ peaks at
7 jobs/step, so no single cluster can drain the backlog alone — waking and
spilling genuinely trade energy against SLA.

---

## 3. Engineering guardrails (debugging playbook)

| # | Failure mode | Guardrail implemented | Verified by |
|---|--------------|----------------------|-------------|
| 1 | State-space explosion | \|S\| = 1344 ∈ [300, 1500]; `P` stored as **scipy.sparse CSR per action** (507k nnz vs 101M dense cells) | `test_state_space_size_in_guardrail` |
| 2 | Lazy-shutdown reward hacking | `w₃·P_drop = 50 > 5 × max(E_IT+E_switch) = 39.5` per step (ratio 1.27) + **action mask** forces a wake when `q>0` and all clusters sleep | `test_reward_guardrail_anti_lazy_shutdown`, `test_action_mask_safety` |
| 3 | Server thrashing | `E_switch = 5.0× idle-step energy` (inside the 3–5 band) → Bellman lookahead learns hysteresis; traces show ~11 wake events/day, no oscillation | `test_switch_cost_hysteresis` |
| 4 | PI infinite loop | strict ε-improvement: update only if `Q(s,a_new) > Q(s,π_old) + 1e-8`, plus policy-hash oscillation detection | PI converges in 5–6 iterations, `oscillation=False` |
| 5 | Invalid probabilities / divergence | `assert_valid_mdp()` runs before any solver: non-negative probs, every (s,a) row = 1.0 ± 1e-6 (tail folded at boundary), γ ∈ [0.90,0.99], finite rewards, no dead-end mask | 12/12 sanity checks pass |

The Pareto sweep additionally **demonstrates** Issue 2: with
`P_drop = 10` (guardrail violated) the DP indeed drifts toward the
low-energy/high-violation corner (§5).

---

## 4. Algorithms

* **Value Iteration** — synchronous (Jacobi) and asynchronous (Gauss–Seidel,
  shuffled in-place sweeps) Bellman optimality backups, θ = 1e-5 default.
  Convergence (measured): sync 225/239/243 sweeps, GS 135/140/161 sweeps for
  λ = 1.2/3.5/6.5 (θ threshold reached in < 2 s each).
* **Policy Iteration** — exact policy evaluation via `(I − γP_π)V = r_π` solved
  with `np.linalg.solve` (evaluation residual ≈ 1e-14) or iterative backups;
  improvement with strict ε-tie-breaking. Converges in **5/5/6 iterations**.
* **RTDP** — sampling-based backups on visited states only (1201/1344 visited
  in 4000 trials); policy agreement with V* on frequently-visited states
  **95.2%**.
* **Linear-VFA ADP** — semi-gradient TD(0) over 10 hand-engineered features
  (`bias, active_frac, mean_power_frac, busy_frac, q_norm, tau_norm,
  q_x_idle_cap, tau_x_busy, util_sum_norm, q_sq`) with greedy one-step
  lookahead `Q(s,a) ≈ r̄(s,a) + γθ·φ(𝔼[s'])`. For N = 8 the exact table would
  need 1,376,256 states and ~2.95e9 transition entries — infeasible; the
  simulator-based ADP trains in ~45 s and still consolidates (67% energy
  savings vs Always-On at N=8, §5).

**Regime deployment.** The tabular MDP is stationary, the diurnal trace is not.
Following the standard production pattern, one policy is solved per load regime
(λ = 1.2 / 3.5 / 6.5) and switched on the forecast arrival rate (cuts at 2.5 /
5.5 jobs/step). VI ↔ PI agreement is verified **inside every regime**.

---

## 5. Results (24-h diurnal trace, 812 offered tasks, common random numbers)

The arrival trace contains an **irreducible surge floor**: 91 tasks
(11.21%) arrive in single steps that exceed the 6-slot buffer even at maximum
throughput — no policy can do better. The comparison is therefore about
*reaching the floor at minimum energy* (and avoiding thermal throttling):

| Policy | Energy (kWh) | SLA violations | Switches | Hot steps | PUE |
|---|---|---|---|---|---|
| **Value Iteration (regime)** | **44.27** | 11.21% (= floor) | 11 | 0% | 1.371 |
| **Policy Iteration (regime)** | **44.27** | 11.21% (= floor) | 11 | 0% | 1.371 |
| RTDP (λ=3.0 design point) | 51.08 | 11.21% | 14 | 0% | 1.356 |
| Always-On (Round-Robin) | 75.25 | 11.21% | 0 | 0% | 1.367 |
| Greedy Consolidation | 84.07 | 11.21% | 23 | **51.0%** | 1.540 |

* **DP saves 41.2% energy vs Always-On at identical SLA**, and never enters the
  Hot state (it pre-emptively buys high cooling when the fleet is busy, while
  the greedy rule-free baseline spends 51% of the day thermally throttled).
* VI and PI traces are **bit-identical**, as required by their 1.86e-05 value
  agreement and 100% policy match.
* **Pareto sweep** (`penalty_drop ∈ {10, 25, 50, 100, 200}`): P=10 (guardrail
  violated) drifts to 41.07 kWh @ 11.58% — the lazy-shutdown corner predicted
  by Issue 2; P ≥ 25 all reach the floor at 44.3 kWh.
* **N = 8 scalability** (linear-VFA ADP): 60.4 kWh vs Always-On 182.8 kWh
  (−67%) and Greedy 162.0 kWh at the same surge floor.

Figures in `results/`:
`convergence_curves.png` (VI sync/GS residuals, PI policy changes, RTDP deltas),
`pareto_frontier.png` (energy vs SLA + surge floor + guardrail-violating points),
`server_state_heatmap.png` (24-h cluster states: night consolidation → morning
wake → peak scale-out → evening sleep), `decision_boundary.png` (policy maps on
three state slices at the peak regime).

---

## 5.5 Production verification on Kaggle

The shipped `results/` and `weights/` are from the verified Kaggle CPU
production run (kernel `kishore00afk/dc-dp-rl-run`, Python 3.13.18); the
tabular sparse-DP workload is CPU-bound, so GPU is unnecessary. Verification
protocol (logs first, artifacts after): kernel status → log markers
(`12/12 checks passed`, `ALL CHECKS PASSED`, `MAIN_EXIT_CODE=0`) → artifact
bundle checks (figures non-blank, CSVs identical, all weight files load).
**Every scientific number is bit-identical to the local run** (44.27 kWh,
11.21% SLA floor, VI↔PI 1.86e-05, identical Pareto sweep) — the pipeline is
fully deterministic given the seed. Reproduction kit in `kaggle/`
(credentials not included). See `kaggle/REPRODUCE_ON_KAGGLE.md`.

---

## 6. Model weights (`weights/*.npz`)

| File | Contents |
|---|---|
| `vi_policy.npz` | `V` (3 × 1344, one value function per regime), `pi` (3 × 1344), `regime_lambdas`, `regime_cuts`, `residuals`, `gamma`, `theta`, `n_sweeps` |
| `vi_gauss_seidel.npz` | same layout for the asynchronous solver |
| `pi_policy.npz` | per-regime `V`, `pi` from Policy Iteration + improvement history |
| `rtdp_value_function.npz` | `V`, `pi`, `visited_mask`, `visit_counts`, `n_trials` |
| `adp_linear_theta.npz` | `theta` (10-dim), `feature_names`, `n_clusters=8`, `td_curve` |

```python
import numpy as np, env.datacenter_mdp as m
w = np.load("weights/vi_policy.npz")
env = m.DataCenterMDP(lambda_arrivals=float(w["regime_lambdas"][1]))
V, pi = w["V"][1], w["pi"][1]           # day-regime optimal policy
s = env.state_to_idx((1, 1, 1), 0, 0)   # all Idle, empty queue, Cool
print("V*(start) =", V[s], " action =", env.idx_to_action(int(pi[s])))
```

---

## 7. Repository layout

```
dc-dp-rl/
├── main.py                       # end-to-end pipeline (8 stages)
├── requirements.txt
├── env/datacenter_mdp.py         # MDP: encoding, P/R tensors, mask, assert_valid_mdp
├── solvers/dynamic_programming.py# VI (sync + Gauss-Seidel), PI (linear + iterative)
├── solvers/modern_extensions.py  # RTDP, Linear-VFA ADP (N > 5)
├── baselines/heuristics.py       # Round-Robin Always-On, Greedy Threshold Consolidation
├── evaluation/metrics_and_plots.py # diurnal harness, 4 figures, CSV/JSON exports
├── tests/test_mdp_sanity.py      # 12 guardrail checks (pytest or standalone)
├── weights/                      # trained policies / value functions (npz)
├── results/                      # figures, tables, dashboard data (Kaggle-verified)
├── weights/                      # trained policies / value functions (npz)
├── kaggle/                       # cloud reproduction kit (kernel script + protocol)
└── web-dashboard/                # standalone Next.js visualization dashboard (see §8)
```

## 8. Web dashboard

```bash
cd web-dashboard
npm install
npm run dev        # http://localhost:3000
```

A shadcn/ui + Tailwind dashboard rendering `results/web_dashboard_data.json`:
MDP specification and guardrail checklist, policy comparison table & energy
bars, the energy/SLA Pareto frontier, Bellman convergence curves, the 24-h
server-state heatmap and decision-boundary maps, and the N=8 ADP scalability
panel.

## 9. Design notes & limitations

* Arrivals are discretized with the tail folded at `Q_MAX` inside the solve
  model, while the evaluation trace samples true Poisson counts (model mismatch
  is intentional and documented — the deployed policy reacts to the realized
  queue state).
* When clamping merges two arrival counts into the same `s'`, `R(s,a,s')`
  stores the probability-weighted mean reward (exact for expected-reward DP).
* The thermal chain is deterministic (per the curriculum spec); stochasticity
  enters through the Poisson arrivals.
* RTDP is trained at a single design point λ = 3.0 (tabular sampling bound);
  the regime-switched VI/PI deployment and the feature-based ADP cover the
  non-stationarity instead.
# Energy-Efficient-Data-Center-Resource-Allocation
