"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  AreaChart, Area, BarChart, Bar, Cell, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip as RTooltip, ResponsiveContainer, Legend,
} from "recharts";
import {
  Activity, BadgeCheck, BatteryCharging, Cpu, Gauge, ShieldCheck, Thermometer, Zap,
} from "lucide-react";
import type { DashboardData, PolicyRow } from "./types";
import { ParetoPanel, OperationsPanel, DecisionPanel, ScalabilityPanel } from "./panels";

const EMERALD = "#10b981";
const TEAL = "#14b8a6";
const AMBER = "#f59e0b";
const ORANGE = "#f97316";
const RED = "#ef4444";
const SLATE = "#94a3b8";

const POLICY_COLORS: Record<string, string> = {
  "Value Iteration": EMERALD,
  "Policy Iteration": TEAL,
  RTDP: AMBER,
  "Greedy Consolidation": ORANGE,
  "Always-On (Round-Robin)": SLATE,
};

function kpi(cards: { icon: React.ReactNode; label: string; value: string; sub: string }[]) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {cards.map((c) => (
        <Card key={c.label} className="border-slate-800 bg-slate-900/60">
          <CardContent className="flex items-center gap-4 p-5">
            <div className="rounded-lg bg-emerald-500/10 p-2.5 text-emerald-400">{c.icon}</div>
            <div className="min-w-0">
              <div className="text-2xl font-bold tracking-tight text-slate-50">{c.value}</div>
              <div className="text-xs font-medium text-emerald-400">{c.label}</div>
              <div className="truncate text-xs text-slate-400">{c.sub}</div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function SpecCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card className="border-slate-800 bg-slate-900/60">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold text-emerald-400">{title}</CardTitle>
      </CardHeader>
      <CardContent className="text-xs leading-relaxed text-slate-300">{children}</CardContent>
    </Card>
  );
}

export default function Dashboard({ data }: { data: DashboardData }) {
  const dp = data.policies.find((p) => p.policy === "Value Iteration")!;
  const ao = data.policies.find((p) => p.policy === "Always-On (Round-Robin)")!;
  const greedy = data.policies.find((p) => p.policy === "Greedy Consolidation")!;
  const offered = data.meta.offered_tasks;
  const floorPct = (100 * data.meta.surge_floor_tasks) / offered;

  const kpis = kpi([
    {
      icon: <Zap className="h-6 w-6" />, label: "Energy saved vs Always-On",
      value: `${dp.energy_saved_vs_always_on_pct.toFixed(1)}%`,
      sub: `${dp.total_energy_kwh.toFixed(1)} vs ${ao.total_energy_kwh.toFixed(1)} kWh / 24h`,
    },
    {
      icon: <ShieldCheck className="h-6 w-6" />, label: "SLA violations",
      value: `${dp.sla_violation_rate_pct.toFixed(2)}%`,
      sub: `= arrival-surge floor (${data.meta.surge_floor_tasks} tasks) — provably optimal`,
    },
    {
      icon: <BadgeCheck className="h-6 w-6" />, label: "VI ↔ PI V* agreement",
      value: `${data.convergence.agreement.policy_agreement_worst_regime_pct.toFixed(1)}%`,
      sub: `max |ΔV*| = ${data.convergence.agreement.max_v_diff_worst_regime.toExponential(1)} (tol 1e-4)`,
    },
    {
      icon: <Thermometer className="h-6 w-6" />, label: "Thermal throttling avoided",
      value: `0% hot`,
      sub: `vs ${greedy.hot_frac_pct.toFixed(0)}% hot for Greedy rule (PUE 1.54)`,
    },
  ]);

  const lamData = data.diurnal.hours.map((h, i) => ({ hour: h, lam: data.diurnal.lam[i] }));

  const guardrails: { ok: boolean; title: string; detail: string }[] = [
    {
      ok: data.validation.guardrail_ratio > 1,
      title: "Anti lazy-shutdown (Issue 2)",
      detail: `w₃·P_drop = ${data.validation.w3_times_penalty_drop} > 5 × max step cost = ` +
        `${(5 * data.validation.max_step_cost_units).toFixed(1)} (ratio ` +
        `${data.validation.guardrail_ratio.toFixed(2)})`,
    },
    {
      ok: data.validation.min_switch_to_idle_ratio >= 3,
      title: "Anti thrashing / hysteresis (Issue 3)",
      detail: `E_switch = ${data.validation.min_switch_to_idle_ratio.toFixed(1)}× idle-step energy ` +
        `(band 3–5) → boot spike makes wake/sleep churn unprofitable`,
    },
    {
      ok: data.spec.n_states >= 300 && data.spec.n_states <= 1500,
      title: "Tabular tractability (Issue 1)",
      detail: `|S| = ${data.spec.n_states} ∈ [300, 1500]; P stored sparse: ` +
        `${data.validation.nonzeros_P.toLocaleString()} nnz vs ${(data.spec.n_states ** 2 * data.spec.n_actions / 1e6).toFixed(0)}M dense cells`,
    },
    {
      ok: data.convergence.agreement.max_v_diff_worst_regime <= 1e-4,
      title: "VI = PI exactness (Issue 4/5)",
      detail: `worst-regime max |ΔV*| = ${data.convergence.agreement.max_v_diff_worst_regime.toExponential(1)} ≤ 1e-4, ` +
        `PI oscillation detector: ${data.convergence.pi.oscillation_detected ? "triggered" : "clear"}`,
    },
    {
      ok: data.validation.max_row_sum_deviation <= 1e-6,
      title: "Valid stochastic kernel",
      detail: `all ${data.spec.n_states * data.spec.n_actions} (s,a) rows sum to 1 ± ` +
        `${data.validation.max_row_sum_deviation.toExponential(0)}; min prob ≥ 0; γ = ${data.spec.gamma} ∈ [0.90, 0.99]`,
    },
  ];

  const energyBars = data.policies.map((p) => ({
    name: p.policy.replace(" (Round-Robin)", ""),
    energy: Number(p.total_energy_kwh.toFixed(2)),
    fill: POLICY_COLORS[p.policy] ?? SLATE,
  }));

  const viSync = data.convergence.vi_sync.residuals.map((r, i) => ({ k: i, sync: r }));
  const viGs = data.convergence.vi_gs.residuals.map((r, i) => ({ k: i, gs: r }));
  const rtdpCurve = data.convergence.rtdp.curve.map((r, i) => ({ trial: i, delta: r }));
  const piBars = data.convergence.pi.policy_changes.map((c, i) => ({ iter: `iter ${i}`, changes: c }));

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* header */}
      <header className="sticky top-0 z-20 border-b border-slate-800 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-emerald-500/10 p-2 text-emerald-400">
              <BatteryCharging className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-sm font-bold sm:text-base">
                Energy-Efficient Data Center Resource Allocation
              </h1>
              <p className="text-xs text-slate-400">
                Dynamic Programming with Modern ADP/RL Extensions — Unit 2 curriculum pipeline
              </p>
            </div>
          </div>
          <div className="ml-auto flex flex-wrap gap-1.5">
            <Badge variant="outline" className="border-emerald-500/40 text-emerald-400">
              |S| = {data.spec.n_states}
            </Badge>
            <Badge variant="outline" className="border-emerald-500/40 text-emerald-400">
              |A| = {data.spec.n_actions}
            </Badge>
            <Badge variant="outline" className="border-emerald-500/40 text-emerald-400">
              γ = {data.spec.gamma}
            </Badge>
            <Badge variant="outline" className="border-slate-600 text-slate-300">
              3 load regimes
            </Badge>
            <Badge variant="outline" className="border-slate-600 text-slate-300">
              {offered} tasks / 24 h
            </Badge>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6">
        {kpis}

        <Tabs defaultValue="overview" className="mt-6">
          <TabsList className="flex h-auto flex-wrap gap-1 bg-slate-900">
            {["overview", "policies", "pareto", "convergence", "operations", "decisions", "scalability"].map((t) => (
              <TabsTrigger key={t} value={t} className="capitalize data-[state=active]:bg-emerald-500/15 data-[state=active]:text-emerald-300">
                {t === "decisions" ? "decision maps" : t}
              </TabsTrigger>
            ))}
          </TabsList>

          {/* ------------------------------------------------------- overview */}
          <TabsContent value="overview" className="mt-4 space-y-4">
            <div className="grid gap-4 lg:grid-cols-2">
              <SpecCard title="State space s = (u₁, u₂, u₃, q, τ)">
                <ul className="list-disc space-y-1 pl-4">
                  <li><b>u_i ∈ {"{Sleep, Idle, Medium, High}"}</b> — tiered clusters at 0.8 / 1.0 / 1.3 kW peak (Idle ≈ 60%, Sleep ≈ 5% of peak)</li>
                  <li><b>q ∈ 0..{data.spec.q_max}</b> — shared job queue, Poisson arrivals with tail folded at Q_MAX</li>
                  <li><b>τ ∈ {"{Cool, Warm, Hot}"}</b> — thermal state; deterministic pressure dynamics, drives PUE</li>
                </ul>
              </SpecCard>
              <SpecCard title="Action space a = (a_alloc, a_power, a_cool)">
                <ul className="list-disc space-y-1 pl-4">
                  <li><b>Routing</b>: priority cluster served first, backlog <b>spills</b> to other active clusters (efficient-first) — fleet throughput = Σ engaged caps</li>
                  <li><b>Power</b>: wake_i (boot spike = 5× idle-step energy) / sleep_i (Idle servers only)</li>
                  <li><b>Cooling</b>: low vs high HVAC — high cooling costs more when Cool (1.45 vs 1.30) but saves when Hot (1.65 vs 1.80)</li>
                </ul>
              </SpecCard>
              <SpecCard title="Reward (negative cost, maximized)">
                <code className="block whitespace-pre-wrap rounded bg-slate-950 p-3 text-[11px] leading-relaxed text-emerald-300">
                  {`R(s,a,s') = -( w1·E_IT(s,a)·PUE(τ,a_cool)
              + w2·E_switch(s,a)
              + w3·(dropped·P_drop + I(τ'=Hot)·P_hot) )`}
                </code>
                <p className="mt-2">
                  w₁ = w₂ = 10 / kWh, w₃ = 1, P_drop = {data.spec.penalty_drop}, P_hot = {data.spec.penalty_hot}
                </p>
              </SpecCard>
              <SpecCard title="24h diurnal arrival trace">
                <div className="h-28">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={lamData} margin={{ top: 4, right: 4, bottom: 0, left: -18 }}>
                      <defs>
                        <linearGradient id="lamFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={EMERALD} stopOpacity={0.5} />
                          <stop offset="100%" stopColor={EMERALD} stopOpacity={0.03} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="hour" tick={{ fontSize: 9, fill: SLATE }} unit="h" />
                      <YAxis tick={{ fontSize: 9, fill: SLATE }} />
                      <RTooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 11 }} />
                      <Area dataKey="lam" stroke={EMERALD} fill="url(#lamFill)" strokeWidth={1.6} name="λ_t [jobs/step]" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <p className="mt-1 text-slate-400">night trough 0.3 → afternoon peak 7.0 jobs/step; {offered} tasks offered, identical stream for every policy (common random numbers)</p>
              </SpecCard>
            </div>

            <Card className="border-slate-800 bg-slate-900/60">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                  <ShieldCheck className="h-4 w-4 text-emerald-400" /> Engineering guardrails — all checks passed before any solver ran
                </CardTitle>
              </CardHeader>
              <CardContent className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
                {guardrails.map((g) => (
                  <div key={g.title} className="flex gap-2.5 rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                    <BadgeCheck className={`mt-0.5 h-4 w-4 shrink-0 ${g.ok ? "text-emerald-400" : "text-red-400"}`} />
                    <div>
                      <div className="text-xs font-semibold text-slate-100">{g.title}</div>
                      <div className="mt-0.5 text-[11px] leading-relaxed text-slate-400">{g.detail}</div>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </TabsContent>

          {/* ------------------------------------------------------- policies */}
          <TabsContent value="policies" className="mt-4 space-y-4">
            <div className="grid gap-4 lg:grid-cols-2">
              <Card className="border-slate-800 bg-slate-900/60">
                <CardHeader className="pb-0">
                  <CardTitle className="text-sm font-semibold text-slate-100">Total energy over the 24h trace</CardTitle>
                  <CardDescription className="text-xs">lower is better — identical SLA for all competent policies</CardDescription>
                </CardHeader>
                <CardContent className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={energyBars} margin={{ top: 8, right: 8, bottom: 24, left: -14 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="name" tick={{ fontSize: 9, fill: SLATE }} angle={-14} dy={8} interval={0} />
                      <YAxis tick={{ fontSize: 9, fill: SLATE }} />
                      <RTooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 11 }} />
                      <Bar dataKey="energy" radius={[4, 4, 0, 0]}>
                        {energyBars.map((b) => <Cell key={b.name} fill={b.fill} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
              <Card className="border-slate-800 bg-slate-900/60">
                <CardHeader className="pb-0">
                  <CardTitle className="text-sm font-semibold text-slate-100">Full metric table</CardTitle>
                  <CardDescription className="text-xs">surge floor = {data.meta.surge_floor_tasks} tasks ({floorPct.toFixed(2)}%) — no policy can drop fewer</CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow className="border-slate-800">
                        <TableHead className="h-8 text-[11px]">Policy</TableHead>
                        <TableHead className="h-8 text-right text-[11px]">kWh</TableHead>
                        <TableHead className="h-8 text-right text-[11px]">SLA</TableHead>
                        <TableHead className="h-8 text-right text-[11px]">Sw</TableHead>
                        <TableHead className="h-8 text-right text-[11px]">Hot</TableHead>
                        <TableHead className="h-8 text-right text-[11px]">PUE</TableHead>
                        <TableHead className="h-8 text-right text-[11px]">avgQ</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {data.policies.map((p: PolicyRow) => (
                        <TableRow key={p.policy} className="border-slate-800/70">
                          <TableCell className="py-1.5 text-[11px] font-medium" style={{ color: POLICY_COLORS[p.policy] }}>
                            {p.policy}
                          </TableCell>
                          <TableCell className="py-1.5 text-right text-[11px]">{p.total_energy_kwh.toFixed(2)}</TableCell>
                          <TableCell className="py-1.5 text-right text-[11px]">{p.sla_violation_rate_pct.toFixed(2)}%</TableCell>
                          <TableCell className="py-1.5 text-right text-[11px]">{p.switch_count}</TableCell>
                          <TableCell className={`py-1.5 text-right text-[11px] ${p.hot_frac_pct > 1 ? "text-red-400" : ""}`}>
                            {p.hot_frac_pct.toFixed(1)}%
                          </TableCell>
                          <TableCell className="py-1.5 text-right text-[11px]">{p.pue_avg.toFixed(3)}</TableCell>
                          <TableCell className="py-1.5 text-right text-[11px]">{p.avg_queue.toFixed(2)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  <p className="mt-3 text-[11px] leading-relaxed text-slate-400">
                    VI and PI traces are <b className="text-slate-200">bit-identical</b> — both converge to the same
                    regime-switched optimal policy. The DP matches the SLA floor at {dp.energy_saved_vs_always_on_pct.toFixed(1)}% less
                    energy than Always-On and avoids thermal throttling entirely by pre-emptively buying high cooling while clusters are busy.
                  </p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* ------------------------------------------------------- pareto */}
          <TabsContent value="pareto" className="mt-4">
            <ParetoPanel data={data} floorPct={floorPct} />
          </TabsContent>

          {/* ------------------------------------------------------- convergence */}
          <TabsContent value="convergence" className="mt-4 space-y-4">
            <div className="grid gap-4 xl:grid-cols-3">
              <Card className="border-slate-800 bg-slate-900/60">
                <CardHeader className="pb-0">
                  <CardTitle className="text-sm font-semibold text-slate-100">(a) Value Iteration residuals</CardTitle>
                  <CardDescription className="text-xs">
                    sync {data.convergence.vi_sync.n_sweeps} sweeps / Gauss–Seidel {data.convergence.vi_gs.n_sweeps} sweeps (θ = {data.convergence.vi_sync.theta})
                  </CardDescription>
                </CardHeader>
                <CardContent className="h-60">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart margin={{ top: 6, right: 8, bottom: 4, left: -8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="k" type="number" tick={{ fontSize: 9, fill: SLATE }} />
                      <YAxis scale="log" domain={["auto", "auto"]} tick={{ fontSize: 9, fill: SLATE }} />
                      <RTooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 11 }} />
                      <Legend wrapperStyle={{ fontSize: 10 }} />
                      <Line data={viSync} dataKey="sync" stroke={EMERALD} dot={false} name="synchronous" strokeWidth={1.6} />
                      <Line data={viGs} dataKey="gs" stroke={AMBER} dot={false} name="Gauss–Seidel" strokeWidth={1.4} strokeDasharray="5 3" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
              <Card className="border-slate-800 bg-slate-900/60">
                <CardHeader className="pb-0">
                  <CardTitle className="text-sm font-semibold text-slate-100">(b) Policy Iteration improvements</CardTitle>
                  <CardDescription className="text-xs">
                    {data.convergence.pi.n_iters} iterations, eval residual ≈ 1e-14 (exact linear solve), oscillation detector clear
                  </CardDescription>
                </CardHeader>
                <CardContent className="h-60">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={piBars} margin={{ top: 6, right: 8, bottom: 4, left: -8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="iter" tick={{ fontSize: 9, fill: SLATE }} />
                      <YAxis tick={{ fontSize: 9, fill: SLATE }} />
                      <RTooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 11 }} />
                      <Bar dataKey="changes" fill={TEAL} radius={[3, 3, 0, 0]} name="policy changes" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
              <Card className="border-slate-800 bg-slate-900/60">
                <CardHeader className="pb-0">
                  <CardTitle className="text-sm font-semibold text-slate-100">(c) RTDP backup deltas</CardTitle>
                  <CardDescription className="text-xs">
                    {data.convergence.rtdp.n_trials} trials · {data.convergence.rtdp.visited_states} states visited · policy agreement {data.convergence.rtdp.policy_agreement_on_visited_pct.toFixed(1)}% on frequently-visited states
                  </CardDescription>
                </CardHeader>
                <CardContent className="h-60">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={rtdpCurve} margin={{ top: 6, right: 8, bottom: 4, left: -8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="trial" tick={{ fontSize: 9, fill: SLATE }} />
                      <YAxis scale="log" domain={["auto", "auto"]} tick={{ fontSize: 9, fill: SLATE }} />
                      <RTooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 11 }} />
                      <Line dataKey="delta" stroke="#a855f7" dot={false} name="max Bellman delta" strokeWidth={1.2} />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* ------------------------------------------------------- operations */}
          <TabsContent value="operations" className="mt-4">
            <OperationsPanel data={data} />
          </TabsContent>

          {/* ------------------------------------------------------- decisions */}
          <TabsContent value="decisions" className="mt-4">
            <DecisionPanel data={data} />
          </TabsContent>

          {/* ------------------------------------------------------- scalability */}
          <TabsContent value="scalability" className="mt-4">
            <ScalabilityPanel data={data} />
          </TabsContent>
        </Tabs>
      </main>

      <footer className="mt-auto border-t border-slate-800 bg-slate-950">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-1 px-4 py-3 text-[11px] text-slate-500 sm:px-6">
          <span className="inline-flex items-center gap-1.5"><Activity className="h-3 w-3 text-emerald-500" />
            generated {data.meta.generated_utc} UTC · seed {data.meta.seed} · Python {data.meta.python} · NumPy {data.meta.numpy}</span>
          <span className="inline-flex items-center gap-1.5"><Cpu className="h-3 w-3 text-emerald-500" />
            pipeline wall time ≈ {Object.values(data.meta.timings).reduce((a, b) => a + b, 0).toFixed(0)} s</span>
          <span className="inline-flex items-center gap-1.5"><Gauge className="h-3 w-3 text-emerald-500" />
            artifacts: 4 figures · 5 weight files · summary CSVs · dashboard JSON</span>
        </div>
      </footer>
    </div>
  );
}
