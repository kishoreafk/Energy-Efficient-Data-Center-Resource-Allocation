"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, Line, ReferenceLine,
  ResponsiveContainer, Scatter, ScatterChart, Tooltip as RTooltip, XAxis, YAxis,
} from "recharts";
import { Server, TrendingDown } from "lucide-react";
import type { DashboardData, Trace } from "./types";

const EMERALD = "#10b981";
const TEAL = "#14b8a6";
const AMBER = "#f59e0b";
const ORANGE = "#f97316";
const RED = "#ef4444";
const SLATE = "#94a3b8";

const U_COLORS = ["#1e293b", "#fbbf24", "#f97316", "#ef4444"]; // Sleep/Idle/Medium/High
const TAU_COLORS = ["#5eead4", "#fb923c", "#ef4444"]; // Cool/Warm/Hot
const CLASS_COLORS = [
  "#94a3b8", // hold
  "#fbbf24", "#d97706", "#92400e", // wake c0..c2
  "#c084fc", "#7e22ce", "#3b0764", // sleep c0..c2
  "#5eead4", "#14b8a6", "#0f766e", // route c0..c2
];

const tooltipStyle = { background: "#0f172a", border: "1px solid #1e293b", fontSize: 11 };

/* ------------------------------------------------------------------ Pareto */
export function ParetoPanel({ data, floorPct }: { data: DashboardData; floorPct: number }) {
  const pts = data.pareto.points;
  const sweep = pts.filter((p) => p.kind === "dp-sweep");
  const good = sweep.filter((p) => p.guardrail_ok);
  const bad = sweep.filter((p) => !p.guardrail_ok);
  const main = pts.filter((p) => p.kind === "dp-main");
  const base = pts.filter((p) => p.kind === "baseline");

  return (
    <Card className="border-slate-800 bg-slate-900/60">
      <CardHeader className="pb-0">
        <CardTitle className="text-sm font-semibold text-slate-100">Energy vs SLA Pareto frontier</CardTitle>
        <CardDescription className="text-xs">
          DP penalty sweep (P_drop 10→200) + heuristic baselines · guardrail threshold w₃·P_drop &gt;{" "}
          {data.pareto.guardrail_min_w3_times_drop.toFixed(1)} — the P=10 point violates it and drifts to the
          lazy-shutdown corner (Issue 2 demonstrated empirically)
        </CardDescription>
      </CardHeader>
      <CardContent className="h-[26rem]">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 12, right: 24, bottom: 18, left: -8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="energy_kwh" type="number" name="energy" unit=" kWh"
              domain={[38, 88]} ticks={[40, 50, 60, 70, 80]} allowDecimals={false}
              tick={{ fontSize: 9, fill: SLATE }}
              label={{ value: "Total energy over the 24h trace [kWh]", position: "insideBottom", offset: -8, fontSize: 10, fill: SLATE }}
            />
            <YAxis
              dataKey="sla_violation_rate_pct" type="number" name="SLA" unit="%"
              domain={[11.1, 11.72]} ticks={[11.2, 11.3, 11.4, 11.5, 11.6, 11.7]}
              tick={{ fontSize: 9, fill: SLATE }} tickFormatter={(v: number) => v.toFixed(1)}
            />
            <RTooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: "3 3" }} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            <ReferenceLine
              y={floorPct} stroke={EMERALD} strokeDasharray="4 4"
              label={{ value: `arrival-surge floor (${floorPct.toFixed(2)}%)`, position: "insideBottomRight", fontSize: 9, fill: EMERALD, offset: 4 }}
            />
            <Scatter name="DP sweep (guardrail OK)" data={good} line shape="circle" fill={EMERALD} stroke={EMERALD} fillOpacity={0.9} />
            <Scatter name="DP sweep (guardrail violated)" data={bad} line shape="circle" fill={RED} stroke={RED} fillOpacity={0.9} />
            <Scatter name="DP main (P_drop=50)" data={main} shape="star" fill="#34d399" stroke="#064e3b" />
            <Scatter name="Greedy Consolidation" data={base.filter((p) => p.label.startsWith("Greedy"))} shape="square" fill={AMBER} stroke="#000" fillOpacity={0.85} />
            <Scatter name="Always-On" data={base.filter((p) => p.label === "Always-On")} shape="diamond" fill={SLATE} stroke="#000" fillOpacity={0.85} />
          </ScatterChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

/* -------------------------------------------------------------- Operations */
function UHeatmap({ trace }: { trace: Trace }) {
  const n = trace.u.length; // clusters
  return (
    <div className="space-y-1.5">
      {trace.u.map((row, ci) => (
        <div key={ci} className="flex items-center gap-2">
          <div className="w-14 shrink-0 text-right text-[10px] leading-tight text-slate-400">
            c{ci}
            <div className="text-[9px] text-slate-500">tier {ci}</div>
          </div>
          <div className="grid flex-1 grid-cols-24 gap-[1px] sm:grid-cols-24" style={{ gridTemplateColumns: "repeat(24, minmax(0, 1fr))" }}>
            {Array.from({ length: 24 }, (_, h) => {
              const slice = row.slice(h * 4, h * 4 + 4);
              const counts = [0, 0, 0, 0];
              slice.forEach((v) => (counts[v] += 1));
              const dom = counts.indexOf(Math.max(...counts));
              return (
                <div
                  key={h}
                  title={`hour ${h}:00 — cluster ${ci}: ${slice.map((v) => ["S", "I", "M", "H"][v]).join("")}`}
                  className="h-5 rounded-[2px] sm:h-6"
                  style={{ background: U_COLORS[dom] }}
                />
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

function TauStrip({ trace }: { trace: Trace }) {
  return (
    <div className="flex items-center gap-2">
      <div className="w-14 shrink-0 text-right text-[10px] text-slate-400">thermal</div>
      <div className="grid flex-1" style={{ gridTemplateColumns: "repeat(24, minmax(0, 1fr))" }}>
        {Array.from({ length: 24 }, (_, h) => {
          const slice = trace.tau.slice(h * 4, h * 4 + 4);
          const counts = [0, 0, 0];
          slice.forEach((v) => (counts[v] += 1));
          const dom = counts.indexOf(Math.max(...counts));
          return (
            <div
              key={h}
              title={`hour ${h}:00 — ${["Cool", "Warm", "Hot"][dom]} (dominant)`}
              className="h-3 rounded-[2px]"
              style={{ background: TAU_COLORS[dom], opacity: 0.9 }}
            />
          );
        })}
      </div>
    </div>
  );
}

export function OperationsPanel({ data }: { data: DashboardData }) {
  const vi = data.traces["Value Iteration"];
  const ao = data.traces["Always-On (Round-Robin)"];
  const qData = vi.hours.map((h, i) => ({
    hour: Number(h.toFixed(1)),
    q: vi.q[i],
    lam: vi.lam[i],
    qAo: ao.q[i],
    drops: vi.dropped[i],
  }));
  const cumEnergy: { hour: number; dp: number; ao: number }[] = [];
  {
    let a = 0, b = 0;
    vi.hours.forEach((h, i) => {
      a += vi.energy_kwh[i];
      b += ao.energy_kwh[i];
      cumEnergy.push({ hour: Number(h.toFixed(1)), dp: Number(a.toFixed(3)), ao: Number(b.toFixed(3)) });
    });
  }

  return (
    <div className="space-y-4">
      <Card className="border-slate-800 bg-slate-900/60">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-sm font-semibold text-slate-100">
            <Server className="h-4 w-4 text-emerald-400" /> Server states under the DP policy — 24h diurnal trace
          </CardTitle>
          <CardDescription className="text-xs">
            dominant state per 15-min bin · night: single efficient tier at Medium/High → morning: scale-out begins →
            midday peak: all three tiers engaged → evening: sleep (hysteresis from the boot-spike cost)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <UHeatmap trace={vi} />
          <TauStrip trace={vi} />
          <div className="flex flex-wrap gap-3 pt-1 text-[10px] text-slate-400">
            {[["Sleep", U_COLORS[0]], ["Idle", U_COLORS[1]], ["Medium", U_COLORS[2]], ["High", U_COLORS[3]]].map(([l, c]) => (
              <span key={l} className="inline-flex items-center gap-1">
                <span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: c }} /> {l}
              </span>
            ))}
            <span className="ml-3" />
            {[["Cool", TAU_COLORS[0]], ["Warm", TAU_COLORS[1]], ["Hot", TAU_COLORS[2]]].map(([l, c]) => (
              <span key={l} className="inline-flex items-center gap-1">
                <span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: c }} /> {l}
              </span>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card className="border-slate-800 bg-slate-900/60">
          <CardHeader className="pb-0">
            <CardTitle className="text-sm font-semibold text-slate-100">Queue vs arrival rate (DP policy)</CardTitle>
            <CardDescription className="text-xs">the queue absorbs surges; drops occur only when a single step delivers more than {data.spec.q_max} jobs</CardDescription>
          </CardHeader>
          <CardContent className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={qData} margin={{ top: 6, right: 8, bottom: 0, left: -14 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="hour" tick={{ fontSize: 9, fill: SLATE }} unit="h" />
                <YAxis tick={{ fontSize: 9, fill: SLATE }} />
                <RTooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line dataKey="lam" stroke={SLATE} dot={false} strokeDasharray="5 3" name="λ_t" />
                <Line dataKey="q" stroke={EMERALD} dot={false} strokeWidth={1.4} name="queue (DP)" />
                <Line dataKey="drops" stroke={RED} dot={false} strokeWidth={1} name="drops/step (DP)" />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <Card className="border-slate-800 bg-slate-900/60">
          <CardHeader className="pb-0">
            <CardTitle className="flex items-center gap-2 text-sm font-semibold text-slate-100">
              <TrendingDown className="h-4 w-4 text-emerald-400" /> Cumulative energy: DP vs Always-On
            </CardTitle>
            <CardDescription className="text-xs">
              gap widens overnight (full consolidation) and stays ~{(
                (1 - data.policies.find((p) => p.policy === "Value Iteration")!.total_energy_kwh /
                  data.policies.find((p) => p.policy === "Always-On (Round-Robin)")!.total_energy_kwh) * 100
              ).toFixed(0)}% through the peak
            </CardDescription>
          </CardHeader>
          <CardContent className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={cumEnergy} margin={{ top: 6, right: 8, bottom: 0, left: -14 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="hour" tick={{ fontSize: 9, fill: SLATE }} unit="h" />
                <YAxis tick={{ fontSize: 9, fill: SLATE }} />
                <RTooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line dataKey="ao" stroke={SLATE} dot={false} strokeWidth={1.4} name="Always-On (cum. kWh)" />
                <Line dataKey="dp" stroke={EMERALD} dot={false} strokeWidth={1.8} name="DP (cum. kWh)" />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------ Decision maps */
export function DecisionPanel({ data }: { data: DashboardData }) {
  const { class_names, slices } = data.decision_boundary;
  return (
    <Card className="border-slate-800 bg-slate-900/60">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold text-slate-100">Policy decision maps (peak-regime optimal policy)</CardTitle>
        <CardDescription className="text-xs">
          chosen action class for every (queue length q, thermal τ) on three state slices — cell captions show the
          chosen cooling level. Note the wake escalation with queue pressure (route → wake bigger tier) and the
          sleep decisions that shed idle capacity under thermal load.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid gap-4 lg:grid-cols-3">
          {slices.map((sl) => (
            <div key={sl.title} className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
              <div className="mb-2 text-[11px] font-semibold text-slate-200">{sl.title}</div>
              <div className="space-y-1">
                {sl.grid.map((row, tau) => (
                  <div key={tau} className="flex items-center gap-1">
                    <div className="w-8 shrink-0 text-[9px] text-slate-500">{["Cool", "Warm", "Hot"][tau]}</div>
                    <div className="grid flex-1 gap-[3px]" style={{ gridTemplateColumns: `repeat(${row.length}, minmax(0, 1fr))` }}>
                      {row.map((cls, q) => (
                        <div
                          key={q}
                          title={`q=${q}, τ=${["Cool", "Warm", "Hot"][tau]} → ${class_names[cls]} (cooling ${sl.cooling[tau][q] ? "high" : "low"})`}
                          className="flex h-9 items-center justify-center rounded text-[8px] font-semibold text-slate-950/80"
                          style={{ background: CLASS_COLORS[cls] }}
                        >
                          {sl.cooling[tau][q] ? "Hi" : "Lo"}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-1.5 text-center text-[9px] text-slate-500">queue length q →</div>
            </div>
          ))}
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-[10px] text-slate-300">
          {class_names.map((c, i) => (
            <span key={c} className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: CLASS_COLORS[i] }} /> {c}
            </span>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

/* ------------------------------------------------------------ Scalability */
export function ScalabilityPanel({ data }: { data: DashboardData }) {
  if (!data.adp) {
    return (
      <Card className="border-slate-800 bg-slate-900/60">
        <CardContent className="p-6 text-sm text-slate-400">
          Run <code className="text-emerald-400">python main.py</code> (without --skip-adp) to generate the N=8
          scalability demo.
        </CardContent>
      </Card>
    );
  }
  const adp = data.adp;
  const td = adp.td_curve.map((v, i) => ({ step: i * (adp.n_steps / adp.td_curve.length), td: v }));
  const bars = Object.entries(adp.metrics).map(([name, m]) => ({
    name: name.replace(" (N=8)", ""),
    energy: Number(m.total_energy_kwh.toFixed(1)),
    fill: name.startsWith("Linear-ADP") ? EMERALD : name.startsWith("Always") ? SLATE : ORANGE,
  }));
  const colors = [TEAL, "#c084fc", "#7e22ce", "#3b0764", "#5eead4", "#14b8a6", "#0f766e", "#94a3b8", AMBER, RED];

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="border-slate-800 bg-slate-900/60 lg:col-span-1">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-semibold text-slate-100">Why approximation is required</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-xs text-slate-300">
            <p>
              For N = {adp.n_clusters} clusters the exact tabular MDP needs
              <br />
              <b className="text-emerald-400">|S| = 4^{adp.n_clusters} × 7 × 3 = {adp.tabular_state_space.toLocaleString()}</b> states and
            </p>
            <p>
              <b className="text-emerald-400">~{(adp.tabular_sparse_entries / 1e9).toFixed(2)}e9</b> sparse transition
              entries — tens of GB and hours of Bellman sweeps. Infeasible.
            </p>
            <p>
              The factored alternative fits V(s) ≈ θ·φ(s) over {adp.feature_names.length} hand-engineered features and
              trains by semi-gradient TD(0) on simulator rollouts in <b>{adp.time_s.toFixed(0)} s</b> ({adp.n_steps.toLocaleString()} steps).
            </p>
            <div className="mt-2 grid grid-cols-2 gap-1">
              {adp.feature_names.map((f, i) => (
                <Badge key={f} variant="outline" className="justify-start border-slate-700 text-[9px] text-slate-300">
                  <span className="h-1.5 w-1.5 rounded-full" style={{ background: colors[i % colors.length] }} />
                  &nbsp;{f}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
        <Card className="border-slate-800 bg-slate-900/60 lg:col-span-1">
          <CardHeader className="pb-0">
            <CardTitle className="text-sm font-semibold text-slate-100">N=8 energy (24h trace)</CardTitle>
            <CardDescription className="text-xs">linear-VFA ADP consolidates {adp.n_clusters} tiers and still scales out at peak</CardDescription>
          </CardHeader>
          <CardContent className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={bars} margin={{ top: 6, right: 8, bottom: 16, left: -14 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" tick={{ fontSize: 9, fill: SLATE }} />
                <YAxis tick={{ fontSize: 9, fill: SLATE }} />
                <RTooltip contentStyle={tooltipStyle} />
                <Bar dataKey="energy" radius={[4, 4, 0, 0]}>
                  {bars.map((b) => <Cell key={b.name} fill={b.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <Card className="border-slate-800 bg-slate-900/60 lg:col-span-1">
          <CardHeader className="pb-0">
            <CardTitle className="text-sm font-semibold text-slate-100">TD-error during training</CardTitle>
            <CardDescription className="text-xs">running mean |δ| over {adp.n_steps.toLocaleString()} steps (α decays 1/t)</CardDescription>
          </CardHeader>
          <CardContent className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={td} margin={{ top: 6, right: 8, bottom: 0, left: -14 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="step" tick={{ fontSize: 9, fill: SLATE }} />
                <YAxis tick={{ fontSize: 9, fill: SLATE }} />
                <RTooltip contentStyle={tooltipStyle} />
                <Line dataKey="td" stroke="#a855f7" dot={false} strokeWidth={1.4} name="mean |TD|" />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
