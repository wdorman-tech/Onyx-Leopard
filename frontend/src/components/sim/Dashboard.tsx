"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, AlertTriangle, Bot, Cog, Target, TrendingUp } from "lucide-react";
import type { CeoDecision, CeoReport, CompanyState, TickEvent } from "@/lib/api";
import { COMPANY_COLORS, type FeedItem, type HistoryPoint } from "@/hooks/useSimulation";

const CHART_POINTS = 300;
const HHI_COMPETITIVE = 0.15;
const HHI_MODERATE = 0.25;
const ICON = "text-surface-400";

function fmt$(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `$${(n / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (abs >= 1e4) return `$${(n / 1e3).toFixed(1)}k`;
  return `$${n.toFixed(0)}`;
}

const pct = (n: number) => `${(n * 100).toFixed(1)}%`;
const titleCase = (s: string) => s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

function hhiLabel(hhi: number): string {
  if (hhi < HHI_COMPETITIVE) return "Competitive";
  if (hhi < HHI_MODERATE) return "Moderate";
  return "Concentrated";
}

function actionSummary(d: CeoDecision): string {
  const parts = [
    ...d.spawn.map((k) => `+${k}`),
    ...d.retire.map((k) => `−${k}`),
    d.open_locations ? `open ${d.open_locations} location${d.open_locations > 1 ? "s" : ""}` : "",
    d.price != null ? `price → $${d.price.toFixed(2)}` : "",
    d.marketing_intensity != null ? `marketing → ${d.marketing_intensity.toFixed(2)}` : "",
    d.raise_amount ? `raise ${fmt$(d.raise_amount)}` : "",
  ].filter(Boolean);
  return parts.length ? parts.join(" · ") : "hold";
}

function downsample<T>(rows: T[]): T[] {
  if (rows.length <= CHART_POINTS) return rows;
  const step = Math.ceil(rows.length / CHART_POINTS);
  return rows.filter((_, i) => i % step === 0 || i === rows.length - 1);
}

function Stat({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div>
      <div className="text-[9px] text-surface-400 uppercase">{label}</div>
      <div
        className={`text-xs font-mono font-semibold ${warn ? "text-negative" : "text-surface-800"}`}
      >
        {value}
      </div>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-surface-50/50 border border-surface-200/50 rounded-xl overflow-hidden">
      <div className="px-3 py-1.5 border-b border-surface-200/50">
        <h3 className="text-[10px] font-semibold text-surface-600 uppercase tracking-wider">
          {title}
        </h3>
      </div>
      {children}
    </div>
  );
}

interface DashboardProps {
  latest: TickEvent | null;
  companies: CompanyState[];
  history: HistoryPoint[];
  feed: FeedItem[];
  events: string[];
  reports: CeoReport[] | null;
  focusedId: string;
  onFocus: (companyId: string) => void;
}

export function Dashboard({
  latest,
  companies,
  history,
  feed,
  events,
  reports,
  focusedId,
  onFocus,
}: DashboardProps) {
  const [showReports, setShowReports] = useState(false);
  const eventsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (reports?.length) setShowReports(true);
  }, [reports]);

  useEffect(() => {
    eventsRef.current?.scrollTo({ top: eventsRef.current.scrollHeight });
  }, [events.length]);

  const color = useMemo(() => {
    const map: Record<string, string> = {};
    companies.forEach((c, i) => (map[c.company_id] = COMPANY_COLORS[i % COMPANY_COLORS.length]));
    return map;
  }, [companies]);
  const nameOf = (id: string) => companies.find((c) => c.company_id === id)?.name ?? id;

  const chart = useMemo(
    () =>
      downsample(history).map((h) => ({
        tick: h.tick,
        ...Object.fromEntries(Object.entries(h.shares).map(([id, v]) => [`s:${id}`, v])),
        ...Object.fromEntries(Object.entries(h.cash).map(([id, v]) => [`c:${id}`, v])),
      })),
    [history],
  );

  const alive = companies.filter((c) => c.alive);
  const tam = latest?.tam ?? 0;
  const captured = tam > 0 ? (alive.reduce((s, c) => s + c.core_revenue, 0) * 365) / tam : 0;
  const hhi = alive.reduce((s, c) => s + c.market_share ** 2, 0);
  const focused = companies.find((c) => c.company_id === focusedId) ?? companies[0];
  const recent = [...feed].reverse().slice(0, 60);
  const tooltipStyle = {
    backgroundColor: "#fff",
    border: "1px solid #e2e8f0",
    borderRadius: 8,
    fontSize: 10,
  };
  const axis = { stroke: "#e2e8f0", tick: { fontSize: 8, fill: "#64748b" }, axisLine: false, tickLine: false };

  return (
    <div className="p-3 space-y-3 overflow-y-auto h-full text-xs">
      <div className="flex flex-wrap gap-1.5">
        {companies.map((c) => (
          <button
            key={c.company_id}
            onClick={() => onFocus(c.company_id)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-medium transition-all ${
              c.company_id === focused?.company_id
                ? "ring-2 ring-offset-1 ring-accent shadow-sm"
                : c.alive
                  ? "hover:bg-surface-100"
                  : "opacity-40 line-through"
            } ${c.alive ? "text-surface-800" : "text-surface-400"}`}
          >
            <span
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ backgroundColor: c.alive ? color[c.company_id] : "#94a3b8" }}
            />
            {c.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-4 gap-1.5">
        {[
          { icon: Target, label: "TAM / yr", value: fmt$(tam) },
          { icon: TrendingUp, label: "Captured", value: pct(captured) },
          { icon: Activity, label: "HHI", value: hhiLabel(hhi) },
          {
            icon: AlertTriangle,
            label: "Shocks",
            value: String(
              (latest?.market_shocks.length ?? 0) +
                companies.reduce((s, c) => s + c.shocks.length, 0),
            ),
          },
        ].map(({ icon: Icon, label, value }) => (
          <div
            key={label}
            className="bg-surface-0/50 border border-surface-200/30 rounded-lg px-3 py-2"
          >
            <div className="flex items-center gap-1.5 mb-0.5">
              <Icon size={11} className={ICON} />
              <span className="text-[10px] text-surface-500 uppercase tracking-wide">{label}</span>
            </div>
            <div className="text-sm font-semibold font-mono text-surface-900">{value}</div>
          </div>
        ))}
      </div>

      {focused && (
        <div
          className="border rounded-xl p-3 space-y-2"
          style={{ borderColor: `${color[focused.company_id]}40` }}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: color[focused.company_id] }}
              />
              <span className="text-xs font-bold text-surface-800">{focused.name}</span>
              <span className="text-[10px] font-medium text-accent bg-accent/10 px-1.5 py-0.5 rounded">
                {titleCase(focused.archetype)}
              </span>
              {!focused.alive && (
                <span className="text-[10px] font-medium text-negative bg-negative/10 px-1.5 py-0.5 rounded">
                  Bankrupt
                </span>
              )}
            </div>
            <span className="text-[10px] text-surface-500">
              bottleneck: <span className="font-medium text-surface-700">{focused.binding}</span>
            </span>
          </div>
          <div className="grid grid-cols-4 gap-1.5">
            <Stat label="Cash" value={fmt$(focused.cash)} warn={focused.cash < 0} />
            <Stat label="Revenue/d" value={fmt$(focused.daily_revenue)} />
            <Stat label="Costs/d" value={fmt$(focused.daily_costs)} />
            <Stat
              label="Runway"
              value={focused.runway_months == null ? "∞" : `${focused.runway_months.toFixed(1)}mo`}
              warn={focused.runway_months != null && focused.runway_months < 3}
            />
            <Stat label="Share" value={pct(focused.market_share)} />
            <Stat label="Price" value={`$${focused.price.toFixed(2)}`} />
            <Stat label="Marketing" value={`${focused.marketing_intensity.toFixed(2)}×`} />
            <Stat label="Satisfaction" value={pct(focused.satisfaction)} />
            <Stat label="Utilization" value={pct(focused.utilization)} />
            <Stat label="Staff" value={focused.employees.toLocaleString()} />
            <Stat label="Locations" value={String(focused.locations)} />
            <Stat label="Nodes" value={String(focused.graph.nodes.length)} />
          </div>
          {focused.shocks.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {focused.shocks.map((s) => (
                <span
                  key={s}
                  className="text-[9px] text-warning bg-warning/10 px-1.5 py-0.5 rounded"
                >
                  {titleCase(s)}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {chart.length > 1 && (
        <Panel title="Market share">
          <div className="h-32 p-2">
            <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 1, height: 1 }}>
              <AreaChart data={chart}>
                <XAxis dataKey="tick" {...axis} />
                <YAxis
                  {...axis}
                  width={30}
                  domain={[0, 1]}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                />
                <Tooltip
                  contentStyle={tooltipStyle}
                  labelFormatter={(l) => `Day ${l}`}
                  formatter={(v, name) => [pct(Number(v)), nameOf(String(name).slice(2))]}
                />
                {companies.map((c) => (
                  <Area
                    key={c.company_id}
                    type="monotone"
                    dataKey={`s:${c.company_id}`}
                    stackId="1"
                    fill={color[c.company_id]}
                    stroke={color[c.company_id]}
                    fillOpacity={0.7}
                    strokeWidth={0}
                    isAnimationActive={false}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}

      {chart.length > 1 && (
        <Panel title="Cash">
          <div className="h-28 p-2">
            <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 1, height: 1 }}>
              <LineChart data={chart}>
                <XAxis dataKey="tick" {...axis} />
                <YAxis {...axis} width={42} tickFormatter={(v) => fmt$(Number(v))} />
                <Tooltip
                  contentStyle={tooltipStyle}
                  labelFormatter={(l) => `Day ${l}`}
                  formatter={(v, name) => [fmt$(Number(v)), nameOf(String(name).slice(2))]}
                />
                {companies.map((c) => (
                  <Line
                    key={c.company_id}
                    type="monotone"
                    dataKey={`c:${c.company_id}`}
                    stroke={color[c.company_id]}
                    dot={false}
                    strokeWidth={1.5}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}

      {recent.length > 0 && (
        <Panel title="CEO decisions">
          <div className="max-h-56 overflow-y-auto divide-y divide-surface-100">
            {recent.map(({ companyId, name, decision }, i) => (
              <div key={`${companyId}-${decision.day}-${i}`} className="px-3 py-2 space-y-0.5">
                <div className="flex items-center gap-1.5 text-[10px]">
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: color[companyId] }}
                  />
                  <span className="font-medium text-surface-700">{name}</span>
                  <span className="text-surface-400">day {decision.day}</span>
                  <span
                    className={`ml-auto flex items-center gap-1 px-1.5 rounded ${
                      decision.source === "ai"
                        ? "text-accent bg-accent/10"
                        : "text-surface-500 bg-surface-100"
                    }`}
                  >
                    {decision.source === "ai" ? <Bot size={9} /> : <Cog size={9} />}
                    {decision.source === "ai" ? "AI" : "Rule"}
                  </span>
                </div>
                <div className="text-[10px] font-mono text-surface-600">
                  {actionSummary(decision)}
                </div>
                <p className="text-[10px] text-surface-500 leading-snug">{decision.reasoning}</p>
              </div>
            ))}
          </div>
        </Panel>
      )}

      {events.length > 0 && (
        <Panel title="Events">
          <div ref={eventsRef} className="max-h-32 overflow-y-auto p-2 space-y-0.5">
            {events.map((e, i) => (
              <p key={i} className="text-[10px] text-surface-600 font-mono">
                {e}
              </p>
            ))}
          </div>
        </Panel>
      )}

      {reports && !showReports && (
        <button
          onClick={() => setShowReports(true)}
          className="w-full py-2 rounded-xl bg-accent text-white text-xs font-semibold hover:bg-accent/90"
        >
          View CEO Reports ({reports.length})
        </button>
      )}
      {reports && showReports && (
        <ReportsModal reports={reports} color={color} onClose={() => setShowReports(false)} />
      )}
    </div>
  );
}

function ReportsModal({
  reports,
  color,
  onClose,
}: {
  reports: CeoReport[];
  color: Record<string, string>;
  onClose: () => void;
}) {
  const [active, setActive] = useState(0);
  const r = reports[active];
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="bg-surface-0 rounded-2xl shadow-2xl w-full max-w-[640px] max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between px-5 py-3 border-b border-surface-200/50">
          <h2 className="text-sm font-bold text-surface-900">CEO post-mortems</h2>
          <button onClick={onClose} className="text-surface-400 hover:text-surface-600 text-lg">
            &times;
          </button>
        </div>
        <div className="flex border-b border-surface-200/50 px-3 gap-1 overflow-x-auto">
          {reports.map((rep, i) => (
            <button
              key={rep.company_id}
              onClick={() => setActive(i)}
              className={`px-3 py-2 text-[11px] font-medium whitespace-nowrap border-b-2 transition-colors ${
                active === i
                  ? "border-accent text-accent"
                  : "border-transparent text-surface-500 hover:text-surface-700"
              }`}
            >
              <span
                className="inline-block w-2 h-2 rounded-full mr-1.5"
                style={{ backgroundColor: color[rep.company_id] ?? "#94a3b8" }}
              />
              {rep.company_name}
            </button>
          ))}
        </div>
        {r && (
          <div className="flex-1 overflow-y-auto p-5 space-y-4 text-xs text-surface-700">
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-medium text-accent bg-accent/10 px-2 py-0.5 rounded">
                {titleCase(r.strategy).toUpperCase()}
              </span>
              <span className="text-[10px] text-surface-400">
                {r.source === "ai" ? "written by the AI CEO" : "heuristic summary"}
              </span>
            </div>
            <section>
              <h4 className="text-[10px] font-semibold text-surface-500 uppercase mb-1">Summary</h4>
              <p className="leading-relaxed">{r.performance_summary}</p>
            </section>
            <div className="grid grid-cols-2 gap-3">
              <section>
                <h4 className="text-[10px] font-semibold text-positive uppercase mb-1">
                  What went well
                </h4>
                <p className="leading-relaxed">{r.what_went_well}</p>
              </section>
              <section>
                <h4 className="text-[10px] font-semibold text-negative uppercase mb-1">
                  What went wrong
                </h4>
                <p className="leading-relaxed">{r.what_went_wrong}</p>
              </section>
            </div>
            {r.key_decisions.length > 0 && (
              <section>
                <h4 className="text-[10px] font-semibold text-surface-500 uppercase mb-1">
                  Key decisions
                </h4>
                <ul className="space-y-1 list-disc pl-4">
                  {r.key_decisions.map((d, j) => (
                    <li key={j}>{d}</li>
                  ))}
                </ul>
              </section>
            )}
            <section className="border-t border-surface-200/50 pt-3">
              <h4 className="text-[10px] font-semibold text-surface-500 uppercase mb-1">
                Final assessment
              </h4>
              <p className="font-medium text-surface-800 leading-relaxed">{r.final_assessment}</p>
            </section>
          </div>
        )}
      </div>
    </div>
  );
}
