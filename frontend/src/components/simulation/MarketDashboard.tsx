"use client";

import { useRef, useEffect, useMemo } from "react";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  DollarSign,
  TrendingUp,
  Users,
  Activity,
  AlertTriangle,
  Target,
} from "@/components/ui/icons";
import type { AgentSnapshot, MarketTickData } from "@/types/market";

interface MarketDashboardProps {
  tam: number;
  captured: number;
  hhi: number;
  agentCount: number;
  agents: AgentSnapshot[];
  status: string;
  tick: number;
  history: MarketTickData[];
  eventLog: string[];
}

function fmt$(n: number): string {
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 10_000) return `$${(n / 1000).toFixed(1)}k`;
  return `$${n.toFixed(0)}`;
}

function fmtPct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

const ICON_SIZE = 11;
const ICON_CLASS = "text-surface-400";

function MetricCard({
  label,
  value,
  icon,
  warn,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  warn?: boolean;
}) {
  return (
    <div
      className={`bg-surface-0/50 border rounded-lg px-3 py-2 ${
        warn ? "border-red-300/50 bg-red-50/30" : "border-surface-200/30"
      }`}
    >
      <div className="flex items-center gap-1.5 mb-0.5">
        {icon}
        <span className="text-[10px] text-surface-500 uppercase tracking-wide">
          {label}
        </span>
      </div>
      <div className="text-sm font-semibold font-mono text-surface-900">
        {value}
      </div>
    </div>
  );
}

function hhiLabel(hhi: number): string {
  if (hhi < 0.15) return "Competitive";
  if (hhi < 0.25) return "Moderate";
  return "Concentrated";
}

export function MarketDashboard({
  tam,
  captured,
  hhi,
  agentCount,
  agents,
  status,
  tick,
  history,
  eventLog,
}: MarketDashboardProps) {
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [eventLog.length]);

  const isCollapsed = status === "collapsed";
  const capturedPct = tam > 0 ? captured / tam : 0;

  // Downsample for charts (max 300 points)
  const chartData = useMemo(() => {
    if (history.length <= 300) return history;
    const step = Math.ceil(history.length / 300);
    return history.filter((_, i) => i % step === 0);
  }, [history]);

  // Build share data for stacked area chart
  const aliveAgentIds = useMemo(() => {
    const ids = new Set<string>();
    for (const h of history) {
      for (const a of h.agents) {
        if (a.alive) ids.add(a.id);
      }
    }
    return Array.from(ids);
  }, [history]);

  const agentColors = useMemo(() => {
    const map: Record<string, string> = {};
    for (const a of agents) {
      map[a.id] = a.color;
    }
    // Also pull from history in case agent is dead now
    for (const h of history) {
      for (const a of h.agents) {
        if (!map[a.id]) map[a.id] = a.color;
      }
    }
    return map;
  }, [agents, history]);

  const shareChartData = useMemo(() => {
    return chartData.map((h) => {
      const point: Record<string, number> = { tick: h.tick };
      // Initialize ALL known agents to 0 so recharts can stack properly
      // (agents spawned mid-sim would be undefined for earlier ticks otherwise)
      for (const id of aliveAgentIds) {
        point[id] = 0;
      }
      for (const a of h.agents) {
        point[a.id] = a.share;
      }
      return point;
    });
  }, [chartData, aliveAgentIds]);

  const aliveAgents = agents.filter((a) => a.alive);
  const deadAgents = agents.filter((a) => !a.alive);

  return (
    <div className="p-3 space-y-3 overflow-y-auto h-full text-xs">
      {/* Metric cards */}
      <div className="grid grid-cols-2 gap-1.5">
        <MetricCard
          label="TAM"
          value={fmt$(tam)}
          icon={<Target size={ICON_SIZE} className={ICON_CLASS} />}
        />
        <MetricCard
          label="Captured"
          value={`${fmt$(captured)} (${fmtPct(capturedPct)})`}
          icon={<TrendingUp size={ICON_SIZE} className={ICON_CLASS} />}
        />
        <MetricCard
          label="HHI"
          value={`${hhi.toFixed(3)} — ${hhiLabel(hhi)}`}
          icon={
            <Activity
              size={ICON_SIZE}
              className={hhi > 0.25 ? "text-red-400" : hhi > 0.15 ? "text-amber-400" : "text-green-500"}
            />
          }
          warn={hhi > 0.25}
        />
        <MetricCard
          label="Active Firms"
          value={`${agentCount}`}
          icon={<Users size={ICON_SIZE} className={ICON_CLASS} />}
        />
        <MetricCard
          label="Day"
          value={`${tick}`}
          icon={<DollarSign size={ICON_SIZE} className={ICON_CLASS} />}
        />
        <MetricCard
          label="Status"
          value={isCollapsed ? "COLLAPSED" : "Operating"}
          icon={
            <AlertTriangle
              size={ICON_SIZE}
              className={isCollapsed ? "text-red-400" : "text-green-500"}
            />
          }
          warn={isCollapsed}
        />
      </div>

      {/* Market Share stacked area chart */}
      {shareChartData.length > 1 && (
        <div className="bg-surface-50/50 border border-surface-200/50 rounded-xl p-2">
          <h3 className="text-[10px] font-semibold text-surface-600 uppercase tracking-wider mb-1">
            Market Share
          </h3>
          <div className="h-36">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={shareChartData}>
                <XAxis
                  dataKey="tick"
                  stroke="#e2e8f0"
                  tick={{ fontSize: 8, fill: "#64748b" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  stroke="#e2e8f0"
                  tick={{ fontSize: 8, fill: "#64748b" }}
                  axisLine={false}
                  tickLine={false}
                  width={35}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                  domain={[0, 1]}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#fff",
                    border: "1px solid #e2e8f0",
                    borderRadius: 8,
                    fontSize: 10,
                  }}
                  labelFormatter={(l) => `Day ${l}`}
                  formatter={(v, name) => {
                    const val = Number(v);
                    const agent = agents.find((a) => a.id === String(name));
                    return [`${(val * 100).toFixed(1)}%`, agent?.name ?? String(name)];
                  }}
                />
                {aliveAgentIds.map((id) => (
                  <Area
                    key={id}
                    type="monotone"
                    dataKey={id}
                    stackId="1"
                    fill={agentColors[id] ?? "#888"}
                    stroke={agentColors[id] ?? "#888"}
                    fillOpacity={0.7}
                    strokeWidth={0}
                    isAnimationActive={false}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* HHI + Agent Count chart */}
      {chartData.length > 1 && (
        <div className="bg-surface-50/50 border border-surface-200/50 rounded-xl p-2">
          <h3 className="text-[10px] font-semibold text-surface-600 uppercase tracking-wider mb-1">
            Concentration & Firms
          </h3>
          <div className="h-24">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <XAxis
                  dataKey="tick"
                  stroke="#e2e8f0"
                  tick={{ fontSize: 8, fill: "#64748b" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  yAxisId="left"
                  stroke="#e2e8f0"
                  tick={{ fontSize: 8, fill: "#64748b" }}
                  axisLine={false}
                  tickLine={false}
                  width={35}
                  domain={[0, 1]}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="#e2e8f0"
                  tick={{ fontSize: 8, fill: "#64748b" }}
                  axisLine={false}
                  tickLine={false}
                  width={25}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#fff",
                    border: "1px solid #e2e8f0",
                    borderRadius: 8,
                    fontSize: 10,
                  }}
                  labelFormatter={(l) => `Day ${l}`}
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="hhi"
                  stroke="#f59e0b"
                  strokeWidth={1.5}
                  dot={false}
                  name="HHI"
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="agent_count"
                  stroke="#8b5cf6"
                  strokeWidth={1.5}
                  dot={false}
                  name="Firms"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Agent table */}
      {agents.length > 0 && (
        <div className="bg-surface-50/50 border border-surface-200/50 rounded-xl overflow-hidden">
          <div className="px-3 py-1.5 border-b border-surface-200/50">
            <h3 className="text-[10px] font-semibold text-surface-600 uppercase tracking-wider">
              Firms
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[10px]">
              <thead>
                <tr className="text-surface-400 uppercase tracking-wider border-b border-surface-100">
                  <th className="text-left px-2 py-1.5 font-medium">Firm</th>
                  <th className="text-right px-2 py-1.5 font-medium">Rev</th>
                  <th className="text-right px-2 py-1.5 font-medium">Cash</th>
                  <th className="text-right px-2 py-1.5 font-medium">Share</th>
                  <th className="text-right px-2 py-1.5 font-medium">Util</th>
                  <th className="text-right px-2 py-1.5 font-medium">Bind</th>
                </tr>
              </thead>
              <tbody>
                {aliveAgents.map((a) => (
                  <tr key={a.id} className="border-b border-surface-50 hover:bg-surface-50/50">
                    <td className="px-2 py-1.5 font-medium text-surface-800">
                      <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ backgroundColor: a.color }} />
                      {a.name}
                    </td>
                    <td className="text-right px-2 py-1.5 font-mono text-surface-700">{fmt$(a.revenue)}</td>
                    <td className={`text-right px-2 py-1.5 font-mono ${a.cash < 0 ? "text-red-500" : "text-surface-700"}`}>
                      {fmt$(a.cash)}
                    </td>
                    <td className="text-right px-2 py-1.5 font-mono text-surface-700">{fmtPct(a.share)}</td>
                    <td className="text-right px-2 py-1.5 font-mono text-surface-700">{fmtPct(a.utilization)}</td>
                    <td className="text-right px-2 py-1.5">
                      <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${
                        a.binding_constraint === "capacity"
                          ? "bg-amber-100 text-amber-700"
                          : "bg-blue-100 text-blue-700"
                      }`}>
                        {a.binding_constraint === "capacity" ? "Cap" : "Dem"}
                      </span>
                    </td>
                  </tr>
                ))}
                {deadAgents.map((a) => (
                  <tr key={a.id} className="border-b border-surface-50 opacity-40">
                    <td className="px-2 py-1.5 font-medium text-surface-500 line-through">
                      <span className="inline-block w-2 h-2 rounded-full mr-1.5 bg-surface-300" />
                      {a.name}
                    </td>
                    <td className="text-right px-2 py-1.5 font-mono text-surface-400">--</td>
                    <td className="text-right px-2 py-1.5 font-mono text-red-400">{fmt$(a.cash)}</td>
                    <td className="text-right px-2 py-1.5 font-mono text-surface-400">--</td>
                    <td className="text-right px-2 py-1.5 font-mono text-surface-400">--</td>
                    <td className="text-right px-2 py-1.5 text-[9px] text-red-400 font-medium">Dead</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Event log */}
      {eventLog.length > 0 && (
        <div className="bg-surface-50/50 border border-surface-200/50 rounded-xl overflow-hidden">
          <div className="px-3 py-1.5 border-b border-surface-200/50">
            <h3 className="text-[10px] font-semibold text-surface-600 uppercase tracking-wider">
              Market Events
            </h3>
          </div>
          <div ref={logRef} className="max-h-32 overflow-y-auto p-2 space-y-0.5">
            {eventLog.map((entry, i) => (
              <p key={i} className="text-[10px] text-surface-600 font-mono">
                {entry}
              </p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
