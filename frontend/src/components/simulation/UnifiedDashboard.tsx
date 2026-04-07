"use client";

import { useRef, useEffect, useMemo, useState } from "react";
import {
  AreaChart,
  Area,
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
  Target,
} from "@/components/ui/icons";
import type { CEOReport, UnifiedAgentSnapshot, UnifiedTickData } from "@/types/unified";

interface UnifiedDashboardProps {
  tam: number;
  captured: number;
  hhi: number;
  agentCount: number;
  agents: UnifiedAgentSnapshot[];
  focusedCompanyId: string;
  status: string;
  tick: number;
  history: UnifiedTickData[];
  eventLog: string[];
  onFocusCompany: (companyId: string) => void;
  reports?: CEOReport[] | null;
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

function hhiLabel(hhi: number): string {
  if (hhi < 0.15) return "Competitive";
  if (hhi < 0.25) return "Moderate";
  return "Concentrated";
}

export function UnifiedDashboard({
  tam,
  captured,
  hhi,
  agentCount,
  agents,
  focusedCompanyId,
  status,
  tick,
  history,
  eventLog,
  onFocusCompany,
  reports,
}: UnifiedDashboardProps) {
  const logRef = useRef<HTMLDivElement>(null);
  const [showReports, setShowReports] = useState(false);
  const [activeReportIdx, setActiveReportIdx] = useState(0);

  // Auto-show reports when they arrive
  useEffect(() => {
    if (reports && reports.length > 0) {
      setShowReports(true);
      setActiveReportIdx(0);
    }
  }, [reports]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [eventLog.length]);

  const capturedPct = tam > 0 ? captured / tam : 0;

  const chartData = useMemo(() => {
    if (history.length <= 300) return history;
    const step = Math.ceil(history.length / 300);
    return history.filter((_, i) => i % step === 0);
  }, [history]);

  const allAgentIds = useMemo(() => {
    const ids = new Set<string>();
    for (const h of history) {
      for (const a of h.agents) {
        ids.add(a.id);
      }
    }
    return Array.from(ids);
  }, [history]);

  const agentColors = useMemo(() => {
    const map: Record<string, string> = {};
    for (const a of agents) {
      map[a.id] = a.color;
    }
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
      for (const id of allAgentIds) {
        point[id] = 0;
      }
      for (const a of h.agents) {
        point[a.id] = a.share;
      }
      return point;
    });
  }, [chartData, allAgentIds]);

  const aliveAgents = agents.filter((a) => a.alive);
  const deadAgents = agents.filter((a) => !a.alive);
  const focused = agents.find((a) => a.id === focusedCompanyId);

  return (
    <div className="p-3 space-y-3 overflow-y-auto h-full text-xs">
      {/* Company selector pills */}
      <div className="flex flex-wrap gap-1.5">
        {agents.map((a) => (
          <button
            key={a.id}
            onClick={() => onFocusCompany(a.id)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-medium transition-all ${
              a.id === focusedCompanyId
                ? "ring-2 ring-offset-1 ring-accent shadow-sm"
                : a.alive
                  ? "hover:bg-surface-100"
                  : "opacity-40 line-through"
            } ${a.alive ? "text-surface-800" : "text-surface-400"}`}
            style={{
              backgroundColor:
                a.id === focusedCompanyId ? `${a.color}15` : undefined,
              borderColor: a.id === focusedCompanyId ? a.color : undefined,
            }}
          >
            <span
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ backgroundColor: a.alive ? a.color : "#94a3b8" }}
            />
            {a.name}
            {a.alive && (
              <span className="text-surface-500 ml-0.5">
                {a.location_count}L
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Market metrics */}
      <div className="grid grid-cols-3 gap-1.5">
        <div className="bg-surface-0/50 border border-surface-200/30 rounded-lg px-3 py-2">
          <div className="flex items-center gap-1.5 mb-0.5">
            <Target size={ICON_SIZE} className={ICON_CLASS} />
            <span className="text-[10px] text-surface-500 uppercase tracking-wide">
              TAM
            </span>
          </div>
          <div className="text-sm font-semibold font-mono text-surface-900">
            {fmt$(tam)}
          </div>
        </div>
        <div className="bg-surface-0/50 border border-surface-200/30 rounded-lg px-3 py-2">
          <div className="flex items-center gap-1.5 mb-0.5">
            <TrendingUp size={ICON_SIZE} className={ICON_CLASS} />
            <span className="text-[10px] text-surface-500 uppercase tracking-wide">
              Captured
            </span>
          </div>
          <div className="text-sm font-semibold font-mono text-surface-900">
            {fmtPct(capturedPct)}
          </div>
        </div>
        <div className="bg-surface-0/50 border border-surface-200/30 rounded-lg px-3 py-2">
          <div className="flex items-center gap-1.5 mb-0.5">
            <Activity
              size={ICON_SIZE}
              className={
                hhi > 0.25
                  ? "text-red-400"
                  : hhi > 0.15
                    ? "text-amber-400"
                    : "text-green-500"
              }
            />
            <span className="text-[10px] text-surface-500 uppercase tracking-wide">
              HHI
            </span>
          </div>
          <div className="text-sm font-semibold font-mono text-surface-900">
            {hhiLabel(hhi)}
          </div>
        </div>
      </div>

      {/* Focused company detail */}
      {focused && focused.alive && (
        <div
          className="border rounded-xl p-3 space-y-2"
          style={{ borderColor: `${focused.color}40` }}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: focused.color }}
              />
              <span className="text-xs font-bold text-surface-800">
                {focused.name}
              </span>
              <span className="text-[10px] font-medium text-accent bg-accent/10 px-1.5 py-0.5 rounded">
                Stage {focused.stage}
              </span>
            </div>
            <span className="text-[10px] font-mono text-surface-500">
              {focused.node_count} nodes
            </span>
          </div>
          <div className="grid grid-cols-4 gap-1.5">
            <div>
              <div className="text-[9px] text-surface-400 uppercase">Cash</div>
              <div
                className={`text-xs font-mono font-semibold ${focused.cash < 0 ? "text-red-500" : "text-surface-800"}`}
              >
                {fmt$(focused.cash)}
              </div>
            </div>
            <div>
              <div className="text-[9px] text-surface-400 uppercase">
                Revenue
              </div>
              <div className="text-xs font-mono font-semibold text-surface-800">
                {fmt$(focused.daily_revenue)}/d
              </div>
            </div>
            <div>
              <div className="text-[9px] text-surface-400 uppercase">Locs</div>
              <div className="text-xs font-mono font-semibold text-surface-800">
                {focused.location_count}
              </div>
            </div>
            <div>
              <div className="text-[9px] text-surface-400 uppercase">Sat</div>
              <div className="text-xs font-mono font-semibold text-surface-800">
                {(focused.avg_satisfaction * 100).toFixed(0)}%
              </div>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-1.5">
            <div>
              <div className="text-[9px] text-surface-400 uppercase">Share</div>
              <div className="text-xs font-mono font-semibold text-surface-800">
                {fmtPct(focused.share)}
              </div>
            </div>
            <div>
              <div className="text-[9px] text-surface-400 uppercase">
                Quality
              </div>
              <div className="text-xs font-mono font-semibold text-surface-800">
                {focused.quality.toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-[9px] text-surface-400 uppercase">
                Staff
              </div>
              <div className="text-xs font-mono font-semibold text-surface-800">
                {focused.total_employees}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Market Share chart */}
      {shareChartData.length > 1 && (
        <div className="bg-surface-50/50 border border-surface-200/50 rounded-xl p-2">
          <h3 className="text-[10px] font-semibold text-surface-600 uppercase tracking-wider mb-1">
            Market Share
          </h3>
          <div className="h-32">
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
                  width={30}
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
                    return [
                      `${(val * 100).toFixed(1)}%`,
                      agent?.name ?? String(name),
                    ];
                  }}
                />
                {allAgentIds.map((id) => (
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

      {/* Firm table */}
      {agents.length > 0 && (
        <div className="bg-surface-50/50 border border-surface-200/50 rounded-xl overflow-hidden">
          <div className="px-3 py-1.5 border-b border-surface-200/50">
            <h3 className="text-[10px] font-semibold text-surface-600 uppercase tracking-wider">
              Companies
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[10px]">
              <thead>
                <tr className="text-surface-400 uppercase tracking-wider border-b border-surface-100">
                  <th className="text-left px-2 py-1.5 font-medium">
                    Company
                  </th>
                  <th className="text-right px-2 py-1.5 font-medium">Locs</th>
                  <th className="text-right px-2 py-1.5 font-medium">Rev</th>
                  <th className="text-right px-2 py-1.5 font-medium">Cash</th>
                  <th className="text-right px-2 py-1.5 font-medium">Share</th>
                </tr>
              </thead>
              <tbody>
                {aliveAgents.map((a) => (
                  <tr
                    key={a.id}
                    onClick={() => onFocusCompany(a.id)}
                    className={`border-b border-surface-50 cursor-pointer transition-colors ${
                      a.id === focusedCompanyId
                        ? "bg-accent/5"
                        : "hover:bg-surface-50/50"
                    }`}
                  >
                    <td className="px-2 py-1.5 font-medium text-surface-800">
                      <span
                        className="inline-block w-2 h-2 rounded-full mr-1.5"
                        style={{ backgroundColor: a.color }}
                      />
                      {a.name}
                    </td>
                    <td className="text-right px-2 py-1.5 font-mono text-surface-700">
                      {a.location_count}
                    </td>
                    <td className="text-right px-2 py-1.5 font-mono text-surface-700">
                      {fmt$(a.daily_revenue)}
                    </td>
                    <td
                      className={`text-right px-2 py-1.5 font-mono ${a.cash < 0 ? "text-red-500" : "text-surface-700"}`}
                    >
                      {fmt$(a.cash)}
                    </td>
                    <td className="text-right px-2 py-1.5 font-mono text-surface-700">
                      {fmtPct(a.share)}
                    </td>
                  </tr>
                ))}
                {deadAgents.map((a) => (
                  <tr
                    key={a.id}
                    className="border-b border-surface-50 opacity-40"
                  >
                    <td className="px-2 py-1.5 font-medium text-surface-500 line-through">
                      <span className="inline-block w-2 h-2 rounded-full mr-1.5 bg-surface-300" />
                      {a.name}
                    </td>
                    <td className="text-right px-2 py-1.5 font-mono text-surface-400">
                      --
                    </td>
                    <td className="text-right px-2 py-1.5 font-mono text-surface-400">
                      --
                    </td>
                    <td className="text-right px-2 py-1.5 font-mono text-red-400">
                      {fmt$(a.cash)}
                    </td>
                    <td className="text-right px-2 py-1.5 text-[9px] text-red-400 font-medium">
                      Dead
                    </td>
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
              Events
            </h3>
          </div>
          <div
            ref={logRef}
            className="max-h-32 overflow-y-auto p-2 space-y-0.5"
          >
            {eventLog.map((entry, i) => (
              <p key={i} className="text-[10px] text-surface-600 font-mono">
                {entry}
              </p>
            ))}
          </div>
        </div>
      )}

      {/* View reports button (after sim completes) */}
      {reports && reports.length > 0 && !showReports && (
        <button
          onClick={() => setShowReports(true)}
          className="w-full py-2 rounded-xl bg-accent text-white text-xs font-semibold transition-all hover:bg-accent/90"
        >
          View CEO Reports ({reports.length})
        </button>
      )}

      {/* Report modal overlay */}
      {showReports && reports && reports.length > 0 && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-surface-0 rounded-2xl shadow-2xl w-[600px] max-h-[80vh] flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-3 border-b border-surface-200/50">
              <h2 className="text-sm font-bold text-surface-900">
                CEO Performance Reports
              </h2>
              <button
                onClick={() => setShowReports(false)}
                className="text-surface-400 hover:text-surface-600 text-lg"
              >
                &times;
              </button>
            </div>

            {/* Tabs */}
            <div className="flex border-b border-surface-200/50 px-3 gap-1 overflow-x-auto">
              {reports.map((r, i) => {
                const agent = agents.find((a) => a.name === r.company_name);
                return (
                  <button
                    key={i}
                    onClick={() => setActiveReportIdx(i)}
                    className={`px-3 py-2 text-[11px] font-medium whitespace-nowrap border-b-2 transition-colors ${
                      activeReportIdx === i
                        ? "border-accent text-accent"
                        : "border-transparent text-surface-500 hover:text-surface-700"
                    }`}
                  >
                    <span
                      className="inline-block w-2 h-2 rounded-full mr-1.5"
                      style={{ backgroundColor: agent?.color ?? "#94a3b8" }}
                    />
                    {r.company_name}
                  </button>
                );
              })}
            </div>

            {/* Report content */}
            <div className="flex-1 overflow-y-auto p-5 space-y-4">
              {(() => {
                const r = reports[activeReportIdx];
                if (!r) return null;
                return (
                  <>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[10px] font-medium text-accent bg-accent/10 px-2 py-0.5 rounded">
                        {r.strategy.replace(/_/g, " ").toUpperCase()}
                      </span>
                    </div>

                    <div>
                      <h4 className="text-[10px] font-semibold text-surface-500 uppercase tracking-wider mb-1">
                        Summary
                      </h4>
                      <p className="text-xs text-surface-700 leading-relaxed">
                        {r.performance_summary}
                      </p>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <h4 className="text-[10px] font-semibold text-green-600 uppercase tracking-wider mb-1">
                          What Went Well
                        </h4>
                        <p className="text-xs text-surface-700 leading-relaxed">
                          {r.what_went_well}
                        </p>
                      </div>
                      <div>
                        <h4 className="text-[10px] font-semibold text-red-500 uppercase tracking-wider mb-1">
                          What Went Wrong
                        </h4>
                        <p className="text-xs text-surface-700 leading-relaxed">
                          {r.what_went_wrong}
                        </p>
                      </div>
                    </div>

                    {r.key_decisions.length > 0 && (
                      <div>
                        <h4 className="text-[10px] font-semibold text-surface-500 uppercase tracking-wider mb-1">
                          Key Decisions
                        </h4>
                        <ul className="space-y-1">
                          {r.key_decisions.map((d, j) => (
                            <li
                              key={j}
                              className="text-xs text-surface-700 flex gap-2"
                            >
                              <span className="text-surface-400 flex-shrink-0">
                                &bull;
                              </span>
                              {d}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="border-t border-surface-200/50 pt-3">
                      <h4 className="text-[10px] font-semibold text-surface-500 uppercase tracking-wider mb-1">
                        Final Assessment
                      </h4>
                      <p className="text-xs text-surface-800 font-medium leading-relaxed">
                        {r.final_assessment}
                      </p>
                    </div>
                  </>
                );
              })()}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
