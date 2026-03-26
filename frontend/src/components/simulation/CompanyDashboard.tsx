"use client";

import { useRef, useEffect } from "react";
import {
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
  TrendingDown,
  Users,
  Building2,
  Activity,
  AlertTriangle,
  Target,
} from "@/components/ui/icons";
import type { TickData } from "@/types/graph";

interface CompanyDashboardProps {
  metrics: Record<string, number>;
  status: string;
  stage: number;
  tick: number;
  metricsHistory: TickData[];
  eventLog: string[];
}

interface MetricCardProps {
  label: string;
  value: string;
  icon: React.ReactNode;
  warn?: boolean;
}

function MetricCard({ label, value, icon, warn }: MetricCardProps) {
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

function fmt$(n: number): string {
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 10_000) return `$${(n / 1000).toFixed(1)}k`;
  return `$${n.toFixed(0)}`;
}

const STAGE_LABELS: Record<number, string> = {
  1: "Single Location",
  2: "Multi-Location",
  3: "Regional Chain",
  4: "National Chain",
};

const ICON_SIZE = 11;
const ICON_CLASS = "text-surface-400";

export function CompanyDashboard({
  metrics,
  status,
  stage,
  tick,
  metricsHistory,
  eventLog,
}: CompanyDashboardProps) {
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [eventLog.length]);

  const cash = metrics.cash ?? 0;
  const revenue = metrics.daily_revenue ?? 0;
  const costs = metrics.daily_costs ?? 0;
  const locations = metrics.total_locations ?? 1;
  const employees = metrics.total_employees ?? 0;
  const satisfaction = metrics.avg_satisfaction ?? 0;
  const isBankrupt = status === "bankrupt";

  // Downsample history for charts (max 200 points)
  const chartData =
    metricsHistory.length > 200
      ? metricsHistory.filter((_, i) => i % Math.ceil(metricsHistory.length / 200) === 0)
      : metricsHistory;

  return (
    <div className="p-3 space-y-3 overflow-y-auto h-full text-xs">
      {/* Stage badge */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-medium text-accent bg-accent/10 px-2 py-0.5 rounded-md">
          Stage {stage}: {STAGE_LABELS[stage] ?? `Stage ${stage}`}
        </span>
        <span className="text-[10px] text-surface-500 font-mono">
          Year {Math.floor(tick / 365) + 1}, Day {tick % 365}
        </span>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 gap-1.5">
        <MetricCard
          label="Cash"
          value={fmt$(cash)}
          icon={<DollarSign size={ICON_SIZE} className={cash < 10000 ? "text-red-400" : ICON_CLASS} />}
          warn={cash < 10000}
        />
        <MetricCard
          label="Revenue"
          value={`${fmt$(revenue)}/day`}
          icon={<TrendingUp size={ICON_SIZE} className={ICON_CLASS} />}
        />
        <MetricCard
          label="Costs"
          value={`${fmt$(costs)}/day`}
          icon={<TrendingDown size={ICON_SIZE} className={ICON_CLASS} />}
        />
        <MetricCard
          label="Locations"
          value={`${locations}`}
          icon={<Building2 size={ICON_SIZE} className={ICON_CLASS} />}
        />
        <MetricCard
          label="Employees"
          value={`${employees}`}
          icon={<Users size={ICON_SIZE} className={ICON_CLASS} />}
        />
        <MetricCard
          label="Satisfaction"
          value={`${(satisfaction * 100).toFixed(0)}%`}
          icon={<Activity size={ICON_SIZE} className={satisfaction < 0.4 ? "text-red-400" : ICON_CLASS} />}
          warn={satisfaction < 0.4}
        />
        <MetricCard
          label="Profit"
          value={`${fmt$(revenue - costs)}/day`}
          icon={<Target size={ICON_SIZE} className={revenue - costs < 0 ? "text-red-400" : "text-green-500"} />}
          warn={revenue - costs < 0}
        />
        <MetricCard
          label="Status"
          value={isBankrupt ? "BANKRUPT" : "Operating"}
          icon={<AlertTriangle size={ICON_SIZE} className={isBankrupt ? "text-red-400" : "text-green-500"} />}
          warn={isBankrupt}
        />
      </div>

      {/* Cash chart */}
      {chartData.length > 1 && (
        <div className="bg-surface-50/50 border border-surface-200/50 rounded-xl p-2">
          <h3 className="text-[10px] font-semibold text-surface-600 uppercase tracking-wider mb-1">
            Cash
          </h3>
          <div className="h-28">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <XAxis dataKey="tick" stroke="#e2e8f0" tick={{ fontSize: 8, fill: "#64748b" }} axisLine={false} tickLine={false} />
                <YAxis stroke="#e2e8f0" tick={{ fontSize: 8, fill: "#64748b" }} axisLine={false} tickLine={false} width={45} tickFormatter={(v) => fmt$(Number(v))} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#fff", border: "1px solid #e2e8f0", borderRadius: 8, fontSize: 10 }}
                  formatter={(v) => [fmt$(Number(v)), "Cash"]}
                  labelFormatter={(l) => `Day ${l}`}
                />
                <Line type="monotone" dataKey="metrics.cash" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Revenue + Locations chart */}
      {chartData.length > 1 && (
        <div className="bg-surface-50/50 border border-surface-200/50 rounded-xl p-2">
          <h3 className="text-[10px] font-semibold text-surface-600 uppercase tracking-wider mb-1">
            Revenue & Locations
          </h3>
          <div className="h-24">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <XAxis dataKey="tick" stroke="#e2e8f0" tick={{ fontSize: 8, fill: "#64748b" }} axisLine={false} tickLine={false} />
                <YAxis yAxisId="left" stroke="#e2e8f0" tick={{ fontSize: 8, fill: "#64748b" }} axisLine={false} tickLine={false} width={40} tickFormatter={(v) => fmt$(Number(v))} />
                <YAxis yAxisId="right" orientation="right" stroke="#e2e8f0" tick={{ fontSize: 8, fill: "#64748b" }} axisLine={false} tickLine={false} width={25} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#fff", border: "1px solid #e2e8f0", borderRadius: 8, fontSize: 10 }}
                  labelFormatter={(l) => `Day ${l}`}
                />
                <Line yAxisId="left" type="monotone" dataKey="metrics.daily_revenue" stroke="#22c55e" strokeWidth={1.5} dot={false} name="Revenue" />
                <Line yAxisId="right" type="monotone" dataKey="metrics.total_locations" stroke="#8b5cf6" strokeWidth={1.5} dot={false} name="Locations" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Event log */}
      {eventLog.length > 0 && (
        <div className="bg-surface-50/50 border border-surface-200/50 rounded-xl overflow-hidden">
          <div className="px-3 py-1.5 border-b border-surface-200/50">
            <h3 className="text-[10px] font-semibold text-surface-600 uppercase tracking-wider">
              Growth Events
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
