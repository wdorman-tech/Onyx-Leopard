"use client";

import { useRef, useEffect, type ComponentType } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Users,
  DollarSign,
  TrendingUp,
  BarChart3,
  Activity,
  AlertTriangle,
  Gauge,
  Target,
  Layers,
} from "@/components/ui/icons";

interface MetricsPanelProps {
  globalMetrics: Record<string, number>;
  tick: number;
  history: Array<{ tick: number; metrics: Record<string, number> }>;
}

const METRIC_COLORS: Record<string, string> = {
  total_headcount: "#3b82f6",
  total_budget: "#ec4899",
  revenue: "#22c55e",
  avg_health_score: "#f59e0b",
  avg_capacity_utilization: "#8b5cf6",
  departments_at_risk: "#ef4444",
  resource_efficiency: "#06b6d4",
  market_share: "#10b981",
};

const METRIC_ICONS: Record<string, ComponentType<{ size?: number; className?: string }>> = {
  total_headcount: Users,
  total_budget: DollarSign,
  revenue: TrendingUp,
  avg_health_score: Activity,
  avg_capacity_utilization: Gauge,
  departments_at_risk: AlertTriangle,
  resource_efficiency: Layers,
  market_share: Target,
};

const DefaultIcon = BarChart3;

function formatMetricValue(key: string, value: number): string {
  if (key.includes("budget") || key.includes("revenue") || key.includes("efficiency")) {
    if (Math.abs(value) >= 1000000) return `$${(value / 1000000).toFixed(1)}M`;
    if (Math.abs(value) >= 1000) return `$${(value / 1000).toFixed(0)}k`;
    return `$${value.toFixed(0)}`;
  }
  if (key.includes("health") || key.includes("utilization") || key.includes("share")) {
    return `${(value * 100).toFixed(0)}%`;
  }
  return value.toFixed(0);
}

export function MetricsPanel({ globalMetrics, tick, history }: MetricsPanelProps) {
  const metricKeys = Object.keys(globalMetrics);
  const prevMetricsRef = useRef<Record<string, number>>({});

  // Separate core metrics from bio-math metrics for layout
  const coreKeys = metricKeys.filter((k) =>
    ["total_headcount", "total_budget", "revenue"].includes(k)
  );
  const bioKeys = metricKeys.filter(
    (k) => !coreKeys.includes(k) && globalMetrics[k] !== undefined
  );

  useEffect(() => {
    prevMetricsRef.current = { ...globalMetrics };
  }, [globalMetrics]);

  const renderMetricCard = (key: string) => {
    const prev = prevMetricsRef.current[key];
    const current = globalMetrics[key];
    const changed = prev !== undefined && prev !== current;
    const direction = changed ? (current > prev ? "up" : "down") : null;
    const Icon = METRIC_ICONS[key] ?? DefaultIcon;

    // Color coding for at-risk metrics
    const isWarning = key === "departments_at_risk" && current > 0;
    const isHealthLow = key === "avg_health_score" && current < 0.5;

    return (
      <div
        key={key}
        className={`bg-surface-0/50 border rounded-lg px-3 py-2.5 ${
          isWarning || isHealthLow
            ? "border-red-300/50 bg-red-50/30"
            : "border-surface-200/30"
        }`}
      >
        <div className="flex items-center gap-1.5 mb-1">
          <Icon
            size={12}
            className={
              isWarning || isHealthLow ? "text-red-400" : "text-surface-400"
            }
          />
          <span className="text-[10px] text-surface-500 uppercase tracking-wide">
            {key.replace(/_/g, " ")}
          </span>
        </div>
        <div
          className={`text-lg font-semibold font-mono text-surface-900 ${
            direction === "up"
              ? "metric-flash-up"
              : direction === "down"
                ? "metric-flash-down"
                : ""
          }`}
          key={`${key}-${tick}`}
        >
          {formatMetricValue(key, current)}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-surface-50/50 border border-surface-200/50 rounded-xl overflow-hidden">
      <div className="px-4 py-2.5 border-b border-surface-200/50 flex items-center justify-between">
        <h3 className="text-xs font-semibold text-surface-700 uppercase tracking-wider">Global Metrics</h3>
        <span className="text-[10px] text-surface-500 font-mono">Week {tick}</span>
      </div>

      <div className="p-3">
        {/* Core metrics row */}
        <div className="grid grid-cols-3 gap-2 mb-2">
          {coreKeys.map(renderMetricCard)}
        </div>

        {/* Bio-math metrics row */}
        {bioKeys.length > 0 && (
          <div className={`grid gap-2 mb-3 ${
            bioKeys.length <= 3 ? "grid-cols-3" : bioKeys.length <= 4 ? "grid-cols-4" : "grid-cols-3"
          }`}>
            {bioKeys.map(renderMetricCard)}
          </div>
        )}

        {history.length > 1 && (
          <div className="h-32 mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <XAxis
                  dataKey="tick"
                  stroke="#e2e8f0"
                  tick={{ fontSize: 9, fill: "#64748b" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  stroke="#e2e8f0"
                  tick={{ fontSize: 9, fill: "#64748b" }}
                  axisLine={false}
                  tickLine={false}
                  width={40}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#ffffff",
                    border: "1px solid #e2e8f0",
                    borderRadius: 12,
                    fontSize: 11,
                    boxShadow: "0 8px 32px rgba(0,0,0,0.08)",
                  }}
                />
                {metricKeys.map((key) => (
                  <Line
                    key={key}
                    type="monotone"
                    dataKey={`metrics.${key}`}
                    stroke={METRIC_COLORS[key] ?? "#94a3b8"}
                    strokeWidth={2}
                    dot={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}
