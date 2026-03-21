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

interface MetricsPanelProps {
  globalMetrics: Record<string, number>;
  tick: number;
  history: Array<{ tick: number; metrics: Record<string, number> }>;
}

const METRIC_COLORS: Record<string, string> = {
  total_headcount: "#3b82f6",
  total_budget: "#ec4899",
  revenue: "#22c55e",
};

const METRIC_ICONS: Record<string, string> = {
  total_headcount: "\u{1F465}",
  total_budget: "\u{1F4B0}",
  revenue: "\u{1F4C8}",
};

export function MetricsPanel({ globalMetrics, tick, history }: MetricsPanelProps) {
  const metricKeys = Object.keys(globalMetrics);
  const prevMetricsRef = useRef<Record<string, number>>({});

  useEffect(() => {
    prevMetricsRef.current = { ...globalMetrics };
  }, [globalMetrics]);

  return (
    <div className="bg-surface-50/50 border border-surface-200/50 rounded-xl overflow-hidden">
      <div className="px-4 py-2.5 border-b border-surface-200/50 flex items-center justify-between">
        <h3 className="text-xs font-semibold text-surface-700 uppercase tracking-wider">Global Metrics</h3>
        <span className="text-[10px] text-surface-500 font-mono">Week {tick}</span>
      </div>

      <div className="p-3">
        <div className="grid grid-cols-3 gap-2 mb-3">
          {metricKeys.map((key) => {
            const prev = prevMetricsRef.current[key];
            const current = globalMetrics[key];
            const changed = prev !== undefined && prev !== current;
            const direction = changed ? (current > prev ? "up" : "down") : null;

            return (
              <div key={key} className="bg-surface-0/50 border border-surface-200/30 rounded-lg px-3 py-2.5">
                <div className="flex items-center gap-1.5 mb-1">
                  <span className="text-xs">{METRIC_ICONS[key] ?? "\u{1F4CA}"}</span>
                  <span className="text-[10px] text-surface-500 uppercase tracking-wide">{key.replace(/_/g, " ")}</span>
                </div>
                <div
                  className={`text-lg font-semibold font-mono text-surface-900 ${
                    direction === "up" ? "metric-flash-up" : direction === "down" ? "metric-flash-down" : ""
                  }`}
                  key={`${key}-${tick}`}
                >
                  {key.includes("budget") || key.includes("revenue")
                    ? `$${(globalMetrics[key] / 1000).toFixed(0)}k`
                    : globalMetrics[key].toFixed(0)}
                </div>
              </div>
            );
          })}
        </div>

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
