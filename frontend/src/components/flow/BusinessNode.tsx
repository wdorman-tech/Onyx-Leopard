import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { NodeData, NodeType } from "@/types/graph";

const NODE_COLORS: Record<NodeType, { bg: string; border: string; badge: string; glow: string }> = {
  department: { bg: "bg-accent/10", border: "border-accent/30", badge: "bg-accent", glow: "shadow-accent/10" },
  team: { bg: "bg-secondary/10", border: "border-secondary/30", badge: "bg-secondary", glow: "shadow-secondary/10" },
  role: { bg: "bg-surface-100", border: "border-surface-300", badge: "bg-surface-500", glow: "shadow-surface-500/10" },
  revenue_stream: { bg: "bg-accent/10", border: "border-accent/30", badge: "bg-accent", glow: "shadow-accent/10" },
  cost_center: { bg: "bg-negative/10", border: "border-negative/30", badge: "bg-negative", glow: "shadow-negative/10" },
  external: { bg: "bg-warning/10", border: "border-warning/30", badge: "bg-warning", glow: "shadow-warning/10" },
};

const HEALTH_COLORS = {
  healthy: "bg-green-500",
  stressed: "bg-yellow-500",
  critical: "bg-red-500",
} as const;

const CELL_CYCLE_LABELS: Record<number, string> = {
  1: "G1",
  2: "S",
  3: "G2",
  4: "M",
};

interface BusinessNodeData extends Record<string, unknown> {
  nodeData: NodeData;
  delta?: Record<string, number>;
}

function formatMetric(key: string, value: number): string {
  if (key === "budget" || key === "revenue") {
    return value >= 1000000 ? `$${(value / 1000000).toFixed(1)}M` : `$${(value / 1000).toFixed(0)}k`;
  }
  return value.toFixed(0);
}

function getHealthStatus(score: number | undefined): keyof typeof HEALTH_COLORS {
  if (score === undefined) return "healthy";
  if (score >= 0.6) return "healthy";
  if (score >= 0.3) return "stressed";
  return "critical";
}

function BusinessNodeComponent({ data, selected }: NodeProps & { data: BusinessNodeData }) {
  const { nodeData, delta } = data;
  const colors = NODE_COLORS[nodeData.type];
  const metrics = nodeData.metrics;

  // Bio-math metrics
  const healthScore = metrics.health_score;
  const capacityUtil = metrics.capacity_utilization;
  const signalActivation = metrics.signal_activation ?? 0;
  const apoptosisTriggered = metrics.apoptosis_triggered === 1;
  const cellCyclePhase = metrics.cell_cycle_phase;
  const healthStatus = getHealthStatus(healthScore);

  // Pick primary metric to display (skip bio-math derived metrics)
  const bioKeys = new Set([
    "health_score", "capacity_utilization", "signal_activation",
    "apoptosis_triggered", "wind_down_remaining", "cell_cycle_phase",
    "adoption_r0", "adoption_pct",
  ]);
  const displayMetrics = Object.entries(metrics).filter(([k]) => !bioKeys.has(k));
  const primaryMetric = displayMetrics[0];

  // Apoptosis styling
  const apoptosisClasses = apoptosisTriggered
    ? "border-red-500/60 opacity-60"
    : "";

  // Signal activation glow
  const signalGlow = signalActivation > 0.1
    ? `shadow-[0_0_${Math.round(signalActivation * 16)}px_rgba(59,130,246,${(signalActivation * 0.4).toFixed(2)})]`
    : "";

  return (
    <div
      className={`${colors.bg} ${apoptosisTriggered ? "border-red-500/60" : colors.border} border backdrop-blur-sm rounded-xl px-4 py-3 min-w-[190px] shadow-lg ${colors.glow} hover:shadow-xl hover:scale-[1.02] transition-all duration-150 ${
        selected ? "ring-2 ring-accent/50" : ""
      } ${apoptosisTriggered ? "opacity-60" : ""}`}
      style={signalActivation > 0.1 ? {
        boxShadow: `0 0 ${Math.round(signalActivation * 16)}px rgba(59, 130, 246, ${(signalActivation * 0.4).toFixed(2)})`,
      } : undefined}
    >
      <Handle type="target" position={Position.Left} id="left" className="!bg-surface-400 !border-surface-300 !w-2 !h-2" />
      <Handle type="target" position={Position.Top} id="top" className="!bg-surface-400 !border-surface-300 !w-2 !h-2" />

      <div className="flex items-center gap-1.5 mb-1">
        {/* Health indicator dot */}
        {healthScore !== undefined && (
          <div
            className={`w-2 h-2 rounded-full ${HEALTH_COLORS[healthStatus]} ${healthStatus === "critical" ? "animate-pulse" : ""}`}
            title={`Health: ${(healthScore * 100).toFixed(0)}%`}
          />
        )}
        <span className="text-[9px] text-surface-500 uppercase tracking-widest font-medium">
          {nodeData.type.replace("_", " ")}
        </span>

        {/* Cell cycle phase badge */}
        {cellCyclePhase !== undefined && (
          <span className={`text-[8px] font-mono font-bold px-1 py-0.5 rounded ${
            cellCyclePhase === 4 ? "bg-green-500/20 text-green-700" : "bg-surface-200 text-surface-600"
          }`}>
            {CELL_CYCLE_LABELS[cellCyclePhase] ?? "G1"}
          </span>
        )}

        {/* Apoptosis badge */}
        {apoptosisTriggered && (
          <span className="text-[8px] font-semibold px-1.5 py-0.5 rounded bg-red-500/20 text-red-600 animate-pulse">
            Shutting Down
          </span>
        )}

        {/* Expansion ready badge */}
        {cellCyclePhase === 4 && !apoptosisTriggered && (
          <span className="text-[8px] font-semibold px-1.5 py-0.5 rounded bg-green-500/20 text-green-700">
            Ready to Expand
          </span>
        )}
      </div>

      <div className="text-sm font-semibold text-surface-900 truncate">
        {nodeData.label}
      </div>

      {/* Capacity utilization bar */}
      {capacityUtil !== undefined && (
        <div className="mt-1.5 h-1 bg-surface-200 rounded-full overflow-hidden" title={`Capacity: ${(capacityUtil * 100).toFixed(0)}%`}>
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              capacityUtil > 0.9 ? "bg-red-500" : capacityUtil > 0.7 ? "bg-yellow-500" : "bg-green-500"
            }`}
            style={{ width: `${Math.min(capacityUtil * 100, 100)}%` }}
          />
        </div>
      )}

      {primaryMetric && (
        <div className="flex items-center gap-2 mt-2">
          <span className={`${colors.badge}/20 text-surface-800 text-[10px] font-mono font-medium px-2 py-0.5 rounded-md border border-current/10`}>
            {formatMetric(primaryMetric[0], primaryMetric[1])}
          </span>
          {delta && delta[primaryMetric[0]] !== undefined && delta[primaryMetric[0]] !== 0 && (
            <span
              className={`text-[10px] font-medium font-mono ${
                delta[primaryMetric[0]] > 0 ? "text-positive" : "text-negative"
              } metric-changed`}
            >
              {delta[primaryMetric[0]] > 0 ? "+" : ""}
              {delta[primaryMetric[0]]}
            </span>
          )}

          {/* Market share trend arrow for competitors */}
          {nodeData.type === "external" && delta?.market_share !== undefined && delta.market_share !== 0 && (
            <span className={`text-[10px] ${delta.market_share > 0 ? "text-red-500" : "text-green-500"}`}>
              {delta.market_share > 0 ? "\u2191" : "\u2193"}
            </span>
          )}
        </div>
      )}

      {/* SIR adoption progress for revenue streams */}
      {metrics.adoption_pct !== undefined && metrics.adoption_pct > 0 && (
        <div className="mt-1.5 flex items-center gap-1">
          <div className="flex-1 h-1.5 bg-surface-200 rounded-full overflow-hidden flex">
            <div
              className="h-full bg-blue-400"
              style={{ width: `${Math.max(0, (1 - metrics.adoption_pct - (metrics.adoption_pct * 0.3)) * 100)}%` }}
              title="Susceptible"
            />
            <div
              className="h-full bg-green-500"
              style={{ width: `${metrics.adoption_pct * 100}%` }}
              title="Adopters"
            />
            <div
              className="h-full bg-surface-400"
              style={{ width: `${metrics.adoption_pct * 30}%` }}
              title="Churned"
            />
          </div>
          <span className="text-[8px] text-surface-500 font-mono">
            R0:{metrics.adoption_r0?.toFixed(1) ?? "?"}
          </span>
        </div>
      )}

      <Handle type="source" position={Position.Right} id="right" className="!bg-surface-400 !border-surface-300 !w-2 !h-2" />
      <Handle type="source" position={Position.Bottom} id="bottom" className="!bg-surface-400 !border-surface-300 !w-2 !h-2" />
    </div>
  );
}

export const BusinessNode = memo(BusinessNodeComponent);
