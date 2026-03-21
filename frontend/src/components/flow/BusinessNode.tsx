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

function BusinessNodeComponent({ data, selected }: NodeProps & { data: BusinessNodeData }) {
  const { nodeData, delta } = data;
  const colors = NODE_COLORS[nodeData.type];
  const primaryMetric = Object.entries(nodeData.metrics)[0];

  return (
    <div
      className={`${colors.bg} ${colors.border} border backdrop-blur-sm rounded-xl px-4 py-3 min-w-[190px] shadow-lg ${colors.glow} hover:shadow-xl hover:scale-[1.02] transition-all duration-150 ${
        selected ? "ring-2 ring-accent/50" : ""
      }`}
    >
      <Handle type="target" position={Position.Left} className="!bg-surface-400 !border-surface-300 !w-2 !h-2" />

      <div className="flex items-center gap-1.5 mb-1">
        <div className={`w-1.5 h-1.5 rounded-full ${colors.badge}`} />
        <span className="text-[9px] text-surface-500 uppercase tracking-widest font-medium">
          {nodeData.type.replace("_", " ")}
        </span>
      </div>
      <div className="text-sm font-semibold text-surface-900 truncate">
        {nodeData.label}
      </div>

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
        </div>
      )}

      <Handle type="source" position={Position.Right} className="!bg-surface-400 !border-surface-300 !w-2 !h-2" />
    </div>
  );
}

export const BusinessNode = memo(BusinessNodeComponent);
