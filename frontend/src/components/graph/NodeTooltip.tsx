"use client";

import { useEffect, useRef } from "react";
import type { NodeCategory } from "@/types/graph";

const CATEGORY_COLORS: Record<NodeCategory, string> = {
  root: "#ec4899",
  location: "#22c55e",
  corporate: "#3b82f6",
  external: "#f59e0b",
  revenue: "#8b5cf6",
};

const CATEGORY_LABELS: Record<NodeCategory, string> = {
  root: "Company",
  location: "Location",
  corporate: "Corporate",
  external: "External",
  revenue: "Revenue",
};

interface NodeTooltipProps {
  label: string;
  type: string;
  category: NodeCategory;
  metrics: Record<string, number>;
  x: number;
  y: number;
  onClose: () => void;
}

export function NodeTooltip({ label, type, category, metrics, x, y, onClose }: NodeTooltipProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    }
    window.addEventListener("keydown", handleKey);
    window.addEventListener("mousedown", handleClick, { capture: true });
    return () => {
      window.removeEventListener("keydown", handleKey);
      window.removeEventListener("mousedown", handleClick, { capture: true });
    };
  }, [onClose]);

  const metricEntries = Object.entries(metrics);
  const color = CATEGORY_COLORS[category] ?? "#94a3b8";

  return (
    <div
      ref={ref}
      className="absolute z-50 pointer-events-auto"
      style={{ left: x + 16, top: y - 12 }}
    >
      <div className="bg-surface-0/95 backdrop-blur-sm border border-surface-200 rounded-lg px-3 py-2.5 shadow-xl max-w-[220px]">
        <div className="flex items-center gap-2 mb-1">
          <span
            className="w-2.5 h-2.5 rounded-full flex-shrink-0"
            style={{ backgroundColor: color }}
          />
          <span className="text-xs font-semibold text-surface-800 truncate">
            {label}
          </span>
        </div>
        <div className="text-[10px] text-surface-500 mb-1.5">
          {CATEGORY_LABELS[category]} &middot; {type.replace(/_/g, " ")}
        </div>
        {metricEntries.length > 0 && (
          <div className="border-t border-surface-200 pt-1.5 space-y-0.5">
            {metricEntries.map(([key, val]) => (
              <div key={key} className="flex items-center justify-between text-[10px]">
                <span className="text-surface-500 capitalize">{key}</span>
                <span className="text-surface-800 font-mono">
                  {key === "satisfaction"
                    ? `${Math.round(val * 100)}%`
                    : val.toLocaleString(undefined, { maximumFractionDigits: 1 })}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
