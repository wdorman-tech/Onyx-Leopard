"use client";

import { useEffect, useState, useCallback } from "react";
import {
  UtensilsCrossed,
  Monitor,
  Briefcase,
  HeartPulse,
  Building2,
  Factory,
  Store,
  ShoppingCart,
  Hammer,
  Landmark,
  Film,
  Wheat,
  Truck,
  Hotel,
  GraduationCap,
  Zap,
  Lock,
  X,
  Play,
  Layers,
  TrendingUp,
  GitBranch,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { getIndustries, type Industry } from "@/lib/api";
import { Logo } from "@/components/ui/Logo";

const ICON_MAP: Record<string, LucideIcon> = {
  "utensils-crossed": UtensilsCrossed,
  monitor: Monitor,
  briefcase: Briefcase,
  "heart-pulse": HeartPulse,
  "building-2": Building2,
  factory: Factory,
  store: Store,
  "shopping-cart": ShoppingCart,
  "hard-hat": Hammer,
  landmark: Landmark,
  film: Film,
  wheat: Wheat,
  truck: Truck,
  hotel: Hotel,
  "graduation-cap": GraduationCap,
  zap: Zap,
};

const CATEGORY_COLORS: Record<string, string> = {
  Locations: "bg-positive/10 text-positive",
  Corporate: "bg-accent/10 text-accent",
  External: "bg-warning/10 text-warning",
  Revenue: "bg-complete/10 text-complete",
};

interface IndustryPickerProps {
  onSelect: (slug: string) => void;
}

export function IndustryPicker({ onSelect }: IndustryPickerProps) {
  const [industries, setIndustries] = useState<Industry[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<Industry | null>(null);

  useEffect(() => {
    getIndustries()
      .then(setIndustries)
      .catch(() => {
        setIndustries([
          {
            slug: "restaurant",
            name: "Restaurant / Food Service",
            description:
              "Grow a grilled chicken chain from a single location to a national brand",
            icon: "utensils-crossed",
            playable: true,
            total_nodes: 50,
            growth_stages: 4,
            key_metrics: ["Daily Revenue", "Avg Satisfaction", "Food Cost %", "Total Locations"],
            example_nodes: ["Restaurant Location", "Commissary Kitchen", "Marketing Dept", "Catering Revenue"],
            categories: { Locations: 8, Corporate: 16, External: 6, Revenue: 6 },
          },
        ]);
      })
      .finally(() => setLoading(false));
  }, []);

  const handleCardClick = useCallback((ind: Industry) => {
    setSelected(ind);
  }, []);

  const handleClose = useCallback(() => {
    setSelected(null);
  }, []);

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-surface-0">
        <div className="text-surface-400 text-sm">Loading industries...</div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-surface-0 phase-enter">
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-12 overflow-auto">
        <div className="flex items-center gap-3 mb-2">
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-surface-100 to-surface-50 border border-surface-200/50">
            <Logo size={20} className="text-accent" />
          </div>
          <h1 className="text-2xl font-bold text-surface-900 tracking-wide">
            ONYX LEOPARD
          </h1>
        </div>
        <p className="text-sm text-surface-500 mb-10">
          Choose an industry to simulate
        </p>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 max-w-4xl w-full">
          {industries.map((ind) => {
            const Icon = ICON_MAP[ind.icon] ?? Monitor;

            return (
              <button
                key={ind.slug}
                onClick={() => handleCardClick(ind)}
                className={`
                  group relative flex flex-col items-start gap-3 p-4 rounded-xl border text-left
                  transition-all duration-150 cursor-pointer bg-surface-0
                  border-surface-200 hover:border-accent/40 hover:shadow-lg hover:shadow-accent/5
                  ${!ind.playable ? "opacity-70" : ""}
                `}
              >
                <div
                  className={`
                    flex items-center justify-center w-10 h-10 rounded-lg
                    ${ind.playable ? "bg-accent/10 text-accent" : "bg-surface-100 text-surface-400"}
                  `}
                >
                  <Icon size={20} />
                </div>
                <div>
                  <div className="text-sm font-semibold text-surface-800 leading-tight">
                    {ind.name}
                  </div>
                  <div className="text-xs text-surface-500 mt-1 leading-relaxed line-clamp-2">
                    {ind.description}
                  </div>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-surface-400">
                  <span>{ind.total_nodes} nodes</span>
                  <span className="text-surface-300">|</span>
                  <span>{ind.growth_stages} stages</span>
                </div>
                {!ind.playable && (
                  <div className="absolute top-3 right-3 flex items-center gap-1 text-[10px] text-surface-400 bg-surface-100 px-2 py-0.5 rounded-md">
                    <Lock size={10} />
                    Soon
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {selected && (
        <IndustryDetail
          industry={selected}
          onClose={handleClose}
          onStart={selected.playable ? () => onSelect(selected.slug) : undefined}
        />
      )}
    </div>
  );
}

function IndustryDetail({
  industry,
  onClose,
  onStart,
}: {
  industry: Industry;
  onClose: () => void;
  onStart?: () => void;
}) {
  const Icon = ICON_MAP[industry.icon] ?? Monitor;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-surface-900/40 backdrop-blur-sm phase-enter"
      onClick={onClose}
    >
      <div
        className="bg-surface-0 rounded-2xl shadow-2xl border border-surface-200/50 w-full max-w-lg mx-4 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-6 pb-4">
          <div className="flex items-center gap-3">
            <div
              className={`flex items-center justify-center w-12 h-12 rounded-xl ${
                industry.playable ? "bg-accent/10 text-accent" : "bg-surface-100 text-surface-400"
              }`}
            >
              <Icon size={24} />
            </div>
            <div>
              <h2 className="text-lg font-bold text-surface-900">{industry.name}</h2>
              <p className="text-xs text-surface-500">{industry.description}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="flex items-center justify-center w-8 h-8 rounded-lg hover:bg-surface-100 transition-colors"
          >
            <X size={16} className="text-surface-400" />
          </button>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-3 px-6 pb-4">
          <div className="flex items-center gap-2 bg-surface-50 rounded-lg px-3 py-2.5">
            <Layers size={14} className="text-accent shrink-0" />
            <div>
              <div className="text-base font-bold text-surface-900">{industry.total_nodes}</div>
              <div className="text-[10px] text-surface-500 uppercase tracking-wider">Nodes</div>
            </div>
          </div>
          <div className="flex items-center gap-2 bg-surface-50 rounded-lg px-3 py-2.5">
            <TrendingUp size={14} className="text-positive shrink-0" />
            <div>
              <div className="text-base font-bold text-surface-900">{industry.growth_stages}</div>
              <div className="text-[10px] text-surface-500 uppercase tracking-wider">Stages</div>
            </div>
          </div>
          <div className="flex items-center gap-2 bg-surface-50 rounded-lg px-3 py-2.5">
            <GitBranch size={14} className="text-complete shrink-0" />
            <div>
              <div className="text-base font-bold text-surface-900">
                {Object.keys(industry.categories).length}
              </div>
              <div className="text-[10px] text-surface-500 uppercase tracking-wider">Types</div>
            </div>
          </div>
        </div>

        {/* Node breakdown */}
        <div className="px-6 pb-4">
          <div className="text-[10px] uppercase tracking-wider text-surface-400 font-medium mb-2">
            Node Breakdown
          </div>
          <div className="flex gap-2 flex-wrap">
            {Object.entries(industry.categories).map(([cat, count]) => (
              <span
                key={cat}
                className={`inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-lg ${
                  CATEGORY_COLORS[cat] ?? "bg-surface-100 text-surface-600"
                }`}
              >
                {cat}
                <span className="font-bold">{count}</span>
              </span>
            ))}
          </div>
        </div>

        {/* Key metrics */}
        <div className="px-6 pb-4">
          <div className="text-[10px] uppercase tracking-wider text-surface-400 font-medium mb-2">
            Key Metrics
          </div>
          <div className="flex gap-1.5 flex-wrap">
            {industry.key_metrics.map((m) => (
              <span
                key={m}
                className="text-xs text-surface-600 bg-surface-50 border border-surface-100 px-2.5 py-1 rounded-lg"
              >
                {m}
              </span>
            ))}
          </div>
        </div>

        {/* Example nodes */}
        <div className="px-6 pb-4">
          <div className="text-[10px] uppercase tracking-wider text-surface-400 font-medium mb-2">
            Example Nodes
          </div>
          <div className="flex gap-1.5 flex-wrap">
            {industry.example_nodes.map((n) => (
              <span
                key={n}
                className="text-xs text-surface-700 bg-surface-50 border border-surface-200/50 px-2.5 py-1 rounded-lg"
              >
                {n}
              </span>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-surface-100 bg-surface-50/50">
          <button onClick={onClose} className="btn-ghost">
            Close
          </button>
          {onStart ? (
            <button onClick={onStart} className="btn-primary flex items-center gap-2">
              <Play size={14} />
              Start Simulation
            </button>
          ) : (
            <span className="flex items-center gap-1.5 text-xs text-surface-400 bg-surface-100 px-4 py-2.5 rounded-xl">
              <Lock size={12} />
              Coming Soon
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
