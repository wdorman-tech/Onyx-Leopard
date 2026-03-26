"use client";

import { useEffect, useState } from "react";
import { Play, Swords, Lightbulb, Crown, Boxes } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { getMarketPresets, type MarketPresetResponse } from "@/lib/api";
import { Logo } from "@/components/ui/Logo";

const PRESET_ICONS: Record<string, LucideIcon> = {
  "price-war": Swords,
  "innovation-race": Lightbulb,
  monopoly: Crown,
  commodity: Boxes,
};

const PRESET_COLORS: Record<string, string> = {
  "price-war": "bg-red-50 text-red-600 border-red-200",
  "innovation-race": "bg-violet-50 text-violet-600 border-violet-200",
  monopoly: "bg-amber-50 text-amber-600 border-amber-200",
  commodity: "bg-cyan-50 text-cyan-600 border-cyan-200",
};

interface PresetPickerProps {
  onSelect: (slug: string) => void;
  onBack: () => void;
}

export function PresetPicker({ onSelect, onBack }: PresetPickerProps) {
  const [presets, setPresets] = useState<MarketPresetResponse[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getMarketPresets()
      .then(setPresets)
      .catch(() => {
        setPresets([
          { slug: "price-war", name: "Price War", description: "High marketing spend drives share", alpha: 1.5, beta: 0.5, delta: 0.15, n_0: 8 },
          { slug: "innovation-race", name: "Innovation Race", description: "Quality determines winners", alpha: 0.3, beta: 1.5, delta: 0.05, n_0: 5 },
          { slug: "monopoly", name: "Monopoly", description: "First-mover advantage dominates", alpha: 0.8, beta: 0.8, delta: 0.03, n_0: 3 },
          { slug: "commodity", name: "Commodity", description: "Many substitutable firms", alpha: 0.5, beta: 0.3, delta: 0.12, n_0: 12 },
        ]);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-surface-0">
        <div className="text-surface-400 text-sm">Loading presets...</div>
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
          Choose a market scenario
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl w-full">
          {presets.map((preset) => {
            const Icon = PRESET_ICONS[preset.slug] ?? Boxes;
            const colorClass = PRESET_COLORS[preset.slug] ?? "bg-surface-50 text-surface-600 border-surface-200";

            return (
              <button
                key={preset.slug}
                onClick={() => onSelect(preset.slug)}
                className="group flex flex-col gap-4 p-5 rounded-xl border border-surface-200 bg-surface-0 text-left transition-all duration-150 hover:border-accent/40 hover:shadow-lg hover:shadow-accent/5"
              >
                <div className="flex items-center gap-3">
                  <div className={`flex items-center justify-center w-10 h-10 rounded-lg border ${colorClass}`}>
                    <Icon size={20} />
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-surface-800">
                      {preset.name}
                    </div>
                    <div className="text-xs text-surface-500 mt-0.5">
                      {preset.description}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-2 text-center">
                  <div className="bg-surface-50 rounded-lg py-1.5 px-1">
                    <div className="text-xs font-bold text-surface-700">{preset.alpha}</div>
                    <div className="text-[9px] text-surface-400 uppercase">Mktg</div>
                  </div>
                  <div className="bg-surface-50 rounded-lg py-1.5 px-1">
                    <div className="text-xs font-bold text-surface-700">{preset.beta}</div>
                    <div className="text-[9px] text-surface-400 uppercase">Quality</div>
                  </div>
                  <div className="bg-surface-50 rounded-lg py-1.5 px-1">
                    <div className="text-xs font-bold text-surface-700">{(preset.delta * 100).toFixed(0)}%</div>
                    <div className="text-[9px] text-surface-400 uppercase">Churn</div>
                  </div>
                  <div className="bg-surface-50 rounded-lg py-1.5 px-1">
                    <div className="text-xs font-bold text-surface-700">{preset.n_0}</div>
                    <div className="text-[9px] text-surface-400 uppercase">Firms</div>
                  </div>
                </div>

                <div className="flex items-center justify-end">
                  <span className="flex items-center gap-1.5 text-xs font-medium text-accent group-hover:text-accent-dark transition-colors">
                    <Play size={12} />
                    Start
                  </span>
                </div>
              </button>
            );
          })}
        </div>

        <button
          onClick={onBack}
          className="mt-8 text-xs text-surface-400 hover:text-surface-600 transition-colors"
        >
          Back to mode selection
        </button>
      </div>
    </div>
  );
}
