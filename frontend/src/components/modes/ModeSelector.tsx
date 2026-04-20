"use client";

import { Sparkles, Users } from "lucide-react";
import { Logo } from "@/components/ui/Logo";

interface ModeSelectorProps {
  onSelectUnified: () => void;
  onSelectAdaptive: () => void;
  onSelectMarket: () => void;
  onSelectGrowth: () => void;
}

export function ModeSelector({
  onSelectUnified,
  onSelectAdaptive,
  onSelectMarket,
  onSelectGrowth,
}: ModeSelectorProps) {
  return (
    <div className="h-screen flex flex-col bg-surface-0 phase-enter relative">
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-12">
        <div className="flex items-center gap-3 mb-2">
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-surface-100 to-surface-50 border border-surface-200/50">
            <Logo size={20} className="text-accent" />
          </div>
          <h1 className="text-2xl font-bold text-surface-900 tracking-wide">
            ONYX LEOPARD
          </h1>
        </div>
        <p className="text-sm text-surface-500 mb-10">
          Choose your simulation mode
        </p>

        <div className="max-w-2xl w-full grid grid-cols-1 sm:grid-cols-2 gap-4">
          <button
            onClick={onSelectUnified}
            className="group flex flex-col items-start gap-3 p-6 rounded-2xl border border-surface-200 hover:border-accent hover:bg-accent/5 transition-all text-left"
          >
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-surface-100 group-hover:bg-accent/10 transition-colors">
              <Users size={20} className="text-surface-500 group-hover:text-accent transition-colors" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-surface-800 mb-1">
                Static Simulation
              </h2>
              <p className="text-xs text-surface-500 leading-relaxed">
                Pre-built companies with configurable AI CEOs competing in a shared market
              </p>
            </div>
          </button>

          <button
            onClick={onSelectAdaptive}
            className="group flex flex-col items-start gap-3 p-6 rounded-2xl border border-surface-200 hover:border-accent hover:bg-accent/5 transition-all text-left"
          >
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-surface-100 group-hover:bg-accent/10 transition-colors">
              <Sparkles size={20} className="text-surface-500 group-hover:text-accent transition-colors" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-surface-800 mb-1">
                Custom Adaptive
              </h2>
              <p className="text-xs text-surface-500 leading-relaxed">
                Add your company and AI generates competitors in your niche
              </p>
            </div>
          </button>
        </div>
      </div>

      <div className="absolute bottom-4 right-5 flex items-center gap-4">
        <button
          onClick={onSelectMarket}
          className="text-xs text-surface-400 hover:text-surface-600 transition-colors"
        >
          Market Competition
        </button>
        <button
          onClick={onSelectGrowth}
          className="text-xs text-surface-400 hover:text-surface-600 transition-colors"
        >
          Company Growth
        </button>
      </div>
    </div>
  );
}
