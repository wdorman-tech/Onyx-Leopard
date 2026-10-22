"use client";

import { Sparkles } from "lucide-react";
import { Logo } from "@/components/ui/Logo";

interface ModeSelectorProps {
  onSelectUnified: () => void;
}

export function ModeSelector({ onSelectUnified }: ModeSelectorProps) {
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

        <div className="max-w-md w-full">
          <button
            onClick={onSelectUnified}
            className="group w-full flex flex-col items-start gap-3 p-6 rounded-2xl border border-surface-200 hover:border-accent hover:bg-accent/5 transition-all text-left"
          >
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-surface-100 group-hover:bg-accent/10 transition-colors">
              <Sparkles size={20} className="text-surface-500 group-hover:text-accent transition-colors" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-surface-800 mb-1">
                Static Simulation
              </h2>
              <p className="text-xs text-surface-500 leading-relaxed">
                Run the v2 simulation engine with real tick state and org graph updates
              </p>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}
