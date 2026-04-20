"use client";

import { ArrowLeft } from "lucide-react";
import { Logo } from "@/components/ui/Logo";
import { TimeControls } from "@/components/simulation/TimeControls";
import { MarketDashboard } from "@/components/simulation/MarketDashboard";
import type { UseMarketSimulationReturn } from "@/hooks/useMarketSimulation";

interface MarketSimViewProps {
  market: UseMarketSimulationReturn;
  presetSlug: string | null;
  onBack: () => void;
}

export function MarketSimView({ market, presetSlug, onBack }: MarketSimViewProps) {
  return (
    <div className="h-screen flex flex-col bg-surface-0 phase-enter">
      <header className="flex items-center justify-between px-5 py-3 bg-surface-0/80 backdrop-blur-md border-b border-surface-200/50 z-10">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="flex items-center justify-center w-8 h-8 rounded-lg hover:bg-surface-100 transition-colors"
          >
            <ArrowLeft size={16} className="text-surface-500" />
          </button>
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-br from-surface-100 to-surface-50 border border-surface-200/50">
            <Logo size={16} className="text-accent" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-surface-900 tracking-wide">
              ONYX LEOPARD
            </h1>
            <p className="text-[10px] text-surface-500 uppercase tracking-widest">
              Market: {presetSlug?.replace(/-/g, " ")}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {market.sessionId && (
            <TimeControls
              playing={market.playing}
              speed={market.speed}
              tick={market.tick}
              isComplete={market.isComplete}
              onPlay={market.play}
              onPause={market.pause}
              onSetSpeed={market.setSpeed}
            />
          )}
        </div>
      </header>

      <div className="flex-1 overflow-hidden">
        <MarketDashboard
          tam={market.tam}
          captured={market.captured}
          hhi={market.hhi}
          agentCount={market.agentCount}
          agents={market.agents}
          status={market.status}
          tick={market.tick}
          history={market.history}
          eventLog={market.eventLog}
        />
      </div>
    </div>
  );
}
