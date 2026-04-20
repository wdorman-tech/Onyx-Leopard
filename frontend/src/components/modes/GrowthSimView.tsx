"use client";

import { ArrowLeft } from "lucide-react";
import { Logo } from "@/components/ui/Logo";
import { TimeControls } from "@/components/simulation/TimeControls";
import { CompanyDashboard } from "@/components/simulation/CompanyDashboard";
import { ForceGraph } from "@/components/graph/ForceGraph";
import type { UseSimulationReturn } from "@/hooks/useSimulation";

interface GrowthSimViewProps {
  sim: UseSimulationReturn;
  industrySlug: string | null;
  stageLabels: Record<number, string>;
  onBack: () => void;
}

export function GrowthSimView({ sim, industrySlug, stageLabels, onBack }: GrowthSimViewProps) {
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
              {industrySlug?.replace(/-/g, " ")}
            </p>
          </div>
          {sim.sessionId && (
            <span className="text-[10px] font-medium text-accent bg-accent/10 px-2 py-0.5 rounded-md ml-2">
              Stage {sim.stage}:{" "}
              {stageLabels[sim.stage] ?? `Stage ${sim.stage}`}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {sim.sessionId && (
            <TimeControls
              playing={sim.playing}
              speed={sim.speed}
              tick={sim.tick}
              isComplete={sim.isComplete}
              onPlay={sim.play}
              onPause={sim.pause}
              onSetSpeed={sim.setSpeed}
            />
          )}
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        <div className="w-[60%] h-full border-r border-surface-200/50">
          <ForceGraph
            graph={sim.graph}
            companyName={industrySlug?.replace(/-/g, " ") ?? "Company"}
          />
        </div>
        <div className="w-[40%] h-full overflow-hidden">
          <CompanyDashboard
            metrics={sim.metrics}
            status={sim.status}
            stage={sim.stage}
            tick={sim.tick}
            metricsHistory={sim.metricsHistory}
            eventLog={sim.eventLog}
          />
        </div>
      </div>
    </div>
  );
}
