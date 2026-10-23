"use client";

import { useState } from "react";
import { ArrowLeft, Bot, Scale } from "lucide-react";
import { Logo } from "@/components/ui/Logo";
import { ForceGraph } from "@/components/graph/ForceGraph";
import { Dashboard } from "@/components/sim/Dashboard";
import { TimeControls } from "@/components/sim/TimeControls";
import { formatUsd } from "@/lib/api";
import type { SimulationState } from "@/hooks/useSimulation";

interface SimViewProps {
  sim: SimulationState;
  modelName: string | null;
  onBack: () => void;
}

export function SimView({ sim, modelName, onBack }: SimViewProps) {
  const [focusedId, setFocusedId] = useState("");
  const tick = sim.latest?.tick ?? 0;
  const maxTicks = sim.latest?.max_ticks ?? 0;
  const subtitle = sim.companies.length
    ? `${sim.companies[0].name} · ${sim.companies.length} companies`
    : "Starting…";

  return (
    <div className="h-screen flex flex-col bg-surface-0 phase-enter">
      <header className="relative flex items-center justify-between gap-3 px-5 py-3 bg-surface-0/80 backdrop-blur-md border-b border-surface-200/50 z-[60]">
        <div className="flex items-center gap-3 min-w-0">
          <button
            onClick={onBack}
            className="flex items-center justify-center w-8 h-8 rounded-lg hover:bg-surface-100 transition-colors"
            aria-label="Back to setup"
          >
            <ArrowLeft size={16} className="text-surface-500" />
          </button>
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-br from-surface-100 to-surface-50 border border-surface-200/50">
            <Logo size={16} className="text-accent" />
          </div>
          <div className="min-w-0">
            <h1 className="text-sm font-bold text-surface-900 tracking-wide whitespace-nowrap">
              ONYX LEOPARD
            </h1>
            <p className="text-[10px] text-surface-500 uppercase tracking-widest truncate">
              {subtitle}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {sim.ai ? (
            <div
              className={`flex items-center gap-1.5 text-[10px] px-2.5 py-1.5 rounded-lg border ${
                sim.ai.error
                  ? "border-negative/40 text-negative bg-negative/5"
                  : "border-surface-200 text-surface-600"
              }`}
              title={
                sim.ai.error ??
                `${sim.ai.calls} AI calls on ${sim.ai.model}` +
                  (sim.ai.paced_rounds
                    ? ` · ${sim.ai.paced_rounds} monthly rounds skipped to pace the budget`
                    : "")
              }
            >
              <Bot size={12} />
              <span className="font-medium truncate max-w-[9rem]">{modelName ?? sim.ai.model}</span>
              <span className="font-mono whitespace-nowrap">
                {formatUsd(sim.ai.spent_usd)} / {formatUsd(sim.ai.budget_usd)}
              </span>
            </div>
          ) : (
            sim.latest && (
              <span className="text-[10px] text-surface-500 px-2">Heuristic CEOs</span>
            )
          )}
          {sim.jev && (
            <div
              className={`flex items-center gap-1.5 text-[10px] px-2.5 py-1.5 rounded-lg border ${
                sim.jev.error
                  ? "border-negative/40 text-negative bg-negative/5"
                  : "border-surface-200 text-surface-600"
              }`}
              title={
                sim.jev.error ??
                `${sim.jev.calls} Jev calls · ${formatUsd(sim.jev.cost_usd)}` +
                  (sim.jev.forced
                    ? ` · ${sim.jev.forced} survival cuts skipped Jev (cash would run out first)`
                    : "")
              }
            >
              <Scale size={12} />
              <span className="font-medium">Jev</span>
              <span className="font-mono whitespace-nowrap">
                {sim.jev.approved} ok · {sim.jev.vetoed} veto
                {sim.jev.forced ? ` · ${sim.jev.forced} forced` : ""}
              </span>
            </div>
          )}
          <TimeControls
            playing={sim.status === "running"}
            speed={sim.speed}
            tick={tick}
            maxTicks={maxTicks}
            isComplete={sim.status === "complete"}
            onPlay={sim.play}
            onPause={sim.pause}
            onSetSpeed={sim.setSpeed}
          />
        </div>
      </header>

      {sim.error && (
        <div className="px-5 py-2 text-xs text-negative bg-negative/5 border-b border-negative/20">
          {sim.error}
        </div>
      )}

      <div className="flex-1 flex overflow-hidden relative">
        <div className="w-[60%] h-full border-r border-surface-200/50 relative">
          {sim.thinking && (
            <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface-0/95 border border-accent/30 shadow-lg shadow-accent/10 text-[11px] font-medium text-surface-700">
              <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
              AI CEOs are deciding…
            </div>
          )}
          <ForceGraph graph={sim.graph} multiCompany onFocusCompany={setFocusedId} />
        </div>
        <div className="w-[40%] h-full overflow-hidden">
          <Dashboard
            latest={sim.latest}
            companies={sim.companies}
            history={sim.history}
            feed={sim.feed}
            events={sim.events}
            reports={sim.reports}
            savedRun={sim.savedRun}
            focusedId={focusedId}
            onFocus={setFocusedId}
          />
        </div>
      </div>
    </div>
  );
}
