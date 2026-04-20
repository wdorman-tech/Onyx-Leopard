"use client";

import { ArrowLeft } from "lucide-react";
import { Logo } from "@/components/ui/Logo";
import { TimeControls } from "@/components/simulation/TimeControls";
import { UnifiedDashboard } from "@/components/simulation/UnifiedDashboard";
import { ForceGraph } from "@/components/graph/ForceGraph";
import type { GraphData } from "@/types/graph";
import type { CEOReport, UnifiedAgentSnapshot, UnifiedTickData } from "@/types/unified";

interface UnifiedSimViewProps {
  subtitle: string;
  sessionId: string | null;
  playing: boolean;
  speed: number;
  tick: number;
  isComplete: boolean;
  ceoThinking: boolean;
  tam: number;
  captured: number;
  hhi: number;
  agentCount: number;
  agents: UnifiedAgentSnapshot[];
  focusedCompanyId: string;
  mergedGraph: GraphData | null;
  founderType?: string;
  status: string;
  history: UnifiedTickData[];
  eventLog: string[];
  reports: CEOReport[] | null;
  durationYears?: number;
  onPlay: () => Promise<void>;
  onPause: () => Promise<void>;
  onSetSpeed: (multiplier: number) => Promise<void>;
  onFocusCompany: (companyId: string) => void;
  onBack: () => void;
}

export function UnifiedSimView({
  subtitle,
  sessionId,
  playing,
  speed,
  tick,
  isComplete,
  ceoThinking,
  tam,
  captured,
  hhi,
  agentCount,
  agents,
  focusedCompanyId,
  mergedGraph,
  founderType,
  status,
  history,
  eventLog,
  reports,
  durationYears,
  onPlay,
  onPause,
  onSetSpeed,
  onFocusCompany,
  onBack,
}: UnifiedSimViewProps) {
  return (
    <div className="h-screen flex flex-col bg-surface-0 phase-enter">
      <header className="relative flex items-center justify-between px-5 py-3 bg-surface-0/80 backdrop-blur-md border-b border-surface-200/50 z-[60]">
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
              {subtitle}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {sessionId && (
            <TimeControls
              playing={playing}
              speed={speed}
              tick={tick}
              isComplete={isComplete}
              onPlay={onPlay}
              onPause={onPause}
              onSetSpeed={onSetSpeed}
              durationYears={durationYears}
            />
          )}
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden relative">
        {ceoThinking && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-surface-900/60 backdrop-blur-sm">
            <div className="flex flex-col items-center gap-4 px-8 py-6 rounded-2xl bg-surface-0/95 border border-accent/30 shadow-2xl shadow-accent/10">
              <div className="flex gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-accent animate-[bounce_1s_ease-in-out_infinite_0ms]" />
                <span className="w-2.5 h-2.5 rounded-full bg-accent animate-[bounce_1s_ease-in-out_infinite_200ms]" />
                <span className="w-2.5 h-2.5 rounded-full bg-accent animate-[bounce_1s_ease-in-out_infinite_400ms]" />
              </div>
              <p className="text-sm font-semibold text-surface-800 tracking-wide">
                AI CEOs are deliberating
              </p>
              <p className="text-xs text-surface-500 max-w-[260px] text-center">
                Agents are reviewing quarterly results and making strategic decisions
              </p>
            </div>
          </div>
        )}

        <div className="w-[60%] h-full border-r border-surface-200/50">
          <ForceGraph
            graph={mergedGraph}
            multiCompany
            founderType={founderType}
            onFocusCompany={onFocusCompany}
          />
        </div>
        <div className="w-[40%] h-full overflow-hidden">
          <UnifiedDashboard
            tam={tam}
            captured={captured}
            hhi={hhi}
            agentCount={agentCount}
            agents={agents}
            focusedCompanyId={focusedCompanyId}
            status={status}
            tick={tick}
            history={history}
            eventLog={eventLog}
            onFocusCompany={onFocusCompany}
            reports={reports}
          />
        </div>
      </div>
    </div>
  );
}
