"use client";

import { useState, useCallback, useEffect } from "react";
import type { CompanyGraph } from "@/types/graph";
import type { CompanyProfile } from "@/types/profile";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { FlowCanvas } from "@/components/flow/FlowCanvas";
import { TimeControls } from "@/components/simulation/TimeControls";
import { OutlookSelector } from "@/components/simulation/OutlookSelector";
import { MetricsPanel } from "@/components/simulation/MetricsPanel";
import { EventInjector } from "@/components/simulation/EventInjector";
import { ApiKeyScreen } from "@/components/onboarding/ApiKeyScreen";
import { OnboardingQuestionnaire } from "@/components/onboarding/OnboardingQuestionnaire";
import { ExportButton } from "@/components/toolbar/ExportButton";
import { ImportButton } from "@/components/toolbar/ImportButton";
import { Logo } from "@/components/ui/Logo";
import { Spinner } from "@/components/ui/Spinner";
import { Play, Zap } from "@/components/ui/icons";
import { useSimulation } from "@/hooks/useSimulation";
import { getStatus } from "@/lib/api";

type AppPhase = "loading" | "api_key" | "onboarding" | "simulation";

const LOADING_MESSAGES = [
  "Connecting to simulation engine...",
  "Preparing AI agents...",
  "Loading strategy models...",
  "Initializing workspace...",
];

function LoadingScreen() {
  const [msgIndex, setMsgIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setMsgIndex((i) => (i + 1) % LOADING_MESSAGES.length);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-surface-0 flex items-center justify-center phase-enter">
      <div className="flex flex-col items-center gap-6">
        <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-surface-100 to-surface-50 border border-surface-200 shadow-2xl">
          <Logo size={32} className="text-accent" />
        </div>
        <div className="text-center space-y-2">
          <h1 className="text-lg font-bold text-surface-900 tracking-wide">
            ONYX LEOPARD
          </h1>
          <div className="w-48 h-1 rounded-full bg-surface-200 overflow-hidden">
            <div className="h-full bg-accent rounded-full animate-pulse w-2/3" />
          </div>
          <p className="text-surface-500 text-xs h-4">
            {LOADING_MESSAGES[msgIndex]}
          </p>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const [phase, setPhase] = useState<AppPhase>("loading");
  const [profileSessionId, setProfileSessionId] = useState<string | null>(null);
  const [companyProfile, setCompanyProfile] = useState<CompanyProfile | null>(null);
  const [companyGraph, setCompanyGraph] = useState<CompanyGraph | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<
    Array<{ tick: number; metrics: Record<string, number> }>
  >([]);
  const [outlook, setOutlook] = useState("normal");
  const sim = useSimulation();

  useEffect(() => {
    getStatus()
      .then((s) => setPhase(s.api_key_set ? "onboarding" : "api_key"))
      .catch(() => setPhase("api_key"));
  }, []);

  const handleOnboardingComplete = useCallback(
    (profile: CompanyProfile, graph: CompanyGraph, sessionId: string) => {
      setCompanyProfile(profile);
      setCompanyGraph(graph);
      setProfileSessionId(sessionId);
      setPhase("simulation");
    },
    [],
  );

  const handleGraphUpdate = useCallback((graph: CompanyGraph) => {
    setCompanyGraph(graph);
  }, []);

  const handleStartSimulation = useCallback(async () => {
    if (!companyGraph) return;
    setMetricsHistory([]);
    await sim.start(companyGraph, outlook);
  }, [companyGraph, sim, outlook]);

  const handleImport = useCallback(
    (profile: CompanyProfile, graph: CompanyGraph, sessionId: string) => {
      setCompanyProfile(profile);
      setCompanyGraph(graph);
      setProfileSessionId(sessionId);
      if (phase === "onboarding") {
        setPhase("simulation");
      }
    },
    [phase],
  );

  const displayGraph = sim.graph ?? companyGraph;
  const currentMetrics = sim.sessionId
    ? sim.globalMetrics
    : (companyGraph?.global_metrics ?? {});

  useEffect(() => {
    if (sim.tick > 0) {
      setMetricsHistory((prev) => {
        if (prev.length > 0 && prev[prev.length - 1].tick === sim.tick) return prev;
        return [...prev, { tick: sim.tick, metrics: { ...sim.globalMetrics } }];
      });
    }
  }, [sim.tick, sim.globalMetrics]);

  if (phase === "loading") {
    return <LoadingScreen />;
  }

  if (phase === "api_key") {
    return <ApiKeyScreen onComplete={() => setPhase("onboarding")} />;
  }

  if (phase === "onboarding") {
    return <OnboardingQuestionnaire onComplete={handleOnboardingComplete} />;
  }

  return (
    <div className="h-screen flex flex-col bg-surface-0 phase-enter">
      {/* Header */}
      <header className="flex items-center justify-between px-5 py-3 bg-surface-0/80 backdrop-blur-md border-b border-surface-200/50 z-10">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-br from-surface-100 to-surface-50 border border-surface-200/50">
            <Logo size={16} className="text-accent" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-surface-900 tracking-wide">
              ONYX LEOPARD
            </h1>
            <p className="text-[10px] text-surface-500 uppercase tracking-widest">
              Strategy Simulator
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <ImportButton
            sessionId={profileSessionId}
            onImport={handleImport}
          />
          <ExportButton
            sessionId={profileSessionId}
            simulationSessionId={sim.sessionId}
            profile={companyProfile}
            graph={companyGraph}
          />

          <div className="w-px h-5 bg-surface-200 mx-1" />

          {companyGraph && (
            <OutlookSelector
              outlook={sim.sessionId ? sim.outlook : outlook}
              onSelect={(o) => {
                if (sim.sessionId) {
                  sim.setOutlook(o);
                } else {
                  setOutlook(o);
                }
              }}
              disabled={sim.isComplete}
            />
          )}

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
          {companyGraph && !sim.sessionId && (
            <button
              onClick={handleStartSimulation}
              className="btn-primary group flex items-center gap-2"
              data-tooltip="Start the simulation"
            >
              <Play
                size={14}
                className="group-hover:scale-110 transition-transform"
                fill="currentColor"
                strokeWidth={0}
              />
              Run Simulation
            </button>
          )}
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel — Chat */}
        <div className="w-[400px] flex-shrink-0 border-r border-surface-200/50">
          <ChatPanel graph={companyGraph} onGraphUpdate={handleGraphUpdate} />
        </div>

        {/* Right panel — Flow + Metrics */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 relative">
            <FlowCanvas
              graph={displayGraph}
              previousGraph={sim.previousGraph}
            />

            {/* Floating node count badge */}
            {displayGraph && (
              <div className="absolute top-4 left-4 bg-surface-50/90 backdrop-blur border border-surface-200 rounded-xl px-3 py-1.5 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-accent" />
                <span className="text-xs text-surface-700">
                  {displayGraph.nodes.length} nodes &middot;{" "}
                  {displayGraph.edges.length} edges
                </span>
              </div>
            )}

            {/* Pre-simulation guidance */}
            {companyGraph && !sim.sessionId && (
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-surface-50/90 backdrop-blur border border-accent/20 rounded-xl px-5 py-3 flex items-center gap-3 shadow-lg shadow-accent/5">
                <Zap size={16} className="text-accent flex-shrink-0" />
                <p className="text-xs text-surface-700">
                  Click <span className="text-accent font-medium">&apos;Run Simulation&apos;</span> to begin. Use the chat panel to refine your company structure.
                </p>
              </div>
            )}
          </div>

          {/* Bottom panel — metrics + event injection */}
          {sim.sessionId && (
            <div className="border-t border-surface-200/50 bg-surface-0/80 backdrop-blur-md p-4 space-y-3">
              <MetricsPanel
                globalMetrics={currentMetrics}
                tick={sim.tick}
                history={metricsHistory}
              />
              <EventInjector
                onInject={sim.injectEvent}
                disabled={!sim.playing}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
