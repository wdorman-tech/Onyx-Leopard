"use client";

import { useCallback, useState } from "react";
import { ArrowLeft, Building2, Swords, Network } from "lucide-react";
import { Logo } from "@/components/ui/Logo";
import { TimeControls } from "@/components/simulation/TimeControls";
import { CompanyDashboard } from "@/components/simulation/CompanyDashboard";
import { MarketDashboard } from "@/components/simulation/MarketDashboard";
import { UnifiedDashboard } from "@/components/simulation/UnifiedDashboard";
import { ForceGraph } from "@/components/graph/ForceGraph";
import { IndustryPicker } from "@/components/simulation/IndustryPicker";
import { PresetPicker } from "@/components/simulation/PresetPicker";
import { useSimulation } from "@/hooks/useSimulation";
import { useMarketSimulation } from "@/hooks/useMarketSimulation";
import { useUnifiedSimulation } from "@/hooks/useUnifiedSimulation";

type AppMode = "select" | "growth-pick" | "growth-sim" | "market-pick" | "market-sim" | "unified-pick" | "unified-sim";

const STAGE_LABELS: Record<number, string> = {
  1: "Single Location",
  2: "Multi-Location",
  3: "Regional Chain",
  4: "National Chain",
};

function ModeSelector({
  onSelectGrowth,
  onSelectMarket,
  onSelectUnified,
}: {
  onSelectGrowth: () => void;
  onSelectMarket: () => void;
  onSelectUnified: () => void;
}) {
  return (
    <div className="h-screen flex flex-col bg-surface-0 phase-enter">
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
          Choose a simulation mode
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl w-full">
          <button
            onClick={onSelectGrowth}
            className="group flex flex-col items-center gap-4 p-8 rounded-xl border border-surface-200 bg-surface-0 transition-all duration-150 hover:border-accent/40 hover:shadow-lg hover:shadow-accent/5"
          >
            <div className="flex items-center justify-center w-14 h-14 rounded-xl bg-green-50 text-green-600 border border-green-200">
              <Building2 size={28} />
            </div>
            <div className="text-center">
              <div className="text-base font-semibold text-surface-800">
                Company Growth
              </div>
              <div className="text-xs text-surface-500 mt-1 max-w-[200px]">
                Watch a single company grow from one location to a national chain
              </div>
            </div>
          </button>

          <button
            onClick={onSelectMarket}
            className="group flex flex-col items-center gap-4 p-8 rounded-xl border border-surface-200 bg-surface-0 transition-all duration-150 hover:border-accent/40 hover:shadow-lg hover:shadow-accent/5"
          >
            <div className="flex items-center justify-center w-14 h-14 rounded-xl bg-violet-50 text-violet-600 border border-violet-200">
              <Swords size={28} />
            </div>
            <div className="text-center">
              <div className="text-base font-semibold text-surface-800">
                Market Competition
              </div>
              <div className="text-xs text-surface-500 mt-1 max-w-[200px]">
                Simulate multiple firms competing for market share in a dynamic TAM
              </div>
            </div>
          </button>

          <button
            onClick={onSelectUnified}
            className="group flex flex-col items-center gap-4 p-8 rounded-xl border border-surface-200 bg-surface-0 transition-all duration-150 hover:border-accent/40 hover:shadow-lg hover:shadow-accent/5"
          >
            <div className="flex items-center justify-center w-14 h-14 rounded-xl bg-amber-50 text-amber-600 border border-amber-200">
              <Network size={28} />
            </div>
            <div className="text-center">
              <div className="text-base font-semibold text-surface-800">
                Unified Compete
              </div>
              <div className="text-xs text-surface-500 mt-1 max-w-[200px]">
                Multiple companies with full org charts competing in a shared market
              </div>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const sim = useSimulation();
  const market = useMarketSimulation();
  const unified = useUnifiedSimulation();
  const [mode, setMode] = useState<AppMode>("select");
  const [industrySlug, setIndustrySlug] = useState<string | null>(null);
  const [presetSlug, setPresetSlug] = useState<string | null>(null);
  const [unifiedStartMode, setUnifiedStartMode] = useState<string>("identical");
  const [unifiedCompanyCount, setUnifiedCompanyCount] = useState(4);

  // ── Growth mode handlers ──
  const handleSelectIndustry = useCallback(
    (slug: string) => {
      setIndustrySlug(slug);
      setMode("growth-sim");
      sim.start(slug);
    },
    [sim],
  );

  // ── Market mode handlers ──
  const handleSelectPreset = useCallback(
    (slug: string) => {
      setPresetSlug(slug);
      setMode("market-sim");
      market.start(slug);
    },
    [market],
  );

  // ── Unified mode handlers ──
  const handleStartUnified = useCallback(() => {
    setMode("unified-sim");
    unified.start(unifiedStartMode, unifiedCompanyCount);
  }, [unified, unifiedStartMode, unifiedCompanyCount]);

  const handleBack = useCallback(() => {
    setMode("select");
    setIndustrySlug(null);
    setPresetSlug(null);
  }, []);

  // ── Mode selection ──
  if (mode === "select") {
    return (
      <ModeSelector
        onSelectGrowth={() => setMode("growth-pick")}
        onSelectMarket={() => setMode("market-pick")}
        onSelectUnified={() => setMode("unified-pick")}
      />
    );
  }

  // ── Growth: Industry picker ──
  if (mode === "growth-pick") {
    return <IndustryPicker onSelect={handleSelectIndustry} />;
  }

  // ── Market: Preset picker ──
  if (mode === "market-pick") {
    return <PresetPicker onSelect={handleSelectPreset} onBack={handleBack} />;
  }

  // ── Unified: Config picker ──
  if (mode === "unified-pick") {
    const START_MODES = [
      { value: "identical", label: "Identical", desc: "All companies start with equal resources" },
      { value: "randomized", label: "Randomized", desc: "Varied starting cash and conditions" },
      { value: "staggered", label: "Staggered", desc: "Companies enter the market at different times" },
    ];
    return (
      <div className="h-screen flex flex-col bg-surface-0 phase-enter">
        <div className="flex-1 flex flex-col items-center justify-center px-6 py-12">
          <div className="flex items-center gap-3 mb-2">
            <button
              onClick={handleBack}
              className="flex items-center justify-center w-8 h-8 rounded-lg hover:bg-surface-100 transition-colors"
            >
              <ArrowLeft size={16} className="text-surface-500" />
            </button>
            <h1 className="text-xl font-bold text-surface-900 tracking-wide">
              Unified Compete
            </h1>
          </div>
          <p className="text-sm text-surface-500 mb-8">
            Configure the simulation
          </p>

          <div className="max-w-md w-full space-y-6">
            <div>
              <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block">
                Starting Conditions
              </label>
              <div className="space-y-2">
                {START_MODES.map((m) => (
                  <button
                    key={m.value}
                    onClick={() => setUnifiedStartMode(m.value)}
                    className={`w-full flex items-center gap-3 p-3 rounded-lg border transition-all text-left ${
                      unifiedStartMode === m.value
                        ? "border-accent bg-accent/5 ring-1 ring-accent/30"
                        : "border-surface-200 hover:border-surface-300"
                    }`}
                  >
                    <div
                      className={`w-3 h-3 rounded-full border-2 flex-shrink-0 ${
                        unifiedStartMode === m.value
                          ? "border-accent bg-accent"
                          : "border-surface-300"
                      }`}
                    />
                    <div>
                      <div className="text-sm font-medium text-surface-800">
                        {m.label}
                      </div>
                      <div className="text-[11px] text-surface-500">
                        {m.desc}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block">
                Companies ({unifiedCompanyCount})
              </label>
              <input
                type="range"
                min={2}
                max={8}
                value={unifiedCompanyCount}
                onChange={(e) => setUnifiedCompanyCount(Number(e.target.value))}
                className="w-full accent-accent"
              />
              <div className="flex justify-between text-[10px] text-surface-400 mt-1">
                <span>2</span>
                <span>8</span>
              </div>
            </div>

            <button
              onClick={handleStartUnified}
              className="w-full py-3 rounded-xl bg-accent text-white font-semibold text-sm transition-all hover:bg-accent/90 hover:shadow-lg hover:shadow-accent/20"
            >
              Start Simulation
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── Unified: Running simulation ──
  if (mode === "unified-sim") {
    return (
      <div className="h-screen flex flex-col bg-surface-0 phase-enter">
        <header className="flex items-center justify-between px-5 py-3 bg-surface-0/80 backdrop-blur-md border-b border-surface-200/50 z-10">
          <div className="flex items-center gap-3">
            <button
              onClick={handleBack}
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
                Unified Compete
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {unified.sessionId && (
              <TimeControls
                playing={unified.playing}
                speed={unified.speed}
                tick={unified.tick}
                isComplete={unified.isComplete}
                onPlay={unified.play}
                onPause={unified.pause}
                onSetSpeed={unified.setSpeed}
              />
            )}
          </div>
        </header>

        <div className="flex-1 flex overflow-hidden">
          <div className="w-[60%] h-full border-r border-surface-200/50">
            <ForceGraph
              graph={unified.graph}
              companyName={unified.agents.find((a) => a.id === unified.focusedCompanyId)?.name ?? "Company"}
            />
          </div>
          <div className="w-[40%] h-full overflow-hidden">
            <UnifiedDashboard
              tam={unified.tam}
              captured={unified.captured}
              hhi={unified.hhi}
              agentCount={unified.agentCount}
              agents={unified.agents}
              focusedCompanyId={unified.focusedCompanyId}
              status={unified.status}
              tick={unified.tick}
              history={unified.history}
              eventLog={unified.eventLog}
              onFocusCompany={unified.setFocusedCompany}
            />
          </div>
        </div>
      </div>
    );
  }

  // ── Growth: Running simulation ──
  if (mode === "growth-sim") {
    return (
      <div className="h-screen flex flex-col bg-surface-0 phase-enter">
        <header className="flex items-center justify-between px-5 py-3 bg-surface-0/80 backdrop-blur-md border-b border-surface-200/50 z-10">
          <div className="flex items-center gap-3">
            <button
              onClick={handleBack}
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
                {STAGE_LABELS[sim.stage] ?? `Stage ${sim.stage}`}
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

  // ── Market: Running simulation ──
  return (
    <div className="h-screen flex flex-col bg-surface-0 phase-enter">
      <header className="flex items-center justify-between px-5 py-3 bg-surface-0/80 backdrop-blur-md border-b border-surface-200/50 z-10">
        <div className="flex items-center gap-3">
          <button
            onClick={handleBack}
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
