"use client";

import { useCallback, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Plus } from "lucide-react";
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

type AppMode = "unified-pick" | "growth-pick" | "growth-sim" | "market-pick" | "market-sim" | "unified-sim";

const DEFAULT_STAGE_LABELS: Record<number, string> = {
  1: "Single Location",
  2: "Multi-Location",
  3: "Regional Chain",
  4: "National Chain",
};

const AGENT_NAMES = [
  "Alpha Corp", "Beta Inc", "Gamma Ltd", "Delta Co", "Epsilon Group",
  "Zeta Holdings", "Eta Ventures", "Theta Partners",
];

const CEO_STRATEGIES = [
  { value: "aggressive_growth", label: "Aggressive Growth", desc: "Rapid expansion, heavy marketing" },
  { value: "quality_focus", label: "Quality Focus", desc: "Satisfaction and quality above all" },
  { value: "cost_leader", label: "Cost Leader", desc: "Minimize costs, operational efficiency" },
  { value: "balanced", label: "Balanced", desc: "Steady growth, no overextension" },
  { value: "market_dominator", label: "Market Dominator", desc: "Pursue market share aggressively" },
  { value: "survivor", label: "Survivor", desc: "Cash preservation, strong margins" },
];

export default function Home() {
  const router = useRouter();
  const sim = useSimulation();
  const market = useMarketSimulation();
  const unified = useUnifiedSimulation();
  const [mode, setMode] = useState<AppMode>("unified-pick");
  const [industrySlug, setIndustrySlug] = useState<string | null>(null);
  const [presetSlug, setPresetSlug] = useState<string | null>(null);
  const [unifiedStartMode, setUnifiedStartMode] = useState<string>("identical");
  const [unifiedCompanyCount, setUnifiedCompanyCount] = useState(4);
  const [aiCeoEnabled, setAiCeoEnabled] = useState(false);
  const [durationYears, setDurationYears] = useState(5);
  const [companyStrategies, setCompanyStrategies] = useState<Record<number, string>>({});

  const stageLabels = useMemo(() => {
    if (!unified.specDisplay?.stage_labels) return DEFAULT_STAGE_LABELS;
    const labels: Record<number, string> = {};
    for (const [k, v] of Object.entries(unified.specDisplay.stage_labels)) {
      labels[Number(k)] = v;
    }
    return labels;
  }, [unified.specDisplay]);

  const durationOptions = unified.specDisplay?.duration_options ?? [5, 10, 20];

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
    unified.start(
      unifiedStartMode,
      unifiedCompanyCount,
      aiCeoEnabled,
      durationYears,
      companyStrategies,
    );
  }, [unified, unifiedStartMode, unifiedCompanyCount, aiCeoEnabled, durationYears, companyStrategies]);

  const handleBack = useCallback(() => {
    setMode("unified-pick");
    setIndustrySlug(null);
    setPresetSlug(null);
  }, []);

  // ── Growth: Industry picker ──
  if (mode === "growth-pick") {
    return <IndustryPicker onSelect={handleSelectIndustry} onBack={handleBack} />;
  }

  // ── Market: Preset picker ──
  if (mode === "market-pick") {
    return <PresetPicker onSelect={handleSelectPreset} onBack={handleBack} />;
  }

  // ── Home: Unified Compete config (default) ──
  if (mode === "unified-pick") {
    const START_MODES = [
      { value: "identical", label: "Identical", desc: "All companies start with equal resources" },
      { value: "randomized", label: "Randomized", desc: "Varied starting cash and conditions" },
      { value: "staggered", label: "Staggered", desc: "Companies enter the market at different times" },
    ];
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
                max={20}
                value={unifiedCompanyCount}
                onChange={(e) => setUnifiedCompanyCount(Number(e.target.value))}
                className="w-full accent-accent"
              />
              <div className="flex justify-between text-[10px] text-surface-400 mt-1">
                <span>2</span>
                <span>20</span>
              </div>
            </div>

            <div>
              <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block">
                AI CEO Agents
              </label>
              <button
                onClick={() => setAiCeoEnabled(!aiCeoEnabled)}
                className={`w-full flex items-center justify-between p-3 rounded-lg border transition-all ${
                  aiCeoEnabled
                    ? "border-accent bg-accent/5 ring-1 ring-accent/30"
                    : "border-surface-200 hover:border-surface-300"
                }`}
              >
                <div className="text-left">
                  <div className="text-sm font-medium text-surface-800">
                    {aiCeoEnabled ? "Enabled" : "Disabled"}
                  </div>
                  <div className="text-[11px] text-surface-500">
                    Each company gets a Claude AI CEO making strategic decisions
                  </div>
                </div>
                <div
                  className={`w-10 h-6 rounded-full transition-all flex items-center ${
                    aiCeoEnabled ? "bg-accent justify-end" : "bg-surface-300 justify-start"
                  }`}
                >
                  <div className="w-4 h-4 bg-white rounded-full mx-1" />
                </div>
              </button>
            </div>

            {aiCeoEnabled && (
              <>
                <div>
                  <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block">
                    Duration
                  </label>
                  <div className="flex gap-2">
                    {durationOptions.map((y) => (
                      <button
                        key={y}
                        onClick={() => setDurationYears(y)}
                        className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${
                          durationYears === y
                            ? "border-accent bg-accent/5 text-accent ring-1 ring-accent/30"
                            : "border-surface-200 text-surface-600 hover:border-surface-300"
                        }`}
                      >
                        {y} Years
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block">
                    CEO Strategies
                  </label>
                  <div className="space-y-2 max-h-[240px] overflow-y-auto">
                    {Array.from({ length: unifiedCompanyCount }, (_, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <span className="text-xs text-surface-500 w-24 truncate flex-shrink-0">
                          {AGENT_NAMES[i]}
                        </span>
                        <select
                          value={companyStrategies[i] || "balanced"}
                          onChange={(e) =>
                            setCompanyStrategies((prev) => ({
                              ...prev,
                              [i]: e.target.value,
                            }))
                          }
                          className="flex-1 text-xs bg-surface-0 border border-surface-200 rounded-lg px-2 py-1.5 text-surface-700 focus:border-accent focus:ring-1 focus:ring-accent/30 outline-none"
                        >
                          {CEO_STRATEGIES.map((s) => (
                            <option key={s.value} value={s.value}>
                              {s.label}
                            </option>
                          ))}
                        </select>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            <button
              onClick={handleStartUnified}
              className="w-full py-3 rounded-xl bg-accent text-white font-semibold text-sm transition-all hover:bg-accent/90 hover:shadow-lg hover:shadow-accent/20"
            >
              Start Simulation
            </button>

            <button
              onClick={() => router.push("/profile")}
              className="w-full py-3 rounded-xl border border-surface-200 text-surface-700 font-semibold text-sm transition-all hover:border-accent hover:text-accent hover:bg-accent/5 flex items-center justify-center gap-2"
            >
              <Plus size={16} />
              Create Your Business
            </button>
          </div>
        </div>

        <div className="absolute bottom-4 right-5 flex items-center gap-4">
          <button
            onClick={() => setMode("market-pick")}
            className="text-xs text-surface-400 hover:text-surface-600 transition-colors"
          >
            Market Competition
          </button>
          <button
            onClick={() => setMode("growth-pick")}
            className="text-xs text-surface-400 hover:text-surface-600 transition-colors"
          >
            Company Growth
          </button>
        </div>
      </div>
    );
  }

  // ── Unified: Running simulation ──
  if (mode === "unified-sim") {
    return (
      <div className="h-screen flex flex-col bg-surface-0 phase-enter">
        <header className="relative flex items-center justify-between px-5 py-3 bg-surface-0/80 backdrop-blur-md border-b border-surface-200/50 z-[60]">
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
                durationYears={aiCeoEnabled ? durationYears : undefined}
              />
            )}
          </div>
        </header>

        <div className="flex-1 flex overflow-hidden relative">
          {unified.ceoThinking && (
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
              graph={unified.mergedGraph}
              multiCompany
              founderType={unified.founderType ?? undefined}
              onFocusCompany={unified.setFocusedCompany}
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
              reports={unified.reports}
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
