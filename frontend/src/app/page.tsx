"use client";

import { useCallback, useMemo, useState } from "react";
import { ModeSelector } from "@/components/modes/ModeSelector";
import { UnifiedSetup } from "@/components/modes/UnifiedSetup";
import { AdaptiveSetup } from "@/components/modes/AdaptiveSetup";
import { UnifiedSimView } from "@/components/modes/UnifiedSimView";
import { GrowthSimView } from "@/components/modes/GrowthSimView";
import { MarketSimView } from "@/components/modes/MarketSimView";
import { IndustryPicker } from "@/components/simulation/IndustryPicker";
import { PresetPicker } from "@/components/simulation/PresetPicker";
import { useSimulation } from "@/hooks/useSimulation";
import { useMarketSimulation } from "@/hooks/useMarketSimulation";
import { useUnifiedSimulation } from "@/hooks/useUnifiedSimulation";
import { useAdaptiveSetup } from "@/hooks/useAdaptiveSetup";

type AppMode =
  | "mode-select"
  | "unified-pick"
  | "adaptive-setup"
  | "adaptive-sim"
  | "growth-pick"
  | "growth-sim"
  | "market-pick"
  | "market-sim"
  | "unified-sim";

export default function Home() {
  const unified = useUnifiedSimulation();
  const sim = useSimulation(unified.specDisplay?.event_noise_filters ?? []);
  const market = useMarketSimulation();
  const adaptive = useAdaptiveSetup();

  const [mode, setMode] = useState<AppMode>("mode-select");
  const [industrySlug, setIndustrySlug] = useState<string | null>(null);
  const [presetSlug, setPresetSlug] = useState<string | null>(null);
  const [unifiedStartMode, setUnifiedStartMode] = useState("identical");
  const [unifiedCompanyCount, setUnifiedCompanyCount] = useState(4);
  const [aiCeoEnabled, setAiCeoEnabled] = useState(false);
  const [durationYears, setDurationYears] = useState(5);
  const [companyStrategies, setCompanyStrategies] = useState<Record<number, string>>({});

  // Stage labels come from the spec; missing entries fall back to "Stage N" in the consumer.
  const stageLabels = useMemo(() => {
    const labels: Record<number, string> = {};
    if (!unified.specDisplay?.stage_labels) return labels;
    for (const [k, v] of Object.entries(unified.specDisplay.stage_labels)) {
      labels[Number(k)] = v;
    }
    return labels;
  }, [unified.specDisplay]);

  const durationOptions = unified.specDisplay?.duration_options ?? [1, 5, 10, 20];

  const handleBack = useCallback(() => {
    setMode("mode-select");
    setIndustrySlug(null);
    setPresetSlug(null);
  }, []);

  const handleSelectIndustry = useCallback(
    (slug: string) => {
      setIndustrySlug(slug);
      setMode("growth-sim");
      sim.start(slug);
    },
    [sim],
  );

  const handleSelectPreset = useCallback(
    (slug: string) => {
      setPresetSlug(slug);
      setMode("market-sim");
      market.start(slug);
    },
    [market],
  );

  const handleStartUnified = useCallback(() => {
    setMode("unified-sim");
    unified.start(unifiedStartMode, unifiedCompanyCount, aiCeoEnabled, durationYears, companyStrategies);
  }, [unified, unifiedStartMode, unifiedCompanyCount, aiCeoEnabled, durationYears, companyStrategies]);

  // ── Route to active mode component ──

  if (mode === "growth-pick") {
    return <IndustryPicker onSelect={handleSelectIndustry} onBack={handleBack} />;
  }

  if (mode === "market-pick") {
    return <PresetPicker onSelect={handleSelectPreset} onBack={handleBack} />;
  }

  if (mode === "mode-select") {
    return (
      <ModeSelector
        onSelectUnified={() => setMode("unified-pick")}
        onSelectAdaptive={() => { adaptive.reset(); setMode("adaptive-setup"); }}
        onSelectMarket={() => setMode("market-pick")}
        onSelectGrowth={() => setMode("growth-pick")}
      />
    );
  }

  if (mode === "adaptive-setup") {
    return (
      <AdaptiveSetup
        adaptive={adaptive}
        onBack={() => setMode("mode-select")}
        onStartSim={() => setMode("adaptive-sim")}
      />
    );
  }

  if (mode === "adaptive-sim") {
    return (
      <UnifiedSimView
        subtitle="Custom Adaptive"
        sessionId={adaptive.sessionId}
        playing={adaptive.playing}
        speed={adaptive.speed}
        tick={adaptive.tick}
        isComplete={adaptive.isComplete}
        ceoThinking={adaptive.ceoThinking}
        tam={adaptive.tam}
        captured={adaptive.captured}
        hhi={adaptive.hhi}
        agentCount={adaptive.agentCount}
        agents={adaptive.agents}
        focusedCompanyId={adaptive.focusedCompanyId}
        mergedGraph={adaptive.mergedGraph}
        founderType={adaptive.founderType ?? undefined}
        status={adaptive.status}
        history={adaptive.history}
        eventLog={adaptive.eventLog}
        reports={adaptive.reports}
        durationYears={adaptive.aiCeoEnabled ? adaptive.durationYears : undefined}
        onPlay={adaptive.play}
        onPause={adaptive.pause}
        onSetSpeed={adaptive.setSpeed}
        onFocusCompany={adaptive.setFocusedCompany}
        onBack={() => { adaptive.reset(); setMode("mode-select"); }}
      />
    );
  }

  if (mode === "unified-pick") {
    return (
      <UnifiedSetup
        startMode={unifiedStartMode}
        companyCount={unifiedCompanyCount}
        aiCeoEnabled={aiCeoEnabled}
        durationYears={durationYears}
        durationOptions={durationOptions}
        companyStrategies={companyStrategies}
        onSetStartMode={setUnifiedStartMode}
        onSetCompanyCount={setUnifiedCompanyCount}
        onSetAiCeoEnabled={setAiCeoEnabled}
        onSetDurationYears={setDurationYears}
        onSetCompanyStrategies={setCompanyStrategies}
        onStart={handleStartUnified}
        onBack={() => setMode("mode-select")}
        onSelectMarket={() => setMode("market-pick")}
        onSelectGrowth={() => setMode("growth-pick")}
      />
    );
  }

  if (mode === "unified-sim") {
    return (
      <UnifiedSimView
        subtitle="Unified Compete"
        sessionId={unified.sessionId}
        playing={unified.playing}
        speed={unified.speed}
        tick={unified.tick}
        isComplete={unified.isComplete}
        ceoThinking={unified.ceoThinking}
        tam={unified.tam}
        captured={unified.captured}
        hhi={unified.hhi}
        agentCount={unified.agentCount}
        agents={unified.agents}
        focusedCompanyId={unified.focusedCompanyId}
        mergedGraph={unified.mergedGraph}
        founderType={unified.founderType ?? undefined}
        status={unified.status}
        history={unified.history}
        eventLog={unified.eventLog}
        reports={unified.reports}
        durationYears={aiCeoEnabled ? durationYears : undefined}
        onPlay={unified.play}
        onPause={unified.pause}
        onSetSpeed={unified.setSpeed}
        onFocusCompany={unified.setFocusedCompany}
        onBack={handleBack}
      />
    );
  }

  if (mode === "growth-sim") {
    return (
      <GrowthSimView
        sim={sim}
        industrySlug={industrySlug}
        stageLabels={stageLabels}
        onBack={handleBack}
      />
    );
  }

  // market-sim (default fallback)
  return (
    <MarketSimView
      market={market}
      presetSlug={presetSlug}
      onBack={handleBack}
    />
  );
}
