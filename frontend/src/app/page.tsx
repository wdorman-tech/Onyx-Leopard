"use client";

import { useCallback, useMemo, useState } from "react";
import { ModeSelector } from "@/components/modes/ModeSelector";
import { UnifiedSetup } from "@/components/modes/UnifiedSetup";
import { UnifiedSimView } from "@/components/modes/UnifiedSimView";
import { useUnifiedSimulation } from "@/hooks/useUnifiedSimulation";
import type { ImportedCompany } from "@/lib/apiV2";

type AppMode = "mode-select" | "unified-pick" | "unified-sim";

export default function Home() {
  const unified = useUnifiedSimulation();

  const [mode, setMode] = useState<AppMode>("mode-select");
  const [unifiedStartMode, setUnifiedStartMode] = useState("identical");
  const [unifiedCompanyCount, setUnifiedCompanyCount] = useState(4);
  const [aiCeoEnabled, setAiCeoEnabled] = useState(true);
  const [durationYears, setDurationYears] = useState(5);
  const [companyStrategies, setCompanyStrategies] = useState<Record<number, string>>({});
  const [selectedCompany, setSelectedCompany] = useState<ImportedCompany | null>(null);

  const stageLabels = useMemo<Record<number, string>>(() => {
    const labels: Record<number, string> = {};
    if (!unified.specDisplay?.stage_labels) return labels;
    for (const [k, v] of Object.entries(unified.specDisplay.stage_labels)) {
      labels[Number(k)] = v;
    }
    return labels;
  }, [unified.specDisplay]);
  void stageLabels;

  const durationOptions = unified.specDisplay?.duration_options ?? [1, 5, 10, 20];

  const handleBack = useCallback(() => {
    setMode("mode-select");
  }, []);

  const handleStartUnified = useCallback(() => {
    setMode("unified-sim");
    unified.start(
      unifiedStartMode,
      unifiedCompanyCount,
      aiCeoEnabled,
      durationYears,
      companyStrategies,
      selectedCompany?.seed,
    );
  }, [
    unified,
    unifiedStartMode,
    unifiedCompanyCount,
    aiCeoEnabled,
    durationYears,
    companyStrategies,
    selectedCompany,
  ]);

  if (mode === "mode-select") {
    return <ModeSelector onSelectUnified={() => setMode("unified-pick")} />;
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
        selectedCompany={selectedCompany}
        onSetStartMode={setUnifiedStartMode}
        onSetCompanyCount={setUnifiedCompanyCount}
        onSetAiCeoEnabled={setAiCeoEnabled}
        onSetDurationYears={setDurationYears}
        onSetCompanyStrategies={setCompanyStrategies}
        onSetSelectedCompany={setSelectedCompany}
        onStart={handleStartUnified}
        onBack={() => setMode("mode-select")}
      />
    );
  }

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
