"use client";

import { useState, useCallback } from "react";
import { forkScenario, compareScenarios } from "@/lib/api";

interface Scenario {
  sessionId: string;
  label: string;
  forkedFrom?: string;
}

interface UseScenarioReturn {
  scenarios: Scenario[];
  activeScenarioId: string | null;
  fork: (sessionId: string) => Promise<string>;
  setActive: (id: string) => void;
  comparison: Record<string, unknown> | null;
  compare: () => Promise<void>;
}

export function useScenarios(): UseScenarioReturn {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [activeScenarioId, setActiveScenarioId] = useState<string | null>(null);
  const [comparison, setComparison] = useState<Record<string, unknown> | null>(null);

  const fork = useCallback(async (sessionId: string) => {
    const { session_id } = await forkScenario(sessionId);
    const newScenario: Scenario = {
      sessionId: session_id,
      label: `Scenario ${scenarios.length + 1}`,
      forkedFrom: sessionId,
    };
    setScenarios((prev) => [...prev, newScenario]);
    setActiveScenarioId(session_id);
    return session_id;
  }, [scenarios.length]);

  const compare = useCallback(async () => {
    if (scenarios.length < 2) return;
    const ids = scenarios.map((s) => s.sessionId);
    const result = await compareScenarios(ids);
    setComparison(result.scenarios as Record<string, unknown>);
  }, [scenarios]);

  return {
    scenarios,
    activeScenarioId,
    fork,
    setActive: setActiveScenarioId,
    comparison,
    compare,
  };
}
