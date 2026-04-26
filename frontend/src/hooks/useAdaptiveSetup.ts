"use client";

import { useCallback, useMemo, useRef, useState } from "react";

const MAX_LOG_ENTRIES = 500;
import type { NicheAnalysis, SpecDisplay } from "@/lib/api";
import {
  analyzeNiche,
  startAdaptiveSimulation,
  controlSimulation,
  createSimulationStream,
} from "@/lib/api";
import type { GraphData, SimNode, SimEdge } from "@/types/graph";
import type {
  CEODecisionEvent,
  CEOReport,
  UnifiedAgentSnapshot,
  UnifiedTickData,
} from "@/types/unified";

export type AdaptiveStep = 1 | 2 | 3;

export interface UseAdaptiveSetupReturn {
  // Setup state
  step: AdaptiveStep;
  companyName: string;
  companyDescription: string;
  nicheAnalysis: NicheAnalysis | null;
  isAnalyzing: boolean;
  error: string | null;

  // Setup actions
  setCompanyName: (name: string) => void;
  setCompanyDescription: (desc: string) => void;
  analyze: () => Promise<void>;
  confirmNiche: () => void;
  backToStep: (step: AdaptiveStep) => void;
  reset: () => void;

  // Simulation config (step 3)
  numCompetitors: number;
  setNumCompetitors: (n: number) => void;
  startMode: string;
  setStartMode: (m: string) => void;
  durationYears: number;
  setDurationYears: (y: number) => void;
  aiCeoEnabled: boolean;
  setAiCeoEnabled: (enabled: boolean) => void;
  companyStrategies: Record<number, string>;
  setCompanyStrategies: React.Dispatch<
    React.SetStateAction<Record<number, string>>
  >;
  competitorNames: string[];

  // Simulation state (after start)
  isGenerating: boolean;
  sessionId: string | null;
  tick: number;
  playing: boolean;
  speed: number;
  tam: number;
  captured: number;
  hhi: number;
  agentCount: number;
  agents: UnifiedAgentSnapshot[];
  focusedCompanyId: string;
  mergedGraph: GraphData | null;
  status: string;
  history: UnifiedTickData[];
  eventLog: string[];
  isComplete: boolean;
  ceoThinking: boolean;
  ceoDecisionLog: CEODecisionEvent[];
  reports: CEOReport[] | null;
  specDisplay: SpecDisplay | null;
  founderType: string | null;

  // Simulation actions
  startSimulation: () => Promise<boolean>;
  play: () => Promise<void>;
  pause: () => Promise<void>;
  setSpeed: (multiplier: number) => Promise<void>;
  setFocusedCompany: (companyId: string) => void;
}

function mergeGraphs(
  graphs: Record<string, GraphData>,
  agents: UnifiedAgentSnapshot[],
): GraphData {
  const allNodes: SimNode[] = [];
  const allEdges: SimEdge[] = [];
  const agentMap = new Map(agents.map((a) => [a.id, a]));

  for (const [companyId, graph] of Object.entries(graphs)) {
    const agent = agentMap.get(companyId);
    const color = agent?.color ?? "#94a3b8";
    const alive = agent?.alive ?? true;

    for (const node of graph.nodes) {
      allNodes.push({
        ...node,
        id: `${companyId}::${node.id}`,
        companyId,
        companyColor: color,
        alive,
      });
    }

    for (const edge of graph.edges) {
      allEdges.push({
        ...edge,
        source: `${companyId}::${edge.source}`,
        target: `${companyId}::${edge.target}`,
      });
    }
  }

  return { nodes: allNodes, edges: allEdges };
}

export function useAdaptiveSetup(): UseAdaptiveSetupReturn {
  // ── Setup state ──
  const [step, setStep] = useState<AdaptiveStep>(1);
  const [companyName, setCompanyName] = useState("");
  const [companyDescription, setCompanyDescription] = useState("");
  const [nicheAnalysis, setNicheAnalysis] = useState<NicheAnalysis | null>(
    null,
  );
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ── Config state (step 3) ──
  const [numCompetitors, setNumCompetitors] = useState(4);
  const [startMode, setStartMode] = useState("identical");
  const [durationYears, setDurationYears] = useState(5);
  const [aiCeoEnabled, setAiCeoEnabled] = useState(true);
  const [companyStrategies, setCompanyStrategies] = useState<
    Record<number, string>
  >({});
  const [competitorNames, setCompetitorNames] = useState<string[]>([]);

  // ── Simulation state ──
  const [isGenerating, setIsGenerating] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeedState] = useState(1);
  const [tam, setTam] = useState(0);
  const [captured, setCaptured] = useState(0);
  const [hhi, setHhi] = useState(0);
  const [agentCount, setAgentCount] = useState(0);
  const [agents, setAgents] = useState<UnifiedAgentSnapshot[]>([]);
  const [focusedCompanyId, setFocusedCompanyId] = useState("");
  const [graphs, setGraphs] = useState<Record<string, GraphData>>({});
  const [status, setStatus] = useState("operating");
  const [history, setHistory] = useState<UnifiedTickData[]>([]);
  const [eventLog, setEventLog] = useState<string[]>([]);
  const [isComplete, setIsComplete] = useState(false);
  const [ceoThinking, setCeoThinking] = useState(false);
  const [ceoDecisionLog, setCeoDecisionLog] = useState<CEODecisionEvent[]>([]);
  const [reports, setReports] = useState<CEOReport[] | null>(null);
  const [specDisplay, setSpecDisplay] = useState<SpecDisplay | null>(null);
  const [founderType, setFounderType] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const mergedGraph = useMemo(
    () =>
      Object.keys(graphs).length > 0 ? mergeGraphs(graphs, agents) : null,
    [graphs, agents],
  );

  // ── Setup actions ──

  const analyze = useCallback(async () => {
    if (!companyName.trim() || !companyDescription.trim()) return;
    setIsAnalyzing(true);
    setError(null);
    try {
      const result = await analyzeNiche(companyName, companyDescription);
      setNicheAnalysis(result);
      setStep(2);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Failed to analyze your business",
      );
    } finally {
      setIsAnalyzing(false);
    }
  }, [companyName, companyDescription]);

  const confirmNiche = useCallback(() => {
    setStep(3);
  }, []);

  const backToStep = useCallback((s: AdaptiveStep) => {
    setStep(s);
    setError(null);
  }, []);

  const reset = useCallback(() => {
    // Always close the SSE connection first so it cannot keep writing into
    // state that we're about to clear.
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    // Setup state
    setStep(1);
    setCompanyName("");
    setCompanyDescription("");
    setNicheAnalysis(null);
    setIsAnalyzing(false);
    setError(null);

    // Config state
    setNumCompetitors(4);
    setStartMode("identical");
    setDurationYears(5);
    setAiCeoEnabled(true);
    setCompanyStrategies({});
    setCompetitorNames([]);

    // Simulation state — without these resets, an old run's data bleeds into
    // a new adaptive sim until the first tick overwrites it.
    setIsGenerating(false);
    setSessionId(null);
    setTick(0);
    setPlaying(false);
    setSpeedState(1);
    setTam(0);
    setCaptured(0);
    setHhi(0);
    setAgentCount(0);
    setAgents([]);
    setFocusedCompanyId("");
    setGraphs({});
    setStatus("operating");
    setHistory([]);
    setEventLog([]);
    setIsComplete(false);
    setCeoThinking(false);
    setCeoDecisionLog([]);
    setReports(null);
    setSpecDisplay(null);
    setFounderType(null);
  }, []);

  // ── Simulation start ──

  const startSimulation = useCallback(async (): Promise<boolean> => {
    if (!nicheAnalysis) return false;
    setIsGenerating(true);
    setError(null);

    try {
      const resp = await startAdaptiveSimulation({
        companyName,
        nicheDescription: nicheAnalysis.niche,
        nicheSummary: nicheAnalysis.summary,
        fullDescription: companyDescription,
        economicsModel: nicheAnalysis.economics_model,
        numCompetitors,
        startMode,
        durationYears,
        aiCeoEnabled,
        companyStrategies,
      });

      setSessionId(resp.session_id);
      setCompetitorNames(resp.competitor_names ?? []);
      if (resp.spec_display) setSpecDisplay(resp.spec_display);
      if (resp.founder_type) setFounderType(resp.founder_type);
      setTick(0);
      setIsComplete(false);
      setStatus("operating");
      setAgents([]);
      setGraphs({});
      setHistory([]);
      setEventLog([]);
      setPlaying(true);
      setCeoThinking(false);
      setCeoDecisionLog([]);
      setReports(null);
      setIsGenerating(false);

      const noiseFilters = resp.spec_display?.event_noise_filters ?? [];

      const es = createSimulationStream(resp.session_id);
      eventSourceRef.current = es;

      es.onmessage = (event) => {
        let data: Record<string, unknown>;
        try {
          data = JSON.parse(event.data);
        } catch {
          return;
        }
        switch (data.type) {
          case "tick": {
            if (data.mode !== "unified") break;
            const tickData = data as unknown as UnifiedTickData;
            setTick(tickData.tick);
            setTam(tickData.tam);
            setCaptured(tickData.captured);
            setHhi(tickData.hhi);
            setAgentCount(tickData.agent_count);
            setAgents(tickData.agents);
            setFocusedCompanyId((prev) =>
              prev || tickData.focused_company_id || "",
            );
            setStatus(tickData.status);
            if (tickData.graphs) {
              setGraphs(
                tickData.graphs as unknown as Record<string, GraphData>,
              );
            }
            const { graphs: _g, ...historyEntry } = tickData;
            setHistory((prev) => {
              const next = [...prev, historyEntry as UnifiedTickData];
              return next.length > MAX_LOG_ENTRIES
                ? next.slice(-MAX_LOG_ENTRIES)
                : next;
            });
            if ((data.events as string[])?.length > 0) {
              const significant = (data.events as string[]).filter(
                (e: string) => !noiseFilters.some((f) => e.includes(f)),
              );
              if (significant.length > 0) {
                setEventLog((prev) => {
                  const next = [
                    ...prev,
                    ...significant.map(
                      (e: string) => `Day ${data.tick}: ${e}`,
                    ),
                  ];
                  return next.length > MAX_LOG_ENTRIES
                    ? next.slice(-MAX_LOG_ENTRIES)
                    : next;
                });
              }
            }
            break;
          }
          case "complete":
            setPlaying(false);
            setIsComplete(true);
            es.close();
            break;
          case "stopped":
            setPlaying(false);
            es.close();
            break;
          case "ceo_thinking":
            setCeoThinking(true);
            break;
          case "ceo_decisions": {
            setCeoThinking(false);
            const decisions = data.decisions as CEODecisionEvent[];
            if (decisions?.length > 0) {
              setCeoDecisionLog((prev) => {
                const next = [...prev, ...decisions];
                return next.length > MAX_LOG_ENTRIES
                  ? next.slice(-MAX_LOG_ENTRIES)
                  : next;
              });
              setEventLog((prev) => {
                const next = [
                  ...prev,
                  ...decisions.map(
                    (d) =>
                      `Year ${d.sim_year} — ${d.company_name} (${d.strategy}): ` +
                      `price=$${d.decision.price_adjustment}, ` +
                      `expansion=${d.decision.expansion_pace}, ` +
                      `marketing=${d.decision.marketing_intensity.toFixed(1)} — ` +
                      `"${d.decision.reasoning}"`,
                  ),
                ];
                return next.length > MAX_LOG_ENTRIES
                  ? next.slice(-MAX_LOG_ENTRIES)
                  : next;
              });
            }
            break;
          }
          case "generating_reports":
            setCeoThinking(true);
            break;
          case "reports":
            setCeoThinking(false);
            setReports(data.reports as CEOReport[]);
            break;
        }
      };

      es.onerror = () => {
        setError("Connection lost");
        setPlaying(false);
        es.close();
      };

      return true;
    } catch (e) {
      setIsGenerating(false);
      setError(
        e instanceof Error ? e.message : "Failed to start adaptive simulation",
      );
      return false;
    }
  }, [
    nicheAnalysis,
    companyName,
    companyDescription,
    numCompetitors,
    startMode,
    durationYears,
    aiCeoEnabled,
    companyStrategies,
  ]);

  // ── Simulation controls ──

  const play = useCallback(async () => {
    if (!sessionId) return;
    await controlSimulation(sessionId, "play");
    setPlaying(true);
  }, [sessionId]);

  const pause = useCallback(async () => {
    if (!sessionId) return;
    await controlSimulation(sessionId, "pause");
    setPlaying(false);
  }, [sessionId]);

  const setSpeed = useCallback(
    async (multiplier: number) => {
      if (!sessionId) return;
      await controlSimulation(sessionId, "set_speed", multiplier);
      setSpeedState(multiplier);
    },
    [sessionId],
  );

  return {
    // Setup
    step,
    companyName,
    companyDescription,
    nicheAnalysis,
    isAnalyzing,
    error,
    setCompanyName,
    setCompanyDescription,
    analyze,
    confirmNiche,
    backToStep,
    reset,

    // Config
    numCompetitors,
    setNumCompetitors,
    startMode,
    setStartMode,
    durationYears,
    setDurationYears,
    aiCeoEnabled,
    setAiCeoEnabled,
    companyStrategies,
    setCompanyStrategies,
    competitorNames,

    // Simulation
    isGenerating,
    sessionId,
    tick,
    playing,
    speed,
    tam,
    captured,
    hhi,
    agentCount,
    agents,
    focusedCompanyId,
    mergedGraph,
    status,
    history,
    eventLog,
    isComplete,
    ceoThinking,
    ceoDecisionLog,
    reports,
    specDisplay,
    founderType,
    startSimulation,
    play,
    pause,
    setSpeed,
    setFocusedCompany: setFocusedCompanyId,
  };
}
