"use client";

import { useState, useCallback, useRef, useMemo } from "react";
import type { GraphData, SimNode, SimEdge } from "@/types/graph";
import type {
  CEODecisionEvent,
  CEOReport,
  UnifiedAgentSnapshot,
  UnifiedTickData,
} from "@/types/unified";
import type { SpecDisplay } from "@/lib/api";
import {
  startUnifiedSimulation,
  controlSimulation,
  createSimulationStream,
} from "@/lib/api";

export interface UseUnifiedSimulationReturn {
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
  error: string | null;
  ceoThinking: boolean;
  ceoDecisionLog: CEODecisionEvent[];
  reports: CEOReport[] | null;
  specDisplay: SpecDisplay | null;
  founderType: string | null;
  start: (
    startMode?: string,
    numCompanies?: number,
    aiCeoEnabled?: boolean,
    durationYears?: number,
    companyStrategies?: Record<number, string>,
  ) => Promise<void>;
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

export function useUnifiedSimulation(): UseUnifiedSimulationReturn {
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
  const [error, setError] = useState<string | null>(null);
  const [ceoThinking, setCeoThinking] = useState(false);
  const [ceoDecisionLog, setCeoDecisionLog] = useState<CEODecisionEvent[]>([]);
  const [reports, setReports] = useState<CEOReport[] | null>(null);
  const [specDisplay, setSpecDisplay] = useState<SpecDisplay | null>(null);
  const [founderType, setFounderType] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const mergedGraph = useMemo(() => {
    if (Object.keys(graphs).length === 0) return null;
    return mergeGraphs(graphs, agents);
  }, [graphs, agents]);

  const start = useCallback(
    async (
      startMode = "identical",
      numCompanies = 4,
      aiCeoEnabled = false,
      durationYears = 5,
      companyStrategies: Record<number, string> = {},
    ) => {
      try {
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
        }

        setError(null);
        setCeoThinking(false);
        setCeoDecisionLog([]);
        setReports(null);

        const maxTicks = aiCeoEnabled ? durationYears * 365 : 0;
        const resp = await startUnifiedSimulation(
          startMode,
          numCompanies,
          maxTicks,
          aiCeoEnabled,
          durationYears,
          companyStrategies,
        );
        setSessionId(resp.session_id);
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

        const noiseFilters = resp.spec_display?.event_noise_filters ?? [
          "spoiled", "Ordered", "Turned away", "Cannot reorder",
        ];

        const es = createSimulationStream(resp.session_id);
        eventSourceRef.current = es;

        es.onmessage = (event) => {
          let data: Record<string, unknown>;
          try {
            data = JSON.parse(event.data);
          } catch (e) {
            console.error("Failed to parse SSE message:", e);
            return;
          }
          switch (data.type) {
            case "tick": {
              if (data.mode !== "unified") break;
              const tickData: UnifiedTickData = {
                tick: data.tick as number,
                status: data.status as string,
                mode: "unified",
                tam: data.tam as number,
                captured: data.captured as number,
                hhi: data.hhi as number,
                agent_count: data.agent_count as number,
                agents: data.agents as UnifiedAgentSnapshot[],
                focused_company_id: data.focused_company_id as string,
                graphs: data.graphs as Record<string, GraphData>,
                events: data.events as string[],
              };
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
                setGraphs(tickData.graphs);
              }
              // Strip graphs from history entries to avoid unbounded memory growth
              const { graphs: _g, ...historyEntry } = tickData;
              setHistory((prev) => [...prev.slice(-499), historyEntry as UnifiedTickData]);
              if ((data.events as string[])?.length > 0) {
                const significant = (data.events as string[]).filter(
                  (e: string) => !noiseFilters.some((f) => e.includes(f)),
                );
                if (significant.length > 0) {
                  setEventLog((prev) => [
                    ...prev.slice(-499),
                    ...significant.map(
                      (e: string) => `Day ${data.tick}: ${e}`,
                    ),
                  ]);
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
                setCeoDecisionLog((prev) => [...prev, ...decisions]);
                setEventLog((prev) => [
                  ...prev,
                  ...decisions.map(
                    (d) =>
                      `Year ${d.sim_year} — ${d.company_name} (${d.strategy}): ` +
                      `price=$${d.decision.price_adjustment}, ` +
                      `expansion=${d.decision.expansion_pace}, ` +
                      `marketing=${d.decision.marketing_intensity.toFixed(1)} — ` +
                      `"${d.decision.reasoning}"`,
                  ),
                ]);
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

        es.onerror = (e) => {
          console.error("SSE connection error:", e);
          setError("Connection lost");
          setPlaying(false);
          es.close();
        };
      } catch (e) {
        console.error("Failed to start simulation:", e);
        setError(e instanceof Error ? e.message : "Failed to start simulation");
        setPlaying(false);
      }
    },
    [],
  );

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
    error,
    ceoThinking,
    ceoDecisionLog,
    reports,
    specDisplay,
    founderType,
    start,
    play,
    pause,
    setSpeed,
    setFocusedCompany: setFocusedCompanyId,
  };
}
