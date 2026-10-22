"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import type { GraphData, SimEdge, SimNode } from "@/types/graph";
import type {
  CEODecisionEvent,
  CEOReport,
  UnifiedAgentSnapshot,
  UnifiedTickData,
} from "@/types/unified";
import {
  type CompanyTickPayload,
  type GraphSnapshotV2,
  controlV2,
  createV2Stream,
  sampleSeed,
  sampleStance,
  startV2,
  type CompanySeed,
} from "@/lib/apiV2";

export interface SpecDisplay {
  stage_labels: Record<string, string>;
  event_noise_filters: string[];
  duration_options: number[];
}

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
    seedOverride?: CompanySeed,
  ) => Promise<void>;
  play: () => Promise<void>;
  pause: () => Promise<void>;
  setSpeed: (multiplier: number) => Promise<void>;
  setFocusedCompany: (companyId: string) => void;
}

const COMPANY_COLORS = [
  "#2563eb",
  "#dc2626",
  "#16a34a",
  "#9333ea",
  "#ea580c",
  "#0891b2",
  "#be123c",
  "#4f46e5",
  "#65a30d",
  "#c2410c",
];

const SPEC_DISPLAY: SpecDisplay = {
  stage_labels: { "1": "Operating" },
  event_noise_filters: [],
  duration_options: [1, 5, 10, 20],
};

const STRATEGY_TO_STANCE: Record<string, string> = {
  aggressive_growth: "venture_growth",
  quality_focus: "founder_operator",
  cost_leader: "bootstrap",
  balanced: "bootstrap",
  market_dominator: "consolidator",
  survivor: "turnaround",
};

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

function asGraphData(graph: GraphSnapshotV2): GraphData {
  return {
    nodes: graph.nodes.map((node) => ({
      ...node,
      category: node.category as SimNode["category"],
    })),
    edges: graph.edges,
  };
}

function countNodes(company: CompanyTickPayload, category: string): number {
  return company.graph.nodes
    .filter((node) => node.category === category)
    .reduce((sum, node) => sum + (node.metrics.count ?? 1), 0);
}

function toAgentSnapshot(
  company: CompanyTickPayload,
  share: number,
  index: number,
  strategy: string | null,
): UnifiedAgentSnapshot {
  const nodeCount = company.graph.nodes.reduce(
    (sum, node) => sum + (node.metrics.count ?? 1),
    0,
  );
  const locationCount = Math.max(1, countNodes(company, "location"));
  const marketingNodes = countNodes(company, "marketing");
  const qualityNodes =
    countNodes(company, "ops") +
    countNodes(company, "exec") +
    countNodes(company, "sales");

  return {
    id: company.company_id,
    name: `Company ${index + 1}`,
    alive: !company.bankrupt,
    color: COMPANY_COLORS[index % COMPANY_COLORS.length],
    quality: company.avg_satisfaction + qualityNodes * 0.05,
    marketing: 1 + marketingNodes * 0.1,
    capacity:
      company.capacity_utilization > 0
        ? company.daily_revenue / company.capacity_utilization
        : company.daily_revenue,
    share,
    utilization: company.capacity_utilization,
    binding_constraint: company.capacity_utilization >= 0.95 ? "capacity" : "demand",
    cash: company.cash,
    daily_revenue: company.daily_revenue,
    daily_costs: company.daily_costs,
    stage: 1,
    location_count: locationCount,
    node_count: nodeCount,
    avg_satisfaction: company.avg_satisfaction,
    total_employees: company.employee_count,
    strategy,
  };
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
  const [ceoDecisionLog] = useState<CEODecisionEvent[]>([]);
  const [reports, setReports] = useState<CEOReport[] | null>(null);
  const [specDisplay, setSpecDisplay] = useState<SpecDisplay | null>(SPEC_DISPLAY);
  const [founderType, setFounderType] = useState<string | null>("founder");
  const eventSourceRef = useRef<EventSource | null>(null);

  const mergedGraph = useMemo(() => {
    if (Object.keys(graphs).length === 0) return null;
    return mergeGraphs(graphs, agents);
  }, [graphs, agents]);

  const start = useCallback(
    async (
      startMode = "identical",
      numCompanies = 4,
      aiCeoEnabled = true,
      durationYears = 5,
      companyStrategies: Record<number, string> = {},
      seedOverride?: CompanySeed,
    ) => {
      void startMode;
      try {
        eventSourceRef.current?.close();
        setError(null);
        setTick(0);
        setTam(0);
        setCaptured(0);
        setHhi(0);
        setAgentCount(0);
        setAgents([]);
        setGraphs({});
        setHistory([]);
        setEventLog([]);
        setFocusedCompanyId("");
        setIsComplete(false);
        setStatus("operating");
        setSpecDisplay(SPEC_DISPLAY);
        setFounderType("founder");
        setReports(null);

        // Per-company stances honor companyStrategies — the previous code
        // sampled only strategy[0] and applied it to every company, which
        // made the per-company strategy UI a label-only feature. Stances
        // are sampled in parallel; seeds use the imported override when
        // provided, otherwise share a single `small_team` sample.
        const baseSeed = seedOverride ?? (await sampleSeed("small_team"));
        const stances = await Promise.all(
          Array.from({ length: numCompanies }, (_, i) =>
            sampleStance(
              STRATEGY_TO_STANCE[
                companyStrategies[i] ?? "balanced"
              ] ?? "bootstrap",
            ),
          ),
        );
        const resp = await startV2({
          seed: baseSeed,
          stance: stances[0],
          stances,
          num_companies: numCompanies,
          duration_ticks: Math.max(10, durationYears * 365),
          tam_initial: baseSeed.tam,
          ai_ceo_enabled: aiCeoEnabled,
        });

        setSessionId(resp.session_id);
        setPlaying(true);

        const es = createV2Stream(resp.session_id);
        eventSourceRef.current = es;

        es.onmessage = (event) => {
          let data: Record<string, unknown>;
          try {
            data = JSON.parse(event.data);
          } catch (e) {
            console.error("Failed to parse SSE message:", e);
            return;
          }

          if (data.type === "tick") {
            const companies = data.companies as CompanyTickPayload[];
            const shares = (data.shares as number[]) ?? [];
            const agentsNext = companies.map((company, index) =>
              toAgentSnapshot(
                company,
                shares[index] ?? 0,
                index,
                companyStrategies[index] ?? null,
              ),
            );
            const graphsNext = Object.fromEntries(
              companies.map((company) => [company.company_id, asGraphData(company.graph)]),
            );
            const events = companies.flatMap((company) =>
              company.decisions.map((decision) =>
                `${company.company_id}: ${decision.tier} decision - ${decision.reasoning}`,
              ),
            );
            const capturedNext = companies.reduce(
              (sum, company) => sum + company.daily_revenue,
              0,
            );
            const hhiNext = shares.reduce((sum, share) => sum + share * share, 0);
            const tickData: UnifiedTickData = {
              tick: data.tick as number,
              status: "operating",
              mode: "unified",
              tam: data.tam as number,
              captured: capturedNext,
              hhi: hhiNext,
              agent_count: agentsNext.length,
              agents: agentsNext,
              focused_company_id: agentsNext[0]?.id ?? "",
              graphs: graphsNext,
              events,
            };

            setTick(tickData.tick);
            setTam(tickData.tam);
            setCaptured(tickData.captured);
            setHhi(tickData.hhi);
            setAgentCount(tickData.agent_count);
            setAgents(tickData.agents);
            setFocusedCompanyId((prev) => prev || tickData.focused_company_id || "");
            setGraphs(tickData.graphs);
            const { graphs: _graphs, ...historyEntry } = tickData;
            setHistory((prev) => [...prev.slice(-499), historyEntry as UnifiedTickData]);
            if (events.length > 0) {
              setEventLog((prev) => [
                ...prev.slice(-499),
                ...events.map((e) => `Day ${data.tick}: ${e}`),
              ]);
            }
            return;
          }

          if (data.type === "reports") {
            const incoming = (data.reports as CEOReport[]) ?? [];
            // Only flip to non-null when there is at least one report — the
            // dashboard gates the "View CEO Reports" button on length > 0,
            // and an empty payload should not pop an empty modal.
            setReports(incoming.length > 0 ? incoming : null);
            return;
          }

          if (data.type === "complete") {
            setPlaying(false);
            setIsComplete(true);
            setStatus("complete");
            es.close();
            return;
          }

          if (data.type === "stopped") {
            setPlaying(false);
            setStatus("stopped");
            es.close();
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
    await controlV2(sessionId, "play");
    setPlaying(true);
  }, [sessionId]);

  const pause = useCallback(async () => {
    if (!sessionId) return;
    await controlV2(sessionId, "pause");
    setPlaying(false);
  }, [sessionId]);

  const setSpeed = useCallback(
    async (multiplier: number) => {
      if (!sessionId) return;
      await controlV2(sessionId, "set_speed", multiplier);
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
    ceoThinking: false,
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
