"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import type { GraphData, SimNode } from "@/types/graph";
import {
  type AiStatus,
  type CeoDecision,
  type CeoReport,
  type CompanyState,
  type StartRequest,
  type StreamEvent,
  type TickEvent,
  controlSimulation,
  openStream,
  startSimulation,
} from "@/lib/api";

export const COMPANY_COLORS = [
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

const HISTORY_LIMIT = 2000;
const FEED_LIMIT = 300;

export type SimStatus = "idle" | "running" | "paused" | "complete" | "stopped" | "error";

export interface HistoryPoint {
  tick: number;
  shares: Record<string, number>;
  cash: Record<string, number>;
}

export interface FeedItem {
  companyId: string;
  name: string;
  decision: CeoDecision;
}

function mergeGraphs(companies: CompanyState[]): GraphData {
  const nodes: SimNode[] = [];
  const edges: GraphData["edges"] = [];
  companies.forEach((c, i) => {
    for (const node of c.graph.nodes) {
      nodes.push({
        ...node,
        category: node.category as SimNode["category"],
        id: `${c.company_id}::${node.id}`,
        companyId: c.company_id,
        companyName: c.name,
        companyColor: COMPANY_COLORS[i % COMPANY_COLORS.length],
        alive: c.alive,
      });
    }
    for (const edge of c.graph.edges) {
      edges.push({
        ...edge,
        source: `${c.company_id}::${edge.source}`,
        target: `${c.company_id}::${edge.target}`,
      });
    }
  });
  return { nodes, edges };
}

/** Changes only when the graph's structure does, so the force layout isn't rebuilt every tick. */
function graphSignature(companies: CompanyState[]): string {
  return companies
    .map((c) => `${c.company_id}:${c.alive}:${c.graph.nodes.map((n) => n.label).join(",")}`)
    .join("|");
}

export function useSimulation() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState<SimStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [speed, setSpeedState] = useState(5);
  const [latest, setLatest] = useState<TickEvent | null>(null);
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [feed, setFeed] = useState<FeedItem[]>([]);
  const [events, setEvents] = useState<string[]>([]);
  const [reports, setReports] = useState<CeoReport[] | null>(null);
  const [thinking, setThinking] = useState(false);
  const [ai, setAi] = useState<AiStatus | null>(null);
  const streamRef = useRef<EventSource | null>(null);
  const signatureRef = useRef("");

  const handle = useCallback((data: StreamEvent) => {
    switch (data.type) {
      case "tick": {
        setThinking(false);
        setLatest(data);
        setAi(data.ai);
        const signature = graphSignature(data.companies);
        if (signature !== signatureRef.current) {
          signatureRef.current = signature;
          setGraph(mergeGraphs(data.companies));
        }
        setHistory((prev) => {
          const point: HistoryPoint = { tick: data.tick, shares: {}, cash: {} };
          for (const c of data.companies) {
            point.shares[c.company_id] = c.market_share;
            point.cash[c.company_id] = c.cash;
          }
          return [...prev.slice(-(HISTORY_LIMIT - 1)), point];
        });
        const fresh = data.companies.flatMap((c) =>
          c.decisions.map((decision) => ({ companyId: c.company_id, name: c.name, decision })),
        );
        if (fresh.length) setFeed((prev) => [...prev, ...fresh].slice(-FEED_LIMIT));
        if (data.events.length) {
          setEvents((prev) => [...prev, ...data.events].slice(-FEED_LIMIT));
        }
        return;
      }
      case "thinking":
        setThinking(true);
        return;
      case "reports":
        setReports(data.reports.length ? data.reports : null);
        return;
      case "complete":
        setThinking(false);
        setAi(data.ai);
        setStatus("complete");
        streamRef.current?.close();
        return;
      case "stopped":
        setStatus("stopped");
        streamRef.current?.close();
        return;
    }
  }, []);

  const start = useCallback(
    async (req: StartRequest) => {
      streamRef.current?.close();
      setError(null);
      setLatest(null);
      setGraph(null);
      setHistory([]);
      setFeed([]);
      setEvents([]);
      setReports(null);
      setThinking(false);
      setAi(null);
      signatureRef.current = "";
      try {
        const resp = await startSimulation(req);
        setSessionId(resp.session_id);
        await controlSimulation(resp.session_id, "set_speed", speed);
        setStatus("running");
        const es = openStream(resp.session_id);
        streamRef.current = es;
        es.onmessage = (event) => handle(JSON.parse(event.data) as StreamEvent);
        es.onerror = () => {
          es.close();
          setStatus((s) => (s === "running" || s === "paused" ? "error" : s));
          setError((e) => e ?? "Lost connection to the backend.");
        };
      } catch (e) {
        setStatus("error");
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [handle, speed],
  );

  const play = useCallback(async () => {
    if (!sessionId) return;
    await controlSimulation(sessionId, "play");
    setStatus("running");
  }, [sessionId]);

  const pause = useCallback(async () => {
    if (!sessionId) return;
    await controlSimulation(sessionId, "pause");
    setStatus("paused");
  }, [sessionId]);

  const stop = useCallback(async () => {
    streamRef.current?.close();
    if (sessionId && (status === "running" || status === "paused")) {
      await controlSimulation(sessionId, "stop").catch(() => undefined);
    }
    setStatus("idle");
  }, [sessionId, status]);

  const setSpeed = useCallback(
    async (multiplier: number) => {
      setSpeedState(multiplier);
      if (sessionId && (status === "running" || status === "paused")) {
        await controlSimulation(sessionId, "set_speed", multiplier);
      }
    },
    [sessionId, status],
  );

  const companies = useMemo(() => latest?.companies ?? [], [latest]);

  return {
    status,
    error,
    speed,
    latest,
    companies,
    graph,
    history,
    feed,
    events,
    reports,
    thinking,
    ai,
    start,
    play,
    pause,
    stop,
    setSpeed,
  };
}

export type SimulationState = ReturnType<typeof useSimulation>;
