"use client";

import { useState, useCallback, useRef } from "react";
import type { GraphData } from "@/types/graph";
import type { UnifiedAgentSnapshot, UnifiedTickData } from "@/types/unified";
import {
  startUnifiedSimulation,
  controlSimulation,
  createSimulationStream,
  focusCompany,
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
  graph: GraphData | null;
  status: string;
  history: UnifiedTickData[];
  eventLog: string[];
  isComplete: boolean;
  start: (startMode?: string, numCompanies?: number) => Promise<void>;
  play: () => Promise<void>;
  pause: () => Promise<void>;
  setSpeed: (multiplier: number) => Promise<void>;
  setFocusedCompany: (companyId: string) => Promise<void>;
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
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [status, setStatus] = useState("operating");
  const [history, setHistory] = useState<UnifiedTickData[]>([]);
  const [eventLog, setEventLog] = useState<string[]>([]);
  const [isComplete, setIsComplete] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  const start = useCallback(
    async (startMode = "identical", numCompanies = 4) => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      const { session_id } = await startUnifiedSimulation(
        startMode,
        numCompanies,
        0,
      );
      setSessionId(session_id);
      setTick(0);
      setIsComplete(false);
      setStatus("operating");
      setAgents([]);
      setGraph(null);
      setHistory([]);
      setEventLog([]);
      setPlaying(true);

      const es = createSimulationStream(session_id);
      eventSourceRef.current = es;

      es.onmessage = (event) => {
        const data = JSON.parse(event.data);
        switch (data.type) {
          case "tick": {
            if (data.mode !== "unified") break;
            const tickData: UnifiedTickData = {
              tick: data.tick,
              status: data.status,
              mode: "unified",
              tam: data.tam,
              captured: data.captured,
              hhi: data.hhi,
              agent_count: data.agent_count,
              agents: data.agents,
              focused_company_id: data.focused_company_id,
              graph: data.graph,
              events: data.events,
            };
            setTick(data.tick);
            setTam(data.tam);
            setCaptured(data.captured);
            setHhi(data.hhi);
            setAgentCount(data.agent_count);
            setAgents(data.agents);
            setFocusedCompanyId(data.focused_company_id);
            setStatus(data.status);
            if (data.graph) {
              setGraph(data.graph);
            }
            setHistory((prev) => [...prev, tickData]);
            if (data.events?.length > 0) {
              const significant = (data.events as string[]).filter(
                (e: string) =>
                  !e.includes("spoiled") &&
                  !e.includes("Ordered") &&
                  !e.includes("Turned away") &&
                  !e.includes("Cannot reorder"),
              );
              if (significant.length > 0) {
                setEventLog((prev) => [
                  ...prev,
                  ...significant.map((e: string) => `Day ${data.tick}: ${e}`),
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
        }
      };

      es.onerror = () => {
        setPlaying(false);
        es.close();
      };
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

  const setFocusedCompany = useCallback(
    async (companyId: string) => {
      if (!sessionId) return;
      setFocusedCompanyId(companyId);
      const res = await focusCompany(sessionId, companyId);
      if (res.graph) {
        setGraph(res.graph);
      }
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
    graph,
    status,
    history,
    eventLog,
    isComplete,
    start,
    play,
    pause,
    setSpeed,
    setFocusedCompany,
  };
}
