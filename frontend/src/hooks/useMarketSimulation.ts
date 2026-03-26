"use client";

import { useState, useCallback, useRef } from "react";
import type { AgentSnapshot, MarketTickData } from "@/types/market";
import {
  startMarketSimulation,
  controlSimulation,
  createSimulationStream,
} from "@/lib/api";

export interface UseMarketSimulationReturn {
  sessionId: string | null;
  tick: number;
  playing: boolean;
  speed: number;
  tam: number;
  captured: number;
  hhi: number;
  agentCount: number;
  agents: AgentSnapshot[];
  status: string;
  history: MarketTickData[];
  eventLog: string[];
  isComplete: boolean;
  start: (preset: string) => Promise<void>;
  play: () => Promise<void>;
  pause: () => Promise<void>;
  setSpeed: (multiplier: number) => Promise<void>;
}

export function useMarketSimulation(): UseMarketSimulationReturn {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeedState] = useState(1);
  const [tam, setTam] = useState(0);
  const [captured, setCaptured] = useState(0);
  const [hhi, setHhi] = useState(0);
  const [agentCount, setAgentCount] = useState(0);
  const [agents, setAgents] = useState<AgentSnapshot[]>([]);
  const [status, setStatus] = useState("operating");
  const [history, setHistory] = useState<MarketTickData[]>([]);
  const [eventLog, setEventLog] = useState<string[]>([]);
  const [isComplete, setIsComplete] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  const start = useCallback(async (preset: string) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const { session_id } = await startMarketSimulation(preset, 0);
    setSessionId(session_id);
    setTick(0);
    setIsComplete(false);
    setStatus("operating");
    setAgents([]);
    setHistory([]);
    setEventLog([]);
    setPlaying(true);

    const es = createSimulationStream(session_id);
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      const data = JSON.parse(event.data);
      switch (data.type) {
        case "tick": {
          if (data.mode !== "market") break;
          const tickData: MarketTickData = {
            tick: data.tick,
            tam: data.tam,
            captured: data.captured,
            hhi: data.hhi,
            agent_count: data.agent_count,
            agents: data.agents,
            events: data.events,
            status: data.status,
          };
          setTick(data.tick);
          setTam(data.tam);
          setCaptured(data.captured);
          setHhi(data.hhi);
          setAgentCount(data.agent_count);
          setAgents(data.agents);
          setStatus(data.status);
          setHistory((prev) => [...prev, tickData]);
          if (data.events?.length > 0) {
            setEventLog((prev) => [
              ...prev,
              ...data.events.map((e: string) => `Day ${data.tick}: ${e}`),
            ]);
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
  }, []);

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
    status,
    history,
    eventLog,
    isComplete,
    start,
    play,
    pause,
    setSpeed,
  };
}
