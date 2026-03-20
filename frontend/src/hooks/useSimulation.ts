"use client";

import { useState, useCallback, useRef } from "react";
import type { CompanyGraph, SimulationEvent } from "@/types/graph";
import {
  startSimulation,
  controlSimulation,
  injectEvent as apiInjectEvent,
  createSimulationStream,
} from "@/lib/api";

interface UseSimulationReturn {
  sessionId: string | null;
  tick: number;
  playing: boolean;
  speed: number;
  graph: CompanyGraph | null;
  previousGraph: CompanyGraph | null;
  globalMetrics: Record<string, number>;
  actions: SimulationEvent["actions"];
  outlook: string;
  start: (graph: CompanyGraph, selectedOutlook?: string) => Promise<void>;
  play: () => Promise<void>;
  pause: () => Promise<void>;
  setSpeed: (multiplier: number) => Promise<void>;
  setOutlook: (outlook: string) => Promise<void>;
  injectEvent: (description: string) => Promise<void>;
  isComplete: boolean;
}

export function useSimulation(): UseSimulationReturn {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeedState] = useState(1);
  const [graph, setGraph] = useState<CompanyGraph | null>(null);
  const [previousGraph, setPreviousGraph] = useState<CompanyGraph | null>(null);
  const [globalMetrics, setGlobalMetrics] = useState<Record<string, number>>({});
  const [actions, setActions] = useState<SimulationEvent["actions"]>(undefined);
  const [outlook, setOutlookState] = useState<string>("normal");
  const [isComplete, setIsComplete] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  const start = useCallback(async (initialGraph: CompanyGraph, selectedOutlook = "normal") => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    setOutlookState(selectedOutlook);

    const { session_id } = await startSimulation(initialGraph, 50, selectedOutlook);
    setSessionId(session_id);
    setTick(0);
    setIsComplete(false);
    setGraph(initialGraph);
    setPreviousGraph(null);
    setPlaying(true);

    const es = createSimulationStream(session_id);
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      const data: SimulationEvent = JSON.parse(event.data);
      switch (data.type) {
        case "tick":
          setGraph((prev) => {
            setPreviousGraph(prev);
            return data.graph ?? prev;
          });
          setTick(data.tick ?? 0);
          setGlobalMetrics(data.global_metrics ?? {});
          setActions(data.actions);
          break;
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
    [sessionId]
  );

  const setOutlook = useCallback(
    async (newOutlook: string) => {
      if (!sessionId) return;
      await controlSimulation(sessionId, "set_outlook", undefined, newOutlook);
      setOutlookState(newOutlook);
    },
    [sessionId]
  );

  const injectEvent = useCallback(
    async (description: string) => {
      if (!sessionId) return;
      await apiInjectEvent(sessionId, description);
    },
    [sessionId]
  );

  return {
    sessionId,
    tick,
    playing,
    speed,
    outlook,
    graph,
    previousGraph,
    globalMetrics,
    actions,
    start,
    play,
    pause,
    setSpeed,
    setOutlook,
    injectEvent,
    isComplete,
  };
}
