"use client";

import { useState, useCallback, useRef } from "react";
import type { GraphData, TickData } from "@/types/graph";
import {
  startSimulation,
  controlSimulation,
  createSimulationStream,
} from "@/lib/api";

const MAX_RECONNECT_ATTEMPTS = 3;
const RECONNECT_BACKOFF_MS = [1000, 2000, 4000];

export interface UseSimulationReturn {
  sessionId: string | null;
  tick: number;
  stage: number;
  playing: boolean;
  speed: number;
  metrics: Record<string, number>;
  status: string;
  graph: GraphData | null;
  metricsHistory: TickData[];
  eventLog: string[];
  isComplete: boolean;
  connectionError: string | null;
  start: (industry?: string) => Promise<void>;
  play: () => Promise<void>;
  pause: () => Promise<void>;
  setSpeed: (multiplier: number) => Promise<void>;
}

export function useSimulation(
  eventNoiseFilters: string[] = [],
): UseSimulationReturn {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const [stage, setStage] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeedState] = useState(1);
  const [metrics, setMetrics] = useState<Record<string, number>>({});
  const [status, setStatus] = useState("operating");
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<TickData[]>([]);
  const [eventLog, setEventLog] = useState<string[]>([]);
  const [isComplete, setIsComplete] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Ref so the SSE callback always sees the latest filter list without needing
  // to recreate the EventSource when filters change.
  const filtersRef = useRef<string[]>(eventNoiseFilters);
  filtersRef.current = eventNoiseFilters;

  const attachStream = useCallback((sessionIdToStream: string) => {
    const es = createSimulationStream(sessionIdToStream);
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // Successful message → reset reconnect counter so transient drops
      // don't burn through retries permanently.
      reconnectAttemptsRef.current = 0;
      switch (data.type) {
        case "tick": {
          const tickData: TickData = {
            tick: data.tick,
            stage: data.stage,
            metrics: data.metrics,
            status: data.status,
            events: data.events,
            graph: data.graph,
          };
          setTick(data.tick);
          setStage(data.stage ?? 1);
          setMetrics(data.metrics);
          setStatus(data.status);
          if (data.graph) {
            setGraph(data.graph);
          }
          setMetricsHistory((prev) => [...prev, tickData]);
          if (data.events?.length > 0) {
            // Drop events matching any industry-declared noise filter substring.
            // Empty filter list = pass through everything.
            const filters = filtersRef.current;
            const significant =
              filters.length === 0
                ? (data.events as string[])
                : (data.events as string[]).filter(
                    (e) => !filters.some((f) => e.includes(f)),
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
      es.close();
      const attempt = reconnectAttemptsRef.current;
      if (attempt >= MAX_RECONNECT_ATTEMPTS) {
        setConnectionError(
          "Lost connection to simulation stream after multiple attempts.",
        );
        setPlaying(false);
        return;
      }
      const delay = RECONNECT_BACKOFF_MS[attempt];
      reconnectAttemptsRef.current = attempt + 1;
      setConnectionError(
        `Connection dropped — reconnecting (attempt ${attempt + 1}/${MAX_RECONNECT_ATTEMPTS})…`,
      );
      reconnectTimerRef.current = setTimeout(() => {
        attachStream(sessionIdToStream);
      }, delay);
    };
  }, []);

  const start = useCallback(
    async (industry = "restaurant") => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      reconnectAttemptsRef.current = 0;
      setConnectionError(null);

      const { session_id } = await startSimulation(0, industry);
      setSessionId(session_id);
      setTick(0);
      setStage(1);
      setIsComplete(false);
      setMetrics({});
      setStatus("operating");
      setGraph(null);
      setMetricsHistory([]);
      setEventLog([]);
      setPlaying(true);

      attachStream(session_id);
    },
    [attachStream],
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
    stage,
    playing,
    speed,
    metrics,
    status,
    graph,
    metricsHistory,
    eventLog,
    isComplete,
    connectionError,
    start,
    play,
    pause,
    setSpeed,
  };
}
