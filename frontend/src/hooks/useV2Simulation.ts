"use client";

// V2 simulation hook — minimal wrapper around the v2 API + SSE stream.
// Exposes the essentials needed to drive a single-page sim view; richer
// per-feature UIs (CEO decision panels, shock timelines, multi-company
// graphs) can compose on top of this.

import { useCallback, useEffect, useRef, useState } from "react";

import {
  CompanySeed,
  CompanyTickPayload,
  CeoStance,
  ControlAction,
  StartV2Request,
  V2StreamEvent,
  controlV2,
  createV2Stream,
  startV2,
} from "@/lib/apiV2";

export interface UseV2SimulationState {
  sessionId: string | null;
  tick: number;
  tam: number;
  alive: number;
  shares: number[];
  companies: CompanyTickPayload[];
  isComplete: boolean;
  isStopped: boolean;
  isPaused: boolean;
  speed: number;
  error: string | null;
}

export interface UseV2SimulationApi extends UseV2SimulationState {
  start: (req: StartV2Request) => Promise<void>;
  play: () => Promise<void>;
  pause: () => Promise<void>;
  stop: () => Promise<void>;
  setSpeed: (speed: number) => Promise<void>;
  reset: () => void;
}

const INITIAL_STATE: UseV2SimulationState = {
  sessionId: null,
  tick: 0,
  tam: 0,
  alive: 0,
  shares: [],
  companies: [],
  isComplete: false,
  isStopped: false,
  isPaused: false,
  speed: 0.05,
  error: null,
};

export function useV2Simulation(): UseV2SimulationApi {
  const [state, setState] = useState<UseV2SimulationState>(INITIAL_STATE);
  const eventSourceRef = useRef<EventSource | null>(null);

  const cleanupStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  useEffect(() => () => cleanupStream(), [cleanupStream]);

  const reset = useCallback(() => {
    cleanupStream();
    setState(INITIAL_STATE);
  }, [cleanupStream]);

  const start = useCallback(
    async (req: StartV2Request) => {
      cleanupStream();
      setState({ ...INITIAL_STATE, error: null });
      try {
        const resp = await startV2(req);
        setState((s) => ({ ...s, sessionId: resp.session_id }));

        const es = createV2Stream(resp.session_id);
        eventSourceRef.current = es;

        es.onmessage = (e) => {
          if (!e.data) return;
          let event: V2StreamEvent;
          try {
            event = JSON.parse(e.data) as V2StreamEvent;
          } catch (err) {
            console.error("Bad SSE payload", err);
            return;
          }
          if (event.type === "tick") {
            setState((s) => ({
              ...s,
              tick: event.tick,
              tam: event.tam,
              alive: event.alive,
              shares: event.shares,
              companies: event.companies,
            }));
          } else if (event.type === "complete") {
            setState((s) => ({ ...s, isComplete: true }));
            cleanupStream();
          } else if (event.type === "stopped") {
            setState((s) => ({ ...s, isStopped: true }));
            cleanupStream();
          }
        };

        es.onerror = (e) => {
          console.warn("SSE error", e);
          // EventSource auto-reconnects on transient errors; only mark error
          // if the connection is fully closed and the sim isn't complete.
          if (es.readyState === EventSource.CLOSED) {
            setState((s) =>
              s.isComplete || s.isStopped
                ? s
                : { ...s, error: "Stream closed unexpectedly" },
            );
          }
        };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setState((s) => ({ ...s, error: msg }));
      }
    },
    [cleanupStream],
  );

  const sendControl = useCallback(
    async (action: ControlAction, speed?: number) => {
      if (!state.sessionId) return;
      try {
        const resp = await controlV2(state.sessionId, action, speed);
        setState((s) => ({
          ...s,
          isPaused: resp.paused,
          isStopped: resp.stopped,
          speed: resp.speed,
        }));
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setState((s) => ({ ...s, error: msg }));
      }
    },
    [state.sessionId],
  );

  const play = useCallback(() => sendControl("play"), [sendControl]);
  const pause = useCallback(() => sendControl("pause"), [sendControl]);
  const stop = useCallback(() => sendControl("stop"), [sendControl]);
  const setSpeed = useCallback(
    (speed: number) => sendControl("set_speed", speed),
    [sendControl],
  );

  return {
    ...state,
    start,
    play,
    pause,
    stop,
    setSpeed,
    reset,
  };
}

export type { CompanySeed, CeoStance, CompanyTickPayload };
