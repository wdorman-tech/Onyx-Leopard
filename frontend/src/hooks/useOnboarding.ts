"use client";

import { useState, useCallback } from "react";
import type { CompanyProfile } from "@/types/profile";
import type { CompanyGraph } from "@/types/graph";
import {
  startOnboarding,
  respondOnboarding,
  skipOnboardingPhase,
  completeOnboarding,
  quickStart,
} from "@/lib/api";

export type OnboardingPhase =
  | "identity"
  | "organization"
  | "financials"
  | "market"
  | "operations"
  | "strategy";

const PHASE_LABELS: Record<OnboardingPhase, string> = {
  identity: "Identity & Industry",
  organization: "Scale & Structure",
  financials: "Financials",
  market: "Market & Competition",
  operations: "Operations",
  strategy: "Strategy & Goals",
};

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface UseOnboardingReturn {
  sessionId: string | null;
  phase: OnboardingPhase;
  phaseLabel: string;
  messages: Message[];
  profile: CompanyProfile | null;
  graph: CompanyGraph | null;
  completionEstimate: number;
  loading: boolean;
  error: string | null;
  start: () => Promise<void>;
  respond: (message: string) => Promise<void>;
  skipPhase: () => Promise<void>;
  complete: () => Promise<{ profile: CompanyProfile; graph: CompanyGraph } | null>;
  doQuickStart: (description: string) => Promise<{
    profile: CompanyProfile;
    graph: CompanyGraph;
    sessionId: string;
  } | null>;
}

export function useOnboarding(): UseOnboardingReturn {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [phase, setPhase] = useState<OnboardingPhase>("identity");
  const [messages, setMessages] = useState<Message[]>([]);
  const [profile, setProfile] = useState<CompanyProfile | null>(null);
  const [graph, setGraph] = useState<CompanyGraph | null>(null);
  const [completionEstimate, setCompletionEstimate] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await startOnboarding();
      setSessionId(res.session_id);
      setPhase(res.phase as OnboardingPhase);
      setMessages([{ role: "assistant", content: res.question }]);
      setProfile(res.profile);
      setCompletionEstimate(res.completion_estimate);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start");
    } finally {
      setLoading(false);
    }
  }, []);

  const respond = useCallback(
    async (message: string) => {
      if (!sessionId) return;
      setLoading(true);
      setError(null);

      setMessages((prev) => [...prev, { role: "user", content: message }]);

      try {
        const res = await respondOnboarding(sessionId, message);
        setPhase(res.phase as OnboardingPhase);
        setMessages((prev) => [
          ...prev,
          { role: "assistant" as const, content: res.question },
        ]);
        setProfile(res.profile);
        setCompletionEstimate(res.completion_estimate);
        if (res.graph) setGraph(res.graph);
      } catch (err) {
        // Remove orphaned user message on error
        setMessages((prev) => prev.slice(0, -1));
        setError(err instanceof Error ? err.message : "Failed to respond");
      } finally {
        setLoading(false);
      }
    },
    [sessionId],
  );

  const skipPhase = useCallback(async () => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await skipOnboardingPhase(sessionId, phase);
      setPhase(res.phase as OnboardingPhase);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.question },
      ]);
      setProfile(res.profile);
      setCompletionEstimate(res.completion_estimate);
      if (res.graph) setGraph(res.graph);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to skip phase");
    } finally {
      setLoading(false);
    }
  }, [sessionId, phase]);

  const complete = useCallback(async () => {
    if (!sessionId) return null;
    setLoading(true);
    setError(null);
    try {
      const res = await completeOnboarding(sessionId);
      setProfile(res.profile);
      setGraph(res.graph);
      return { profile: res.profile, graph: res.graph };
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to complete");
      return null;
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  const doQuickStart = useCallback(async (description: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await quickStart(description);
      setSessionId(res.session_id);
      setProfile(res.profile);
      setGraph(res.graph);
      return {
        profile: res.profile,
        graph: res.graph,
        sessionId: res.session_id,
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to quick start");
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    sessionId,
    phase,
    phaseLabel: PHASE_LABELS[phase],
    messages,
    profile,
    graph,
    completionEstimate,
    loading,
    error,
    start,
    respond,
    skipPhase,
    complete,
    doQuickStart,
  };
}
