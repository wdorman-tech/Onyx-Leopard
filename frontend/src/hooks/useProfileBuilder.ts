"use client";

import { useCallback, useState } from "react";
import {
  answerProfile,
  confirmProfile,
  startProfile,
  uploadDocument,
} from "@/lib/api";

interface Message {
  role: "assistant" | "user";
  content: string;
}

interface DocumentSummary {
  filename: string;
  summary: string;
}

interface ProfileBuilderState {
  sessionId: string | null;
  messages: Message[];
  progress: number;
  isComplete: boolean;
  isLoading: boolean;
  isUploading: boolean;
  industrySpec: Record<string, unknown> | null;
  confirmedSlug: string | null;
  documents: DocumentSummary[];
  error: string | null;
}

export function useProfileBuilder() {
  const [state, setState] = useState<ProfileBuilderState>({
    sessionId: null,
    messages: [],
    progress: 0,
    isComplete: false,
    isLoading: false,
    isUploading: false,
    industrySpec: null,
    confirmedSlug: null,
    documents: [],
    error: null,
  });

  const start = useCallback(async () => {
    setState((s) => ({ ...s, isLoading: true, error: null }));
    try {
      const res = await startProfile();
      setState((s) => ({
        ...s,
        sessionId: res.session_id,
        messages: [{ role: "assistant", content: res.first_question }],
        isLoading: false,
      }));
    } catch (e) {
      setState((s) => ({
        ...s,
        isLoading: false,
        error: e instanceof Error ? e.message : "Failed to start interview",
      }));
    }
  }, []);

  const sendAnswer = useCallback(
    async (answer: string) => {
      if (!state.sessionId) return;
      setState((s) => ({
        ...s,
        messages: [...s.messages, { role: "user", content: answer }],
        isLoading: true,
        error: null,
      }));
      try {
        const res = await answerProfile(state.sessionId, answer);
        setState((s) => ({
          ...s,
          messages: [
            ...s.messages,
            { role: "assistant", content: res.next_question },
          ],
          progress: res.progress,
          isComplete: res.is_complete,
          industrySpec: res.industry_spec ?? s.industrySpec,
          isLoading: false,
          error: res.error ?? null,
        }));
      } catch (e) {
        setState((s) => ({
          ...s,
          isLoading: false,
          error: e instanceof Error ? e.message : "Failed to process answer",
        }));
      }
    },
    [state.sessionId],
  );

  const confirm = useCallback(
    async (slug: string) => {
      if (!state.sessionId) return;
      setState((s) => ({ ...s, isLoading: true, error: null }));
      try {
        const res = await confirmProfile(state.sessionId, slug);
        setState((s) => ({
          ...s,
          confirmedSlug: res.slug,
          isLoading: false,
        }));
      } catch (e) {
        setState((s) => ({
          ...s,
          isLoading: false,
          error: e instanceof Error ? e.message : "Failed to save industry",
        }));
      }
    },
    [state.sessionId],
  );

  const upload = useCallback(
    async (file: File) => {
      if (!state.sessionId) return;
      setState((s) => ({ ...s, isUploading: true, error: null }));
      try {
        const result = await uploadDocument(state.sessionId, file);
        setState((s) => ({
          ...s,
          isUploading: false,
          documents: [...s.documents, result],
          messages: [
            ...s.messages,
            {
              role: "assistant" as const,
              content: `I've analyzed "${result.filename}". Here's what I found:\n\n${result.summary}`,
            },
          ],
        }));
      } catch (e) {
        setState((s) => ({
          ...s,
          isUploading: false,
          error: e instanceof Error ? e.message : "Upload failed",
        }));
      }
    },
    [state.sessionId],
  );

  const reset = useCallback(() => {
    setState({
      sessionId: null,
      messages: [],
      progress: 0,
      isComplete: false,
      isLoading: false,
      isUploading: false,
      industrySpec: null,
      confirmedSlug: null,
      documents: [],
      error: null,
    });
  }, []);

  return {
    ...state,
    start,
    sendAnswer,
    upload,
    confirm,
    reset,
  };
}
