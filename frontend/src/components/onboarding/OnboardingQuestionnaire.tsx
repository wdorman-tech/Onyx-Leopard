"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import type { CompanyProfile } from "@/types/profile";
import type { CompanyGraph } from "@/types/graph";
import { useOnboarding, type OnboardingPhase } from "@/hooks/useOnboarding";
import { Logo } from "@/components/ui/Logo";
import { Spinner } from "@/components/ui/Spinner";
import { MessageSquare, Zap, ChevronLeft, Send } from "@/components/ui/icons";

const PHASES: OnboardingPhase[] = [
  "identity",
  "organization",
  "financials",
  "market",
  "operations",
  "strategy",
];

const PHASE_LABELS: Record<OnboardingPhase, string> = {
  identity: "Identity",
  organization: "Structure",
  financials: "Financials",
  market: "Market",
  operations: "Operations",
  strategy: "Strategy",
};

interface OnboardingQuestionnaireProps {
  onComplete: (
    profile: CompanyProfile,
    graph: CompanyGraph,
    sessionId: string,
  ) => void;
}

export function OnboardingQuestionnaire({
  onComplete,
}: OnboardingQuestionnaireProps) {
  const [mode, setMode] = useState<"select" | "guided" | "quick">("select");
  const [quickText, setQuickText] = useState("");
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const onboarding = useOnboarding();

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [onboarding.messages]);

  const handleStartGuided = useCallback(async () => {
    setMode("guided");
    await onboarding.start();
  }, [onboarding]);

  const handleSend = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!input.trim() || onboarding.loading) return;
      const msg = input.trim();
      setInput("");
      await onboarding.respond(msg);
    },
    [input, onboarding],
  );

  const handleComplete = useCallback(async () => {
    const result = await onboarding.complete();
    if (result && onboarding.sessionId) {
      onComplete(result.profile, result.graph, onboarding.sessionId);
    }
  }, [onboarding, onComplete]);

  const handleQuickStart = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!quickText.trim() || onboarding.loading) return;
      const result = await onboarding.doQuickStart(quickText.trim());
      if (result) {
        onComplete(result.profile, result.graph, result.sessionId);
      }
    },
    [quickText, onboarding, onComplete],
  );

  // Mode selection screen
  if (mode === "select") {
    return (
      <div className="min-h-screen bg-surface-0 flex items-center justify-center p-6 phase-enter">
        <div className="w-full max-w-lg space-y-6">
          <div className="text-center space-y-2">
            <div className="flex items-center justify-center gap-2 mb-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-surface-100 to-surface-50 border border-surface-200/50 flex items-center justify-center">
                <Logo size={20} className="text-accent" />
              </div>
            </div>
            <h1 className="text-xl font-bold text-surface-900">
              Set Up Your Company
            </h1>
            <p className="text-sm text-surface-500">
              Choose how you&apos;d like to describe your company for the
              simulation.
            </p>
          </div>

          <button
            onClick={handleStartGuided}
            className="w-full text-left p-5 rounded-2xl border-l-4 border-l-accent border border-surface-200 hover:border-accent/40 hover:bg-surface-50/50 transition-all group"
          >
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center text-accent">
                <MessageSquare size={16} />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold text-surface-800 group-hover:text-surface-900">
                    Guided Interview
                  </span>
                  <span className="text-[10px] font-medium bg-accent/10 text-accent px-2 py-0.5 rounded-full">
                    Recommended
                  </span>
                </div>
                <div className="text-xs text-surface-500">Step-by-step setup</div>
              </div>
            </div>
            <p className="text-xs text-surface-500 ml-11">
              AI asks structured questions across 6 phases. Graph updates in
              real-time as you answer.
            </p>
          </button>

          <button
            onClick={() => setMode("quick")}
            className="w-full text-left p-5 rounded-2xl border border-surface-200 hover:border-surface-300 hover:bg-surface-50/50 transition-all group"
          >
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 rounded-lg bg-surface-100 border border-surface-200 flex items-center justify-center text-surface-500">
                <Zap size={16} />
              </div>
              <div>
                <div className="text-sm font-semibold text-surface-800 group-hover:text-surface-900">
                  Quick Start
                </div>
                <div className="text-xs text-surface-500">
                  Describe in one go
                </div>
              </div>
            </div>
            <p className="text-xs text-surface-500 ml-11">
              Paste a company description and jump straight to the simulation.
            </p>
          </button>
        </div>
      </div>
    );
  }

  // Quick start mode
  if (mode === "quick") {
    return (
      <div className="min-h-screen bg-surface-0 flex items-center justify-center p-6 phase-enter">
        <div className="w-full max-w-lg space-y-4">
          <button
            onClick={() => setMode("select")}
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <ChevronLeft size={12} />
            Back
          </button>

          <h2 className="text-lg font-bold text-surface-900">Quick Start</h2>
          <p className="text-sm text-surface-500">
            Describe your company in detail. Include industry, size, revenue,
            departments, competitors — anything relevant.
          </p>

          <form onSubmit={handleQuickStart} className="space-y-3">
            <textarea
              value={quickText}
              onChange={(e) => setQuickText(e.target.value)}
              placeholder="We are a B2B SaaS company with 120 employees across engineering, sales, and marketing..."
              className="w-full h-48 bg-surface-50 border border-surface-200 rounded-xl p-4 text-sm text-surface-800 placeholder:text-surface-500 resize-none focus:outline-none focus:ring-2 focus:ring-accent/30 focus:border-accent/40 transition-all"
            />
            <button
              type="submit"
              disabled={!quickText.trim() || onboarding.loading}
              className="btn-primary w-full"
            >
              {onboarding.loading ? (
                <span className="flex items-center justify-center gap-2">
                  <Spinner className="h-4 w-4" />
                  Analyzing...
                </span>
              ) : (
                "Generate Profile & Graph"
              )}
            </button>
          </form>

          {onboarding.error && (
            <div className="text-xs text-negative bg-negative/10 px-3 py-2 rounded-lg">
              {onboarding.error}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Guided interview mode
  return (
    <div className="min-h-screen bg-surface-0 flex flex-col phase-enter">
      {/* Phase progress bar */}
      <div className="px-6 py-4 border-b border-surface-200/50">
        <div className="flex items-center justify-between mb-3">
          <button
            onClick={() => setMode("select")}
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <ChevronLeft size={12} />
            Back
          </button>
          <span className="text-xs text-surface-500">
            {Math.round(onboarding.completionEstimate * 100)}% complete
          </span>
        </div>

        <div className="flex gap-1.5">
          {PHASES.map((p) => {
            const isActive = p === onboarding.phase;
            const idx = PHASES.indexOf(p);
            const currentIdx = PHASES.indexOf(onboarding.phase);
            const isPast = idx < currentIdx;

            return (
              <div key={p} className="flex-1 space-y-1">
                <div
                  className={`h-1 rounded-full transition-colors ${
                    isPast
                      ? "bg-accent"
                      : isActive
                        ? "bg-accent/50"
                        : "bg-surface-200"
                  }`}
                />
                <div
                  className={`text-[10px] text-center transition-colors ${
                    isActive ? "text-accent" : "text-surface-500"
                  }`}
                >
                  {PHASE_LABELS[p]}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {onboarding.messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] px-4 py-3 rounded-2xl text-sm ${
                msg.role === "user"
                  ? "bg-accent/15 text-surface-800 rounded-br-md"
                  : "bg-surface-100/50 text-surface-700 rounded-bl-md"
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}

        {onboarding.loading && (
          <div className="flex justify-start">
            <div className="bg-surface-100/50 px-4 py-3 rounded-2xl rounded-bl-md">
              <div className="flex gap-1">
                <div className="w-1.5 h-1.5 bg-surface-400 rounded-full animate-bounce" />
                <div
                  className="w-1.5 h-1.5 bg-surface-400 rounded-full animate-bounce"
                  style={{ animationDelay: "0.15s" }}
                />
                <div
                  className="w-1.5 h-1.5 bg-surface-400 rounded-full animate-bounce"
                  style={{ animationDelay: "0.3s" }}
                />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-6 py-4 border-t border-surface-200/50">
        {onboarding.error && (
          <div className="text-xs text-negative mb-2">{onboarding.error}</div>
        )}

        <form onSubmit={handleSend} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your answer..."
            disabled={onboarding.loading}
            className="flex-1 bg-surface-50 border border-surface-200 rounded-xl px-4 py-2.5 text-sm text-surface-800 placeholder:text-surface-500 focus:outline-none focus:ring-2 focus:ring-accent/30 focus:border-accent/40 disabled:opacity-50 transition-all"
          />
          <button
            type="submit"
            disabled={!input.trim() || onboarding.loading}
            className="btn-primary px-4 py-2.5"
          >
            <Send size={16} />
          </button>
        </form>

        <div className="flex items-center justify-between mt-3">
          <button
            onClick={onboarding.skipPhase}
            disabled={onboarding.loading}
            className="btn-ghost text-xs"
          >
            Skip this phase →
          </button>
          <button
            onClick={handleComplete}
            disabled={onboarding.loading || onboarding.completionEstimate < 0.15}
            className="text-xs text-accent hover:text-accent-bright font-medium transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Finish & Generate Graph
          </button>
        </div>
      </div>
    </div>
  );
}
