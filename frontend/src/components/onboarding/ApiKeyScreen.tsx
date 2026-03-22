"use client";

import { useState } from "react";
import { setApiKey } from "@/lib/api";
import { Logo } from "@/components/ui/Logo";
import { Spinner } from "@/components/ui/Spinner";
import { MessageSquare, Zap, Share2 } from "@/components/ui/icons";

interface ApiKeyScreenProps {
  onComplete: () => void;
}

export function ApiKeyScreen({ onComplete }: ApiKeyScreenProps) {
  const [key, setKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = key.trim();
    if (!trimmed) return;

    if (!trimmed.startsWith("sk-ant-")) {
      setError("API key should start with sk-ant-");
      return;
    }

    setLoading(true);
    setError("");
    try {
      await setApiKey(trimmed);
      onComplete();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to set API key");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-surface-0 flex items-center justify-center relative overflow-hidden phase-enter">
      {/* Animated gradient background */}
      <div className="absolute inset-0">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-accent/8 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-secondary/8 rounded-full blur-3xl animate-pulse delay-1000" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-accent/3 rounded-full blur-3xl" />
      </div>

      {/* Grid pattern overlay */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
          backgroundSize: "60px 60px",
        }}
      />

      <div className="relative z-10 w-full max-w-md mx-4">
        {/* Logo / Brand */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-surface-100 to-surface-50 border border-surface-200/50 shadow-2xl mb-6">
            <Logo size={32} className="text-accent" />
          </div>
          <p className="text-surface-500 text-xs font-medium tracking-widest uppercase mb-2">Step 1 of 2</p>
          <h1 className="text-3xl font-bold text-surface-900 tracking-tight">
            ONYX LEOPARD
          </h1>
          <p className="text-surface-500 mt-2 text-sm leading-relaxed">
            AI-powered business strategy simulator.<br />
            Paste your Anthropic API key to get started.
          </p>
        </div>

        {/* Card */}
        <div className="bg-surface-50/80 backdrop-blur-xl border border-surface-200 rounded-2xl p-6 shadow-2xl">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="api-key" className="block text-sm font-medium text-surface-700 mb-2">
                Anthropic API Key
              </label>
              <input
                id="api-key"
                type="password"
                value={key}
                onChange={(e) => { setKey(e.target.value); setError(""); }}
                placeholder="sk-ant-api03-..."
                autoFocus
                className="w-full bg-surface-0 border border-surface-300 rounded-xl px-4 py-3 text-sm text-surface-900 placeholder-surface-500 focus:outline-none focus:ring-2 focus:ring-accent/30 focus:border-accent/40 transition-all"
              />
            </div>

            {error && (
              <p className="text-negative text-xs">{error}</p>
            )}

            <button
              type="submit"
              disabled={loading || !key.trim()}
              className="btn-primary w-full"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <Spinner className="h-4 w-4" />
                  Connecting...
                </span>
              ) : (
                "Launch Simulator"
              )}
            </button>
          </form>

          <p className="text-surface-500 text-xs text-center mt-4">
            Your key stays local — never sent anywhere except Anthropic&apos;s API.
          </p>
        </div>

        {/* Features preview */}
        <div className="mt-8 grid grid-cols-3 gap-3">
          {[
            { Icon: MessageSquare, label: "Natural Language", desc: "Describe companies in plain English" },
            { Icon: Zap, label: "Live Simulation", desc: "Multi-agent weekly decisions" },
            { Icon: Share2, label: "Strategy Testing", desc: "Fork scenarios & compare" },
          ].map((f) => (
            <div key={f.label} className="bg-surface-50/50 border border-surface-200/50 rounded-xl p-3 text-center">
              <div className="flex justify-center mb-1">
                <f.Icon size={18} className="text-surface-500" />
              </div>
              <div className="text-xs font-medium text-surface-700">{f.label}</div>
              <div className="text-xs text-surface-500 mt-0.5">{f.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
