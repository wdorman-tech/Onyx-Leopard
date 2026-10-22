"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, ArrowRight, KeyRound, Loader2 } from "lucide-react";
import { Logo } from "@/components/ui/Logo";
import {
  type AppConfig,
  type ModelCatalog,
  type ModelOption,
  type Tier,
  estimateRunCost,
  formatUsd,
  getConfig,
  getModels,
} from "@/lib/api";

const STORAGE_KEY = "onyx.model";
// Reference run for the cost hint: 4 AI CEOs over 5 years.
const HINT_COMPANIES = 4;
const HINT_DAYS = 5 * 365;
// Upper end of the intelligence-index bar (frontier models sit near 60).
const INDEX_SCALE = 60;

const TIER_ACCENT: Record<Tier, string> = {
  bad: "#ef4444",
  mediocre: "#f59e0b",
  good: "#22c55e",
  super: "#8b5cf6",
};

function readStored(): string | null {
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function store(id: string) {
  try {
    window.localStorage.setItem(STORAGE_KEY, id);
  } catch {
    // storage unavailable (private mode); selection just isn't remembered
  }
}

function IndexBar({ value, color }: { value: number | null; color: string }) {
  if (value == null) return <span className="text-[10px] text-surface-400">no score</span>;
  return (
    <div className="flex items-center gap-1.5" title="Artificial Analysis intelligence index">
      <div className="w-16 h-1.5 rounded-full bg-surface-200 overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{ width: `${Math.min(100, (value / INDEX_SCALE) * 100)}%`, background: color }}
        />
      </div>
      <span className="text-[10px] font-mono text-surface-600">{value.toFixed(1)}</span>
    </div>
  );
}

function Price({ model }: { model: ModelOption }) {
  return (
    <span className="text-[10px] font-mono text-surface-500">
      ${model.prompt_per_million.toFixed(2)} / ${model.completion_per_million.toFixed(2)} per 1M
    </span>
  );
}

interface ModelPickerProps {
  onContinue: (model: ModelOption, config: AppConfig) => void;
}

export function ModelPicker({ onContinue }: ModelPickerProps) {
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const load = useCallback(() => {
    Promise.all([getModels(), getConfig()])
      .then(([models, cfg]) => {
        setError(null);
        setCatalog(models);
        setConfig(cfg);
        const all = [...models.tiers, ...models.anthropic].map((m) => m.id);
        const stored = readStored();
        setSelected(stored && all.includes(stored) ? stored : models.default);
      })
      .catch(() =>
        setError("Can't reach the backend on localhost:8000. Is `python start.py` running?"),
      );
  }, []);

  useEffect(load, [load]);

  const all: ModelOption[] = catalog ? [...catalog.tiers, ...catalog.anthropic] : [];
  const choice = all.find((m) => m.id === selected) ?? null;

  const pick = (id: string) => {
    setSelected(id);
    store(id);
  };

  return (
    <div className="min-h-screen flex flex-col items-center bg-surface-0 phase-enter px-6 py-12">
      <div className="flex items-center gap-3 mb-2">
        <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-surface-100 to-surface-50 border border-surface-200/50">
          <Logo size={20} className="text-accent" />
        </div>
        <h1 className="text-2xl font-bold text-surface-900 tracking-wide">ONYX LEOPARD</h1>
      </div>
      <p className="text-sm text-surface-500 mb-8">Pick the model behind every AI CEO</p>

      {error && (
        <div className="max-w-xl w-full flex items-center justify-between gap-3 p-4 rounded-xl border border-negative/30 bg-negative/5 text-sm text-surface-700">
          <span>{error}</span>
          <button onClick={load} className="btn-secondary !py-1.5 !px-3 text-xs">
            Retry
          </button>
        </div>
      )}

      {!catalog && !error && <Loader2 className="animate-spin text-surface-400" size={20} />}

      {catalog && config && (
        <div className="max-w-4xl w-full space-y-6">
          {!config.ai_available && (
            <div className="flex gap-3 p-4 rounded-xl border border-warning/40 bg-warning/5">
              <KeyRound size={18} className="text-warning flex-shrink-0 mt-0.5" />
              <div className="text-xs text-surface-700 leading-relaxed">
                <div className="font-semibold text-surface-800 mb-0.5">No OpenRouter key yet</div>
                Add <code className="font-mono">{config.key_env}=sk-or-...</code> to{" "}
                <code className="font-mono">backend/.env</code> (get one at openrouter.ai/keys)
                and reload this page; no restart needed. Until then CEOs run on free heuristics.
              </div>
            </div>
          )}
          {!catalog.live && (
            <div className="flex items-center gap-2 text-[11px] text-surface-500">
              <AlertTriangle size={12} /> OpenRouter catalog unreachable; showing offline list
              without prices.
            </div>
          )}

          <section>
            <h2 className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2">
              Chinese open-weight models
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {catalog.tiers.map((t) => {
                const active = selected === t.id;
                const accent = TIER_ACCENT[t.tier];
                return (
                  <button
                    key={t.id}
                    onClick={() => pick(t.id)}
                    className={`flex flex-col text-left gap-2 p-4 rounded-2xl border transition-all ${
                      active
                        ? "border-accent bg-accent/5 ring-1 ring-accent/30"
                        : "border-surface-200 hover:border-surface-300"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span
                        className="text-[10px] font-bold uppercase tracking-widest"
                        style={{ color: accent }}
                      >
                        {t.label}
                      </span>
                      {t.id === catalog.default && (
                        <span className="text-[9px] font-semibold text-accent bg-accent/10 px-1.5 py-0.5 rounded">
                          RECOMMENDED
                        </span>
                      )}
                    </div>
                    <div>
                      <div className="text-sm font-semibold text-surface-900 leading-tight">
                        {t.name}
                      </div>
                      <div className="text-[10px] font-mono text-surface-400 truncate">{t.id}</div>
                    </div>
                    <p className="text-[11px] text-surface-600 leading-snug flex-1">{t.blurb}</p>
                    <IndexBar value={t.intelligence_index} color={accent} />
                    <Price model={t} />
                    <div className="text-[10px] text-surface-500">
                      ≈ {formatUsd(estimateRunCost(t, HINT_COMPANIES, HINT_DAYS))} per 5-yr run
                    </div>
                  </button>
                );
              })}
            </div>
          </section>

          <section>
            <h2 className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2">
              Claude (Anthropic)
            </h2>
            <div className="border border-surface-200 rounded-2xl divide-y divide-surface-100 max-h-72 overflow-y-auto">
              {catalog.anthropic.map((m) => {
                const active = selected === m.id;
                return (
                  <button
                    key={m.id}
                    onClick={() => pick(m.id)}
                    className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                      active ? "bg-accent/5" : "hover:bg-surface-50"
                    }`}
                  >
                    <span
                      className={`w-3 h-3 rounded-full border-2 flex-shrink-0 ${
                        active ? "border-accent bg-accent" : "border-surface-300"
                      }`}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-medium text-surface-800 truncate">
                        {m.name.replace(/^Anthropic:\s*/, "")}
                      </div>
                      <div className="text-[10px] font-mono text-surface-400 truncate">{m.id}</div>
                    </div>
                    <IndexBar value={m.intelligence_index} color="#d97706" />
                    <div className="hidden md:block w-40 text-right">
                      <Price model={m} />
                    </div>
                    <div className="w-24 text-right text-[10px] text-surface-500">
                      ≈ {formatUsd(estimateRunCost(m, HINT_COMPANIES, HINT_DAYS))}/run
                    </div>
                  </button>
                );
              })}
            </div>
          </section>

          <div className="flex items-center justify-between gap-4 pt-2">
            <p className="text-[11px] text-surface-500">
              Run estimates assume {HINT_COMPANIES} AI CEOs deciding monthly for 5 years. You set a
              hard budget on the next screen.
            </p>
            <button
              disabled={!choice}
              onClick={() => choice && onContinue(choice, config)}
              className="btn-primary flex items-center gap-2 flex-shrink-0"
            >
              Continue <ArrowRight size={14} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
