"use client";

import { useEffect, useState } from "react";
import { ArrowLeft, Building2, Cpu, Plus, Scale, Shuffle, X } from "lucide-react";
import { Logo } from "@/components/ui/Logo";
import { COMPANY_COLORS } from "@/hooks/useSimulation";
import {
  AI_CADENCE_DAYS,
  type AppConfig,
  type Archetypes,
  type CompanyProfile,
  type ModelOption,
  type StartRequest,
  estimateRunCost,
  formatUsd,
  getArchetypes,
  getCompanies,
  pacedCadenceDays,
} from "@/lib/api";

const DURATIONS = [1, 5, 10, 20];
const MAX_COMPANIES = 10;
const DEFAULT_CEOS = ["founder_operator", "venture_growth", "bootstrap", "consolidator"];
const DEFAULT_BUDGET_USD = 1;

const COMPANY_LABELS: Record<string, string> = {
  solo_founder: "Solo founder",
  small_team: "Small team",
  venture_funded: "Venture-funded",
  enterprise: "Enterprise",
};

const label = "text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block";
const option = (active: boolean) =>
  `w-full flex items-center gap-3 p-3 rounded-lg border transition-all text-left ${
    active ? "border-accent bg-accent/5 ring-1 ring-accent/30" : "border-surface-200 hover:border-surface-300"
  }`;

function Toggle({ on, onClick, disabled }: { on: boolean; onClick: () => void; disabled?: boolean }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`w-10 h-6 rounded-full transition-all flex items-center flex-shrink-0 disabled:opacity-40 ${
        on ? "bg-accent justify-end" : "bg-surface-300 justify-start"
      }`}
    >
      <span className="w-4 h-4 bg-white rounded-full mx-1" />
    </button>
  );
}

interface SimSetupProps {
  model: ModelOption;
  config: AppConfig;
  onBack: () => void;
  onStart: (req: StartRequest) => void;
}

export function SimSetup({ model, config, onBack, onStart }: SimSetupProps) {
  const [archetypes, setArchetypes] = useState<Archetypes | null>(null);
  const [profiles, setProfiles] = useState<CompanyProfile[]>([]);
  const [profile, setProfile] = useState<CompanyProfile | null>(null);
  const [companyArchetype, setCompanyArchetype] = useState("small_team");
  const [ceos, setCeos] = useState<string[]>(DEFAULT_CEOS);
  const [years, setYears] = useState(5);
  const [aiEnabled, setAiEnabled] = useState(config.ai_available);
  const [budget, setBudget] = useState(DEFAULT_BUDGET_USD);
  const [shocks, setShocks] = useState(true);
  const [jevEnabled, setJevEnabled] = useState(config.jev_available);

  useEffect(() => {
    getArchetypes().then(setArchetypes).catch(() => setArchetypes({ ceo: [], company: [] }));
    getCompanies().then(setProfiles).catch(() => setProfiles([]));
  }, []);

  const estimate = estimateRunCost(model, ceos.length, years * 365);
  const cadence = pacedCadenceDays(model, ceos.length, years * 365, budget);
  const ceoLabel = (id: string) => archetypes?.ceo.find((a) => a.id === id)?.label ?? id;
  const nextCeo = () => archetypes?.ceo[ceos.length % archetypes.ceo.length]?.id ?? "bootstrap";

  const start = () =>
    onStart({
      seed: profile?.seed,
      company_archetype: companyArchetype,
      ceo_archetypes: ceos,
      duration_days: years * 365,
      ai_enabled: aiEnabled,
      model: aiEnabled ? model.id : undefined,
      budget_usd: budget,
      shocks_enabled: shocks,
      jev_enabled: jevEnabled,
    });

  return (
    <div className="min-h-screen flex flex-col items-center bg-surface-0 phase-enter px-6 py-12">
      <div className="flex items-center gap-3 mb-2">
        <button
          onClick={onBack}
          className="flex items-center justify-center w-8 h-8 rounded-lg hover:bg-surface-100 transition-colors"
          aria-label="Back to model picker"
        >
          <ArrowLeft size={16} className="text-surface-500" />
        </button>
        <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-surface-100 to-surface-50 border border-surface-200/50">
          <Logo size={20} className="text-accent" />
        </div>
        <h1 className="text-2xl font-bold text-surface-900 tracking-wide">ONYX LEOPARD</h1>
      </div>
      <button onClick={onBack} className="text-xs text-surface-500 mb-8 hover:text-accent">
        CEO model: <span className="font-medium text-surface-700">{model.name}</span> · change
      </button>

      <div className="max-w-3xl w-full grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-6">
          <div>
            <span className={label}>Company</span>
            <div className="space-y-2">
              {profiles.map((p) => (
                <button
                  key={p.filename}
                  onClick={() => setProfile(profile?.filename === p.filename ? null : p)}
                  className={option(profile?.filename === p.filename)}
                >
                  <Building2 size={16} className="text-surface-500 flex-shrink-0" />
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-surface-800 truncate">{p.name}</div>
                    <div className="text-[11px] text-surface-500 truncate">{p.industry}</div>
                  </div>
                </button>
              ))}
              <div className={option(profile === null)}>
                <Shuffle size={16} className="text-surface-500 flex-shrink-0" />
                <button onClick={() => setProfile(null)} className="flex-1 text-left">
                  <div className="text-sm font-medium text-surface-800">Random company</div>
                  <div className="text-[11px] text-surface-500">Sampled from an archetype</div>
                </button>
                <select
                  value={companyArchetype}
                  onChange={(e) => {
                    setCompanyArchetype(e.target.value);
                    setProfile(null);
                  }}
                  className="text-xs bg-surface-0 border border-surface-200 rounded-lg px-2 py-1.5 text-surface-700 outline-none focus:border-accent"
                >
                  {(archetypes?.company ?? Object.keys(COMPANY_LABELS)).map((a) => (
                    <option key={a} value={a}>
                      {COMPANY_LABELS[a] ?? a}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          <div>
            <span className={label}>CEOs ({ceos.length} companies, same starting point)</span>
            <div className="space-y-2">
              {ceos.map((ceo, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span
                    className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                    style={{ background: COMPANY_COLORS[i % COMPANY_COLORS.length] }}
                  />
                  <select
                    value={ceo}
                    onChange={(e) =>
                      setCeos((prev) => prev.map((c, j) => (j === i ? e.target.value : c)))
                    }
                    className="flex-1 text-xs bg-surface-0 border border-surface-200 rounded-lg px-2 py-1.5 text-surface-700 outline-none focus:border-accent"
                    title={archetypes?.ceo.find((a) => a.id === ceo)?.summary}
                  >
                    {(archetypes?.ceo ?? []).map((a) => (
                      <option key={a.id} value={a.id}>
                        {a.label}
                      </option>
                    ))}
                    {!archetypes?.ceo.length && <option value={ceo}>{ceoLabel(ceo)}</option>}
                  </select>
                  <button
                    onClick={() => setCeos((prev) => prev.filter((_, j) => j !== i))}
                    disabled={ceos.length <= 1}
                    className="p-1.5 rounded-md text-surface-400 hover:text-negative hover:bg-surface-100 disabled:opacity-30"
                    aria-label="Remove company"
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
              {ceos.length < MAX_COMPANIES && (
                <button
                  onClick={() => setCeos((prev) => [...prev, nextCeo()])}
                  className="flex items-center gap-1.5 text-xs text-accent hover:text-accent-dim"
                >
                  <Plus size={12} /> Add competitor
                </button>
              )}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div>
            <span className={label}>Duration</span>
            <div className="flex gap-2">
              {DURATIONS.map((y) => (
                <button
                  key={y}
                  onClick={() => setYears(y)}
                  className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${
                    years === y
                      ? "border-accent bg-accent/5 text-accent ring-1 ring-accent/30"
                      : "border-surface-200 text-surface-600 hover:border-surface-300"
                  }`}
                >
                  {y}y
                </button>
              ))}
            </div>
          </div>

          <div className="p-4 rounded-xl border border-surface-200 space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-start gap-2">
                <Cpu size={16} className="text-surface-500 mt-0.5" />
                <div>
                  <div className="text-sm font-medium text-surface-800">AI CEOs</div>
                  <div className="text-[11px] text-surface-500">
                    {config.ai_available
                      ? `${model.name} decides about monthly for every company`
                      : `Needs ${config.key_env} in backend/.env, then reload`}
                  </div>
                </div>
              </div>
              <Toggle
                on={aiEnabled}
                onClick={() => setAiEnabled((v) => !v)}
                disabled={!config.ai_available}
              />
            </div>
            {aiEnabled && (
              <div className="flex items-center justify-between gap-3 pt-3 border-t border-surface-100">
                <div className="text-[11px] text-surface-500">
                  Estimated <span className="font-mono text-surface-700">{formatUsd(estimate)}</span>
                  . Hard budget, paced over the whole run
                  {cadence > AI_CADENCE_DAYS ? (
                    <>
                      : at this budget the AI decides about every{" "}
                      <span className="font-mono text-surface-700">{cadence}</span> days instead of
                      monthly.
                    </>
                  ) : (
                    "."
                  )}
                </div>
                <label className="flex items-center gap-1 text-xs text-surface-600 flex-shrink-0">
                  $
                  <input
                    type="number"
                    min={0.05}
                    step={0.25}
                    value={budget}
                    onChange={(e) => setBudget(Math.max(0.05, Number(e.target.value) || 0))}
                    className="w-16 bg-surface-0 border border-surface-200 rounded-lg px-2 py-1 font-mono text-surface-800 outline-none focus:border-accent"
                  />
                </label>
              </div>
            )}
          </div>

          <div className="p-4 rounded-xl border border-surface-200 flex items-center justify-between gap-3">
            <div className="flex items-start gap-2">
              <Scale size={16} className="text-surface-500 mt-0.5" />
              <div>
                <div className="text-sm font-medium text-surface-800">Jev weekly judge</div>
                <div className="text-[11px] text-surface-500">
                  {config.jev_available
                    ? "Jev approves or vetoes each weekly rule-based move"
                    : "Needs TYPESAFE_API_KEY in backend/.env, then reload"}
                </div>
              </div>
            </div>
            <Toggle
              on={jevEnabled}
              onClick={() => setJevEnabled((v) => !v)}
              disabled={!config.jev_available}
            />
          </div>

          <div className="p-4 rounded-xl border border-surface-200 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium text-surface-800">Market shocks</div>
              <div className="text-[11px] text-surface-500">
                Crashes, recessions, competitor entries, departures
              </div>
            </div>
            <Toggle on={shocks} onClick={() => setShocks((v) => !v)} />
          </div>

          <button
            onClick={start}
            disabled={ceos.length === 0}
            className="w-full py-3 rounded-xl bg-accent text-white font-semibold text-sm transition-all hover:bg-accent/90 hover:shadow-lg hover:shadow-accent/20 disabled:opacity-40"
          >
            Start Simulation
          </button>
        </div>
      </div>
    </div>
  );
}
