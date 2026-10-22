"use client";

import { useEffect, useState } from "react";
import { ArrowLeft, Building2 } from "lucide-react";
import { Logo } from "@/components/ui/Logo";
import { getImportedCompanies, type ImportedCompany } from "@/lib/apiV2";

const AGENT_NAMES = [
  "Alpha Corp", "Beta Inc", "Gamma Ltd", "Delta Co", "Epsilon Group",
  "Zeta Holdings", "Eta Ventures", "Theta Partners",
];

const START_MODES = [
  { value: "identical", label: "Identical", desc: "All companies start with equal resources" },
  { value: "randomized", label: "Randomized", desc: "Varied starting cash and conditions" },
  { value: "staggered", label: "Staggered", desc: "Companies enter the market at different times" },
];

const CEO_STRATEGIES = [
  { value: "aggressive_growth", label: "Aggressive Growth", desc: "Rapid expansion, heavy marketing" },
  { value: "quality_focus", label: "Quality Focus", desc: "Satisfaction and quality above all" },
  { value: "cost_leader", label: "Cost Leader", desc: "Minimize costs, operational efficiency" },
  { value: "balanced", label: "Balanced", desc: "Steady growth, no overextension" },
  { value: "market_dominator", label: "Market Dominator", desc: "Pursue market share aggressively" },
  { value: "survivor", label: "Survivor", desc: "Cash preservation, strong margins" },
];

interface UnifiedSetupProps {
  startMode: string;
  companyCount: number;
  aiCeoEnabled: boolean;
  durationYears: number;
  durationOptions: number[];
  companyStrategies: Record<number, string>;
  selectedCompany: ImportedCompany | null;
  onSetStartMode: (mode: string) => void;
  onSetCompanyCount: (count: number) => void;
  onSetAiCeoEnabled: (enabled: boolean) => void;
  onSetDurationYears: (years: number) => void;
  onSetCompanyStrategies: (fn: (prev: Record<number, string>) => Record<number, string>) => void;
  onSetSelectedCompany: (company: ImportedCompany | null) => void;
  onStart: () => void;
  onBack: () => void;
}

export function UnifiedSetup({
  startMode,
  companyCount,
  aiCeoEnabled,
  durationYears,
  durationOptions,
  companyStrategies,
  selectedCompany,
  onSetStartMode,
  onSetCompanyCount,
  onSetAiCeoEnabled,
  onSetDurationYears,
  onSetCompanyStrategies,
  onSetSelectedCompany,
  onStart,
  onBack,
}: UnifiedSetupProps) {
  const [importedCompanies, setImportedCompanies] = useState<ImportedCompany[]>([]);

  useEffect(() => {
    getImportedCompanies()
      .then(setImportedCompanies)
      .catch(() => setImportedCompanies([]));
  }, []);

  return (
    <div className="h-screen flex flex-col bg-surface-0 phase-enter relative">
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-12">
        <div className="flex items-center gap-3 mb-2">
          <button
            onClick={onBack}
            className="flex items-center justify-center w-8 h-8 rounded-lg hover:bg-surface-100 transition-colors"
          >
            <ArrowLeft size={16} className="text-surface-500" />
          </button>
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-surface-100 to-surface-50 border border-surface-200/50">
            <Logo size={20} className="text-accent" />
          </div>
          <h1 className="text-2xl font-bold text-surface-900 tracking-wide">
            ONYX LEOPARD
          </h1>
        </div>
        <p className="text-sm text-surface-500 mb-8">
          Configure the static simulation
        </p>

        <div className="max-w-md w-full space-y-6">
          {importedCompanies.length > 0 && (
            <div>
              <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block">
                Imported Profile
              </label>
              <div className="grid grid-cols-1 gap-2">
                {importedCompanies.map((company) => (
                  <button
                    key={company.filename}
                    onClick={() =>
                      onSetSelectedCompany(
                        selectedCompany?.filename === company.filename ? null : company,
                      )
                    }
                    className={`w-full flex items-center gap-3 p-3 rounded-lg border transition-all text-left ${
                      selectedCompany?.filename === company.filename
                        ? "border-accent bg-accent/5 ring-1 ring-accent/30"
                        : "border-surface-200 hover:border-surface-300"
                    }`}
                  >
                    <Building2 size={16} className="text-surface-500 flex-shrink-0" />
                    <div className="min-w-0">
                      <div className="text-sm font-medium text-surface-800 truncate">
                        {company.name}
                      </div>
                      <div className="text-[11px] text-surface-500 truncate">
                        {company.industry || company.seed.economics_model}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div>
            <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block">
              Starting Conditions
            </label>
            <div className="space-y-2">
              {START_MODES.map((m) => (
                <button
                  key={m.value}
                  onClick={() => onSetStartMode(m.value)}
                  className={`w-full flex items-center gap-3 p-3 rounded-lg border transition-all text-left ${
                    startMode === m.value
                      ? "border-accent bg-accent/5 ring-1 ring-accent/30"
                      : "border-surface-200 hover:border-surface-300"
                  }`}
                >
                  <div
                    className={`w-3 h-3 rounded-full border-2 flex-shrink-0 ${
                      startMode === m.value
                        ? "border-accent bg-accent"
                        : "border-surface-300"
                    }`}
                  />
                  <div>
                    <div className="text-sm font-medium text-surface-800">{m.label}</div>
                    <div className="text-[11px] text-surface-500">{m.desc}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block">
              Companies ({companyCount})
            </label>
            <input
              type="range"
              min={2}
              max={20}
              value={companyCount}
              onChange={(e) => onSetCompanyCount(Number(e.target.value))}
              className="w-full accent-accent"
            />
            <div className="flex justify-between text-[10px] text-surface-400 mt-1">
              <span>2</span>
              <span>20</span>
            </div>
          </div>

          <div>
            <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block">
              AI CEO Agents
            </label>
            <button
              onClick={() => onSetAiCeoEnabled(!aiCeoEnabled)}
              className={`w-full flex items-center justify-between p-3 rounded-lg border transition-all ${
                aiCeoEnabled
                  ? "border-accent bg-accent/5 ring-1 ring-accent/30"
                  : "border-surface-200 hover:border-surface-300"
              }`}
            >
              <div className="text-left">
                <div className="text-sm font-medium text-surface-800">
                  {aiCeoEnabled ? "Enabled" : "Disabled"}
                </div>
                <div className="text-[11px] text-surface-500">
                  Each company gets a Claude AI CEO making strategic decisions
                </div>
              </div>
              <div
                className={`w-10 h-6 rounded-full transition-all flex items-center ${
                  aiCeoEnabled ? "bg-accent justify-end" : "bg-surface-300 justify-start"
                }`}
              >
                <div className="w-4 h-4 bg-white rounded-full mx-1" />
              </div>
            </button>
          </div>

          {aiCeoEnabled && (
            <>
              <div>
                <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block">
                  Duration
                </label>
                <div className="flex gap-2">
                  {durationOptions.map((y) => (
                    <button
                      key={y}
                      onClick={() => onSetDurationYears(y)}
                      className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${
                        durationYears === y
                          ? "border-accent bg-accent/5 text-accent ring-1 ring-accent/30"
                          : "border-surface-200 text-surface-600 hover:border-surface-300"
                      }`}
                    >
                      {y} Years
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block">
                  CEO Strategies
                </label>
                <div className="space-y-2 max-h-[240px] overflow-y-auto">
                  {Array.from({ length: companyCount }, (_, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <span className="text-xs text-surface-500 w-24 truncate flex-shrink-0">
                        {i === 0 && selectedCompany ? selectedCompany.name : AGENT_NAMES[i]}
                      </span>
                      <select
                        value={companyStrategies[i] || "balanced"}
                        onChange={(e) =>
                          onSetCompanyStrategies((prev) => ({
                            ...prev,
                            [i]: e.target.value,
                          }))
                        }
                        className="flex-1 text-xs bg-surface-0 border border-surface-200 rounded-lg px-2 py-1.5 text-surface-700 focus:border-accent focus:ring-1 focus:ring-accent/30 outline-none"
                      >
                        {CEO_STRATEGIES.map((s) => (
                          <option key={s.value} value={s.value}>
                            {s.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          <button
            onClick={onStart}
            className="w-full py-3 rounded-xl bg-accent text-white font-semibold text-sm transition-all hover:bg-accent/90 hover:shadow-lg hover:shadow-accent/20"
          >
            Start Simulation
          </button>
        </div>
      </div>

    </div>
  );
}
