"use client";

import { ArrowLeft, Loader2, Mic, Sparkles } from "lucide-react";
import { Logo } from "@/components/ui/Logo";
import type { UseAdaptiveSetupReturn } from "@/hooks/useAdaptiveSetup";

const DESCRIPTION_GUIDE_TOPICS = [
  "Pricing", "Costs", "Customers", "Team", "Revenue", "Operations", "Growth",
];

const ADAPTIVE_START_MODES = [
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

interface AdaptiveSetupProps {
  adaptive: UseAdaptiveSetupReturn;
  onBack: () => void;
  onStartSim: () => void;
}

export function AdaptiveSetup({ adaptive, onBack, onStartSim }: AdaptiveSetupProps) {
  const adaptiveDurationOptions = adaptive.specDisplay?.duration_options ?? [1, 5, 10, 20];
  const totalCompanies = adaptive.numCompetitors + 1;
  const descriptionTrimmed = adaptive.companyDescription.trim();
  const wordCount = descriptionTrimmed ? descriptionTrimmed.split(/\s+/).length : 0;

  return (
    <div className="h-screen flex flex-col bg-surface-0 phase-enter">
      <header className="flex items-center gap-3 px-5 py-3 border-b border-surface-200/50">
        <button
          onClick={() => {
            if (adaptive.step === 1) {
              onBack();
            } else {
              adaptive.backToStep((adaptive.step - 1) as 1 | 2);
            }
          }}
          className="flex items-center justify-center w-8 h-8 rounded-lg hover:bg-surface-100 transition-colors"
        >
          <ArrowLeft size={16} className="text-surface-500" />
        </button>
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-br from-surface-100 to-surface-50 border border-surface-200/50">
          <Logo size={16} className="text-accent" />
        </div>
        <div>
          <h1 className="text-sm font-bold text-surface-900 tracking-wide">
            Custom Adaptive
          </h1>
          <p className="text-[10px] text-surface-500 uppercase tracking-widest">
            Step {adaptive.step} of 3
          </p>
        </div>
      </header>

      <div className="flex-1 flex flex-col items-center justify-center px-6 py-12">
        {/* Step 1: Describe your company */}
        {adaptive.step === 1 && (
          <div className="max-w-lg w-full space-y-5">
            <div>
              <h2 className="text-lg font-semibold text-surface-800 mb-1">
                Tell Us Everything
              </h2>
              <p className="text-xs text-surface-500 leading-relaxed">
                The more detail you provide, the more accurate your simulation will be.
                Don't hold back — every detail about pricing, costs, customers, and operations helps.
              </p>
            </div>

            <div>
              <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block">
                Company Name
              </label>
              <input
                type="text"
                value={adaptive.companyName}
                onChange={(e) => adaptive.setCompanyName(e.target.value)}
                placeholder="e.g. Acme Coffee Roasters"
                className="w-full px-3 py-2.5 rounded-lg border border-surface-200 bg-surface-0 text-sm text-surface-800 placeholder:text-surface-400 focus:border-accent focus:ring-1 focus:ring-accent/30 outline-none"
              />
            </div>

            <div>
              <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block">
                Describe Your Business
              </label>

              <div className="p-3 rounded-lg bg-accent/5 border border-accent/20 mb-3 space-y-2">
                <div className="flex items-start gap-2">
                  <Mic size={14} className="text-accent mt-0.5 flex-shrink-0" />
                  <p className="text-xs text-surface-600 leading-relaxed">
                    <span className="font-semibold text-surface-700">Try voice typing</span>{" "}
                    — press <kbd className="px-1 py-0.5 rounded bg-surface-200 text-[10px] font-mono">Win+H</kbd>{" "}
                    or <kbd className="px-1 py-0.5 rounded bg-surface-200 text-[10px] font-mono">Fn Fn</kbd>{" "}
                    and just talk. It's the fastest way to share detail.
                  </p>
                </div>
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {DESCRIPTION_GUIDE_TOPICS.map((topic) => (
                    <span
                      key={topic}
                      className="px-2 py-0.5 rounded-full bg-surface-0 border border-surface-200 text-[10px] font-medium text-surface-500"
                    >
                      {topic}
                    </span>
                  ))}
                </div>
              </div>

              <textarea
                value={adaptive.companyDescription}
                onChange={(e) => adaptive.setCompanyDescription(e.target.value)}
                placeholder="Describe your business in as much detail as possible — what you sell, pricing, costs, team, customers, how you grow..."
                rows={10}
                className="w-full px-3 py-2.5 rounded-lg border border-surface-200 bg-surface-0 text-sm text-surface-800 placeholder:text-surface-400 focus:border-accent focus:ring-1 focus:ring-accent/30 outline-none resize-none min-h-[200px]"
              />

              <div className="flex justify-end mt-1.5">
                <span className={`text-xs ${
                  wordCount < 20
                    ? "text-surface-400"
                    : wordCount < 50
                      ? "text-amber-500"
                      : "text-emerald-500"
                }`}>
                  {wordCount} {wordCount === 1 ? "word" : "words"}
                  {wordCount > 0 && wordCount < 20 && " — keep going, more detail = better simulation"}
                  {wordCount >= 50 && " — great level of detail"}
                </span>
              </div>
            </div>

            {adaptive.error && (
              <p className="text-xs text-red-500">{adaptive.error}</p>
            )}

            <button
              onClick={adaptive.analyze}
              disabled={adaptive.isAnalyzing || !adaptive.companyName.trim() || wordCount < 20}
              className="w-full py-3 rounded-xl bg-accent text-white font-semibold text-sm transition-all hover:bg-accent/90 hover:shadow-lg hover:shadow-accent/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {adaptive.isAnalyzing ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles size={16} />
                  Analyze My Business
                </>
              )}
            </button>
          </div>
        )}

        {/* Step 2: Confirm niche */}
        {adaptive.step === 2 && adaptive.nicheAnalysis && (
          <div className="max-w-md w-full space-y-5">
            <div>
              <h2 className="text-lg font-semibold text-surface-800 mb-1">
                Confirm Your Niche
              </h2>
              <p className="text-xs text-surface-500">
                We identified the following about your business
              </p>
            </div>

            <div className="p-5 rounded-xl border border-accent/20 bg-accent/5 space-y-3">
              <div>
                <span className="text-[10px] font-semibold text-accent uppercase tracking-wider">
                  Niche
                </span>
                <p className="text-sm font-medium text-surface-800 mt-0.5">
                  {adaptive.nicheAnalysis.niche}
                </p>
              </div>
              <div>
                <span className="text-[10px] font-semibold text-accent uppercase tracking-wider">
                  Business Model
                </span>
                <p className="text-xs text-surface-600 mt-0.5 leading-relaxed">
                  {adaptive.nicheAnalysis.summary}
                </p>
              </div>
              <div>
                <span className="text-[10px] font-semibold text-accent uppercase tracking-wider">
                  Economics Type
                </span>
                <p className="text-xs text-surface-600 mt-0.5 capitalize">
                  {adaptive.nicheAnalysis.economics_model}
                </p>
              </div>
            </div>

            {adaptive.error && (
              <p className="text-xs text-red-500">{adaptive.error}</p>
            )}

            <div className="flex gap-3">
              <button
                onClick={() => adaptive.backToStep(1)}
                className="flex-1 py-3 rounded-xl border border-surface-200 text-surface-700 font-semibold text-sm transition-all hover:border-surface-300"
              >
                Edit
              </button>
              <button
                onClick={adaptive.confirmNiche}
                className="flex-1 py-3 rounded-xl bg-accent text-white font-semibold text-sm transition-all hover:bg-accent/90 hover:shadow-lg hover:shadow-accent/20"
              >
                Confirm
              </button>
            </div>
          </div>
        )}

        {/* Step 3: Configure simulation */}
        {adaptive.step === 3 && (
          <div className="max-w-md w-full space-y-6">
            <div>
              <h2 className="text-lg font-semibold text-surface-800 mb-1">
                Configure Simulation
              </h2>
              <p className="text-xs text-surface-500">
                Set up the competitive landscape for{" "}
                <span className="font-medium text-surface-700">{adaptive.companyName}</span>
              </p>
            </div>

            <div>
              <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block">
                Starting Conditions
              </label>
              <div className="space-y-2">
                {ADAPTIVE_START_MODES.map((m) => (
                  <button
                    key={m.value}
                    onClick={() => adaptive.setStartMode(m.value)}
                    className={`w-full flex items-center gap-3 p-3 rounded-lg border transition-all text-left ${
                      adaptive.startMode === m.value
                        ? "border-accent bg-accent/5 ring-1 ring-accent/30"
                        : "border-surface-200 hover:border-surface-300"
                    }`}
                  >
                    <div
                      className={`w-3 h-3 rounded-full border-2 flex-shrink-0 ${
                        adaptive.startMode === m.value
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
                Competitors ({adaptive.numCompetitors})
              </label>
              <input
                type="range"
                min={1}
                max={19}
                value={adaptive.numCompetitors}
                onChange={(e) => adaptive.setNumCompetitors(Number(e.target.value))}
                className="w-full accent-accent"
              />
              <div className="flex justify-between text-[10px] text-surface-400 mt-1">
                <span>1</span>
                <span>19</span>
              </div>
            </div>

            <div>
              <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block">
                AI CEO Agents
              </label>
              <button
                onClick={() => adaptive.setAiCeoEnabled(!adaptive.aiCeoEnabled)}
                className={`w-full flex items-center justify-between p-3 rounded-lg border transition-all ${
                  adaptive.aiCeoEnabled
                    ? "border-accent bg-accent/5 ring-1 ring-accent/30"
                    : "border-surface-200 hover:border-surface-300"
                }`}
              >
                <div className="text-left">
                  <div className="text-sm font-medium text-surface-800">
                    {adaptive.aiCeoEnabled ? "Enabled" : "Disabled"}
                  </div>
                  <div className="text-[11px] text-surface-500">
                    Each company gets a Claude AI CEO making strategic decisions
                  </div>
                </div>
                <div
                  className={`w-10 h-6 rounded-full transition-all flex items-center ${
                    adaptive.aiCeoEnabled ? "bg-accent justify-end" : "bg-surface-300 justify-start"
                  }`}
                >
                  <div className="w-4 h-4 bg-white rounded-full mx-1" />
                </div>
              </button>
            </div>

            {adaptive.aiCeoEnabled && (
              <>
                <div>
                  <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2 block">
                    Duration
                  </label>
                  <div className="flex gap-2">
                    {adaptiveDurationOptions.map((y) => (
                      <button
                        key={y}
                        onClick={() => adaptive.setDurationYears(y)}
                        className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${
                          adaptive.durationYears === y
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
                    {Array.from({ length: totalCompanies }, (_, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <span className={`text-xs w-24 truncate flex-shrink-0 ${
                          i === 0 ? "text-accent font-semibold" : "text-surface-500"
                        }`}>
                          {i === 0 ? adaptive.companyName : `Competitor ${i}`}
                        </span>
                        <select
                          value={adaptive.companyStrategies[i] || "balanced"}
                          onChange={(e) =>
                            adaptive.setCompanyStrategies((prev) => ({
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

            {adaptive.error && (
              <p className="text-xs text-red-500">{adaptive.error}</p>
            )}

            <button
              onClick={async () => {
                const ok = await adaptive.startSimulation();
                if (ok) onStartSim();
              }}
              disabled={adaptive.isGenerating}
              className="w-full py-3 rounded-xl bg-accent text-white font-semibold text-sm transition-all hover:bg-accent/90 hover:shadow-lg hover:shadow-accent/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {adaptive.isGenerating ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Generating Your Industry...
                </>
              ) : (
                "Start Simulation"
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
