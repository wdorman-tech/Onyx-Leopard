"use client";

import { useState, useCallback } from "react";
import type { CompanyProfile } from "@/types/profile";
import { edgarLookup, edgarConfirm } from "@/lib/api";
import { Spinner } from "@/components/ui/Spinner";
import { Search, Check } from "@/components/ui/icons";

interface EdgarLookupProps {
  sessionId: string;
  onProfileUpdate: (profile: CompanyProfile) => void;
}

interface LookupResult {
  company_info: {
    name: string;
    ticker: string;
    industry: string;
  } | null;
  financials: Record<string, unknown> | null;
  error: string | null;
}

export function EdgarLookup({
  sessionId,
  onProfileUpdate,
}: EdgarLookupProps) {
  const [ticker, setTicker] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<LookupResult | null>(null);
  const [confirmed, setConfirmed] = useState(false);

  const handleSearch = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!ticker.trim()) return;

      setLoading(true);
      setResult(null);
      setConfirmed(false);

      try {
        const res = await edgarLookup(ticker.trim().toUpperCase());
        setResult({
          company_info: res.company_info ?? null,
          financials: res.financials ?? null,
          error: res.error ?? null,
        });
      } catch (err) {
        setResult({
          company_info: null,
          financials: null,
          error: err instanceof Error ? err.message : "Lookup failed",
        });
      } finally {
        setLoading(false);
      }
    },
    [ticker],
  );

  const handleConfirm = useCallback(async () => {
    setLoading(true);
    try {
      const res = await edgarConfirm(sessionId, ticker.trim().toUpperCase());
      onProfileUpdate(res.profile);
      setConfirmed(true);
    } catch (err) {
      setResult((prev) =>
        prev
          ? {
              ...prev,
              error: err instanceof Error ? err.message : "Confirm failed",
            }
          : null,
      );
    } finally {
      setLoading(false);
    }
  }, [sessionId, ticker, onProfileUpdate]);

  return (
    <div className="space-y-3">
      <form onSubmit={handleSearch} className="flex gap-2">
        <input
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          placeholder="Enter ticker (e.g., AAPL)"
          className="flex-1 bg-surface-50 border border-surface-200 rounded-lg px-3 py-2 text-sm text-surface-800 placeholder:text-surface-500 focus:outline-none focus:ring-2 focus:ring-accent/30 focus:border-accent/40 font-mono transition-all"
          maxLength={10}
        />
        <button
          type="submit"
          disabled={!ticker.trim() || loading}
          className="btn-secondary px-4 py-2 text-sm flex items-center gap-1.5"
        >
          {loading ? (
            <Spinner className="h-4 w-4" />
          ) : (
            <>
              <Search size={14} />
              Search
            </>
          )}
        </button>
      </form>

      {result?.error && (
        <div className="text-xs text-negative bg-negative/10 px-3 py-2 rounded-lg">
          {result.error}
        </div>
      )}

      {result?.company_info && (
        <div className="bg-surface-50/50 border border-surface-200 rounded-xl p-4 space-y-3">
          <div>
            <div className="text-sm font-medium text-surface-800">
              {result.company_info.name}
            </div>
            <div className="text-xs text-surface-500">
              {result.company_info.ticker}
              {result.company_info.industry &&
                ` · ${result.company_info.industry}`}
            </div>
          </div>

          {result.financials && (
            <div className="space-y-1">
              <div className="text-[10px] text-surface-500 font-medium uppercase tracking-wider">
                10-K Financials
              </div>
              <div className="bg-surface-0/50 rounded-lg p-2 max-h-48 overflow-y-auto">
                {Object.entries(result.financials)
                  .filter(
                    ([k]) =>
                      k !== "operating_expenses" && typeof result.financials![k] !== "object",
                  )
                  .map(([key, val]) => (
                    <div
                      key={key}
                      className="flex justify-between text-[11px] py-0.5"
                    >
                      <span className="text-surface-500">
                        {key.replace(/_/g, " ")}
                      </span>
                      <span className="text-surface-700 font-mono">
                        {typeof val === "number"
                          ? val > 1000000
                            ? `$${(val / 1000000).toFixed(1)}M`
                            : val < 1 && val > 0
                              ? `${(val * 100).toFixed(1)}%`
                              : `$${val.toLocaleString()}`
                          : String(val)}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {confirmed ? (
            <div className="text-xs text-accent bg-accent/10 px-3 py-2 rounded-lg text-center flex items-center justify-center gap-1.5">
              <Check size={12} />
              Data merged into profile
            </div>
          ) : (
            <button
              onClick={handleConfirm}
              disabled={loading || !result.financials}
              className="btn-primary w-full text-xs py-2.5"
            >
              Confirm & Merge into Profile
            </button>
          )}
        </div>
      )}

      <div className="text-[10px] text-surface-500">
        Data sourced from SEC EDGAR. No API key required.
      </div>
    </div>
  );
}
