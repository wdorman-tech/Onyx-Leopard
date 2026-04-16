"use client";

import { useCallback, useState } from "react";
import { Check, Loader2 } from "lucide-react";

interface ProfileSummaryProps {
  spec: Record<string, unknown>;
  isLoading: boolean;
  error: string | null;
  onConfirm: (slug: string) => void;
}

export function ProfileSummary({
  spec,
  isLoading,
  error,
  onConfirm,
}: ProfileSummaryProps) {
  const meta = spec.meta as Record<string, unknown> | undefined;
  const nodes = spec.nodes as Record<string, unknown> | undefined;
  const triggers = spec.triggers as unknown[] | undefined;
  const locationDefaults = spec.location_defaults as Record<string, unknown> | undefined;

  const suggestedSlug = (meta?.slug as string) || "custom_business";
  const [slug, setSlug] = useState(suggestedSlug);

  const handleConfirm = useCallback(() => {
    onConfirm(slug);
  }, [slug, onConfirm]);

  return (
    <div className="flex flex-col h-full overflow-y-auto px-6 py-6 space-y-6">
      <div>
        <h2 className="text-lg font-bold text-surface-900 mb-1">
          {(meta?.name as string) || "Your Business"}
        </h2>
        <p className="text-sm text-surface-500">
          {(meta?.description as string) || "Custom business simulation profile"}
        </p>
      </div>

      {/* Key stats */}
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-surface-50 rounded-xl p-3 border border-surface-200/50">
          <div className="text-lg font-bold text-surface-900">
            {nodes ? Object.keys(nodes).length : 0}
          </div>
          <div className="text-[11px] text-surface-500 uppercase tracking-wider">
            Node Types
          </div>
        </div>
        <div className="bg-surface-50 rounded-xl p-3 border border-surface-200/50">
          <div className="text-lg font-bold text-surface-900">
            {triggers ? triggers.length : 0}
          </div>
          <div className="text-[11px] text-surface-500 uppercase tracking-wider">
            Growth Triggers
          </div>
        </div>
        <div className="bg-surface-50 rounded-xl p-3 border border-surface-200/50">
          <div className="text-lg font-bold text-surface-900 capitalize">
            {(locationDefaults?.economics_model as string) || "physical"}
          </div>
          <div className="text-[11px] text-surface-500 uppercase tracking-wider">
            Model
          </div>
        </div>
      </div>

      {/* Node categories */}
      {nodes && (
        <div>
          <h3 className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-2">
            Organization Structure
          </h3>
          <div className="space-y-1">
            {Object.entries(nodes).slice(0, 12).map(([key, val]) => {
              const node = val as Record<string, unknown>;
              return (
                <div
                  key={key}
                  className="flex items-center justify-between py-1.5 px-3 bg-surface-50 rounded-lg text-sm"
                >
                  <span className="text-surface-800">{node.label as string}</span>
                  <span className="text-[11px] text-surface-400 capitalize">
                    {node.category as string}
                  </span>
                </div>
              );
            })}
            {Object.keys(nodes).length > 12 && (
              <div className="text-[11px] text-surface-400 text-center py-1">
                +{Object.keys(nodes).length - 12} more nodes
              </div>
            )}
          </div>
        </div>
      )}

      {/* Slug input + confirm */}
      <div className="mt-auto pt-4 border-t border-surface-200/50 space-y-3">
        <div>
          <label className="text-xs font-semibold text-surface-600 uppercase tracking-wider mb-1.5 block">
            Industry ID
          </label>
          <input
            type="text"
            value={slug}
            onChange={(e) =>
              setSlug(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, "_"))
            }
            className="w-full bg-surface-50 border border-surface-200 rounded-xl px-4 py-2.5 text-sm text-surface-800 outline-none focus:border-accent focus:ring-1 focus:ring-accent/30"
          />
          <p className="text-[11px] text-surface-400 mt-1">
            Lowercase letters, numbers, and underscores only
          </p>
        </div>

        {error && (
          <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
            {error}
          </div>
        )}

        <button
          onClick={handleConfirm}
          disabled={isLoading || !slug}
          className="w-full py-3 rounded-xl bg-accent text-white font-semibold text-sm transition-all hover:bg-accent/90 hover:shadow-lg hover:shadow-accent/20 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              Saving...
            </>
          ) : (
            <>
              <Check size={16} />
              Save & Start Simulation
            </>
          )}
        </button>
      </div>
    </div>
  );
}
