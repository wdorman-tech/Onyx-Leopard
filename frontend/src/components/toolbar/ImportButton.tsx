"use client";

import { useState, useRef, useCallback } from "react";
import type { CompanyProfile } from "@/types/profile";
import type { CompanyGraph } from "@/types/graph";
import { validateImport, loadImport } from "@/lib/api";
import { Upload } from "@/components/ui/icons";

interface ImportButtonProps {
  sessionId: string | null;
  onImport: (profile: CompanyProfile, graph: CompanyGraph, sessionId: string) => void;
}

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  needs_migration: boolean;
  preview: {
    name: string;
    industry: string;
    stage: string;
    node_count: number;
    has_simulation: boolean;
  } | null;
}

export function ImportButton({ sessionId, onImport }: ImportButtonProps) {
  const [showModal, setShowModal] = useState(false);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [fileData, setFileData] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      setError(null);
      setValidation(null);
      setLoading(true);

      try {
        const text = await file.text();
        let data: Record<string, unknown>;
        try {
          data = JSON.parse(text);
        } catch {
          setError("Invalid JSON file");
          setLoading(false);
          return;
        }

        setFileData(data);
        const result = await validateImport(data);
        setValidation(result);
        setShowModal(true);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to validate file");
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const handleConfirm = useCallback(async () => {
    if (!fileData) return;

    setLoading(true);
    try {
      const result = await loadImport(fileData, sessionId, "replace");
      onImport(result.profile, result.graph, result.session_id);
      setShowModal(false);
      setValidation(null);
      setFileData(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load import");
    } finally {
      setLoading(false);
    }
  }, [fileData, sessionId, onImport]);

  const handleClose = useCallback(() => {
    setShowModal(false);
    setValidation(null);
    setFileData(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept=".oleo,.json"
        onChange={handleFileSelect}
        className="hidden"
      />
      <button
        onClick={() => fileInputRef.current?.click()}
        disabled={loading}
        className="btn-ghost flex items-center gap-1.5 text-xs px-2.5 py-1.5"
        data-tooltip="Import .oleo file"
      >
        <Upload size={14} />
        Import
      </button>

      {showModal && validation && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="bg-surface-50 border border-surface-200 rounded-2xl p-6 w-[420px] shadow-2xl">
            <h2 className="text-sm font-semibold text-surface-900 mb-4">
              Import .oleo File
            </h2>

            {!validation.valid ? (
              <div className="space-y-2">
                <div className="text-xs font-medium text-negative">
                  Validation failed:
                </div>
                {validation.errors.map((err, i) => (
                  <div
                    key={i}
                    className="text-xs text-negative/80 bg-negative/10 px-3 py-2 rounded-lg"
                  >
                    {err}
                  </div>
                ))}
              </div>
            ) : (
              <>
                {validation.preview && (
                  <div className="bg-surface-100/50 rounded-xl p-4 mb-4 space-y-1.5">
                    <div className="text-sm font-medium text-surface-800">
                      {validation.preview.name}
                    </div>
                    {validation.preview.industry && (
                      <div className="text-xs text-surface-500">
                        {validation.preview.industry}
                        {validation.preview.stage &&
                          ` · ${validation.preview.stage}`}
                      </div>
                    )}
                    <div className="text-xs text-surface-500">
                      {validation.preview.node_count} nodes
                      {validation.preview.has_simulation &&
                        " · includes simulation snapshot"}
                    </div>
                  </div>
                )}

                {validation.warnings.length > 0 && (
                  <div className="space-y-1.5 mb-4">
                    {validation.warnings.map((w, i) => (
                      <div
                        key={i}
                        className="text-xs text-warning/80 bg-warning/10 px-3 py-2 rounded-lg"
                      >
                        {w}
                      </div>
                    ))}
                  </div>
                )}

                {sessionId && (
                  <div className="text-xs text-surface-500 mb-4">
                    This will replace your current profile and graph.
                  </div>
                )}

                <button
                  onClick={handleConfirm}
                  disabled={loading}
                  className="btn-primary w-full"
                >
                  {loading ? "Loading..." : "Load Profile"}
                </button>
              </>
            )}

            {error && (
              <div className="mt-3 text-xs text-negative">{error}</div>
            )}

            <button
              onClick={handleClose}
              className="btn-ghost mt-3 w-full text-xs py-2"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </>
  );
}
