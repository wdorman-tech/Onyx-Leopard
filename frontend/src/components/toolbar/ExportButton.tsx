"use client";

import { useState, useCallback } from "react";
import type { CompanyProfile, OleoExport } from "@/types/profile";
import type { CompanyGraph } from "@/types/graph";
import { Download } from "@/components/ui/icons";

interface ExportButtonProps {
  sessionId: string | null;
  simulationSessionId: string | null;
  profile: CompanyProfile | null;
  graph: CompanyGraph | null;
}

export function ExportButton({
  sessionId,
  simulationSessionId,
  profile,
  graph,
}: ExportButtonProps) {
  const [showModal, setShowModal] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const downloadBlob = useCallback(
    (data: OleoExport) => {
      const payload = JSON.stringify(data, null, 2);
      const blob = new Blob([payload], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const name = data.profile?.identity?.name || "company";
      const safeName = name
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .slice(0, 30);
      a.download = `${safeName}.oleo`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    },
    [],
  );

  const handleExport = useCallback(
    async (includeSimulation: boolean) => {
      if (!profile || !graph) return;

      setExporting(true);
      setError(null);
      try {
        if (sessionId) {
          const { exportProfile, exportSimulation } = await import("@/lib/api");
          const data =
            includeSimulation && simulationSessionId
              ? await exportSimulation(sessionId, simulationSessionId)
              : await exportProfile(sessionId);
          downloadBlob(data);
        } else {
          const exportData: OleoExport = {
            format: "onyx-leopard-export",
            format_version: "1.0.0",
            exported_at: new Date().toISOString(),
            profile,
            graph,
            simulation_snapshot: null,
            checksum: "",
          };
          downloadBlob(exportData);
        }
        setShowModal(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Export failed");
      } finally {
        setExporting(false);
      }
    },
    [profile, graph, sessionId, simulationSessionId, downloadBlob],
  );

  const disabled = !profile || !graph;

  return (
    <>
      <button
        onClick={() => setShowModal(true)}
        disabled={disabled}
        className="btn-ghost flex items-center gap-1.5 text-xs px-2.5 py-1.5"
        data-tooltip="Export simulation"
      >
        <Download size={14} />
        Export
      </button>

      {showModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="bg-surface-50 border border-surface-200 rounded-2xl p-6 w-[380px] shadow-2xl">
            <h2 className="text-sm font-semibold text-surface-900 mb-1">
              Export Simulation
            </h2>
            <p className="text-xs text-surface-500 mb-5">
              Download as .oleo file for sharing or backup.
            </p>

            <div className="space-y-2.5">
              <button
                onClick={() => handleExport(false)}
                disabled={exporting}
                className="w-full text-left px-4 py-3 rounded-xl border border-surface-200 hover:border-surface-300 hover:bg-surface-100/50 transition-all group active:scale-[0.99]"
              >
                <div className="text-sm font-medium text-surface-800 group-hover:text-surface-900">
                  Profile + Graph Only
                </div>
                <div className="text-xs text-surface-500 mt-0.5">
                  Company data and structure, no simulation history.
                </div>
              </button>

              <button
                onClick={() => handleExport(true)}
                disabled={exporting || !simulationSessionId}
                className="w-full text-left px-4 py-3 rounded-xl border border-surface-200 hover:border-surface-300 hover:bg-surface-100/50 transition-all group disabled:opacity-40 disabled:cursor-not-allowed active:scale-[0.99]"
              >
                <div className="text-sm font-medium text-surface-800 group-hover:text-surface-900">
                  With Simulation Snapshot
                </div>
                <div className="text-xs text-surface-500 mt-0.5">
                  Includes current tick, metrics, and recent history.
                </div>
              </button>
            </div>

            {error && (
              <div className="mt-3 text-xs text-negative bg-negative/10 px-3 py-2 rounded-lg">
                {error}
              </div>
            )}

            <button
              onClick={() => { setShowModal(false); setError(null); }}
              className="btn-ghost mt-4 w-full text-xs py-2"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </>
  );
}
