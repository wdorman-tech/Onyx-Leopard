"use client";

import { useState, useRef, useCallback } from "react";
import type { CompanyProfile } from "@/types/profile";
import { uploadDocument, getDocumentResult, confirmDocument } from "@/lib/api";
import { Upload, Check } from "@/components/ui/icons";

interface DocumentUploadProps {
  sessionId: string;
  onProfileUpdate: (profile: CompanyProfile) => void;
}

interface UploadState {
  fileName: string;
  jobId: string | null;
  status: "uploading" | "processing" | "done" | "error";
  category: string | null;
  extraction: Record<string, unknown> | null;
  error: string | null;
}

export function DocumentUpload({
  sessionId,
  onProfileUpdate,
}: DocumentUploadProps) {
  const [uploads, setUploads] = useState<UploadState[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    async (files: FileList) => {
      for (const file of Array.from(files)) {
        const uploadId = crypto.randomUUID();
        const upload: UploadState = {
          fileName: file.name,
          jobId: uploadId,
          status: "uploading",
          category: null,
          extraction: null,
          error: null,
        };
        setUploads((prev) => [...prev, upload]);

        let currentJobId = uploadId;
        try {
          const res = await uploadDocument(file);
          currentJobId = res.job_id;
          setUploads((prev) =>
            prev.map((u) =>
              u.jobId === uploadId
                ? { ...u, jobId: res.job_id, status: "processing" }
                : u,
            ),
          );

          const result = await getDocumentResult(res.job_id);
          setUploads((prev) =>
            prev.map((u) =>
              u.jobId === res.job_id
                ? {
                    ...u,
                    status: "done",
                    category: result.category ?? null,
                    extraction: result.extraction ?? null,
                  }
                : u,
            ),
          );
        } catch (err) {
          setUploads((prev) =>
            prev.map((u) =>
              u.jobId === currentJobId || u.jobId === uploadId
                ? {
                    ...u,
                    status: "error",
                    error:
                      err instanceof Error ? err.message : "Upload failed",
                  }
                : u,
            ),
          );
        }
      }
    },
    [],
  );

  const handleConfirm = useCallback(
    async (jobId: string) => {
      try {
        const result = await confirmDocument(jobId, sessionId);
        onProfileUpdate(result.profile);
        setUploads((prev) => prev.filter((u) => u.jobId !== jobId));
      } catch (err) {
        setUploads((prev) =>
          prev.map((u) =>
            u.jobId === jobId
              ? {
                  ...u,
                  error:
                    err instanceof Error ? err.message : "Confirm failed",
                }
              : u,
          ),
        );
      }
    },
    [sessionId, onProfileUpdate],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (e.dataTransfer.files.length > 0) {
        handleFiles(e.dataTransfer.files);
      }
    },
    [handleFiles],
  );

  return (
    <div className="space-y-3">
      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => fileInputRef.current?.click()}
        className="border-2 border-dashed border-surface-200 hover:border-surface-300 rounded-xl p-6 text-center cursor-pointer transition-colors"
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.xlsx,.csv,.png,.jpg,.jpeg"
          onChange={(e) => e.target.files && handleFiles(e.target.files)}
          className="hidden"
        />
        <Upload size={24} className="mx-auto text-surface-500 mb-2" />
        <div className="text-xs text-surface-500">
          Drop files here or click to browse
        </div>
        <div className="text-[10px] text-surface-500 mt-1">
          PDF, Excel, CSV, or images up to 10MB
        </div>
      </div>

      {/* Upload items */}
      {uploads.map((upload, i) => (
        <div
          key={i}
          className="bg-surface-50/50 border border-surface-200 rounded-xl p-3"
        >
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-surface-700 font-medium truncate max-w-[200px]">
              {upload.fileName}
            </span>
            <span
              className={`text-[10px] px-2 py-0.5 rounded-full ${
                upload.status === "done"
                  ? "bg-accent/10 text-accent"
                  : upload.status === "error"
                    ? "bg-negative/10 text-negative"
                    : "bg-surface-200 text-surface-500"
              }`}
            >
              {upload.status === "uploading"
                ? "Uploading..."
                : upload.status === "processing"
                  ? "Processing..."
                  : upload.status === "done"
                    ? upload.category
                    : "Error"}
            </span>
          </div>

          {upload.error && (
            <div className="text-xs text-negative/80 mt-1">{upload.error}</div>
          )}

          {upload.status === "done" && upload.extraction && (
            <div className="mt-2 space-y-2">
              <div className="text-[10px] text-surface-500 font-medium uppercase tracking-wider">
                Extracted Data
              </div>
              <div className="bg-surface-0/50 rounded-lg p-2 max-h-32 overflow-y-auto">
                {Object.entries(upload.extraction).map(([key, val]) => (
                  <div
                    key={key}
                    className="flex justify-between text-[11px] py-0.5"
                  >
                    <span className="text-surface-500">{key}</span>
                    <span className="text-surface-700 font-mono">
                      {typeof val === "object"
                        ? JSON.stringify(val)
                        : String(val)}
                    </span>
                  </div>
                ))}
              </div>
              {upload.jobId && (
                <button
                  onClick={() => handleConfirm(upload.jobId!)}
                  className="btn-primary w-full text-xs py-2 flex items-center justify-center gap-1.5"
                >
                  <Check size={12} />
                  Confirm & Merge into Profile
                </button>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
