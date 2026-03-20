import type { CompanyGraph } from "@/types/graph";
import type { CompanyProfile, OleoExport } from "@/types/profile";

const API_BASE = "/api/backend";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`API error ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

// --- Company parsing (existing) ---

export function parseCompany(message: string): Promise<CompanyGraph> {
  return request("/api/parse-company", {
    method: "POST",
    body: JSON.stringify({ message }),
  });
}

export function refineCompany(
  message: string,
  currentGraph: CompanyGraph,
): Promise<CompanyGraph> {
  return request("/api/refine-company", {
    method: "POST",
    body: JSON.stringify({ message, current_graph: currentGraph }),
  });
}

// --- Simulation (existing) ---

export function startSimulation(
  graph: CompanyGraph,
  maxTicks = 50,
  outlook = "normal",
): Promise<{ session_id: string }> {
  return request("/api/simulate/start", {
    method: "POST",
    body: JSON.stringify({ graph, max_ticks: maxTicks, outlook }),
  });
}

export function controlSimulation(
  sessionId: string,
  action: string,
  speed?: number,
  outlook?: string,
): Promise<{ status: string }> {
  return request(`/api/simulate/control/${sessionId}`, {
    method: "POST",
    body: JSON.stringify({ action, speed, outlook }),
  });
}

export function injectEvent(
  sessionId: string,
  description: string,
): Promise<{ status: string }> {
  return request(`/api/simulate/inject/${sessionId}`, {
    method: "POST",
    body: JSON.stringify({ description }),
  });
}

export function createSimulationStream(sessionId: string): EventSource {
  return new EventSource(`${API_BASE}/api/simulate/stream/${sessionId}`);
}

// --- Scenarios (existing) ---

export function forkScenario(
  sessionId: string,
): Promise<{ session_id: string }> {
  return request(`/api/scenarios/fork/${sessionId}`, { method: "POST" });
}

export function compareScenarios(
  sessionIds: string[],
): Promise<{ scenarios: Record<string, unknown> }> {
  return request(
    `/api/scenarios/compare?session_ids=${sessionIds.join(",")}`,
  );
}

// --- Settings (existing) ---

export function getStatus(): Promise<{ api_key_set: boolean }> {
  return request("/api/settings/status");
}

export function setApiKey(
  apiKey: string,
): Promise<{ api_key_set: boolean }> {
  return request("/api/settings/api-key", {
    method: "POST",
    body: JSON.stringify({ api_key: apiKey }),
  });
}

// --- Onboarding ---

interface OnboardingStartResponse {
  session_id: string;
  question: string;
  phase: string;
  fields_targeted: string[];
  completion_estimate: number;
  profile: CompanyProfile;
}

interface OnboardingRespondResponse {
  question: string;
  phase: string;
  fields_targeted: string[];
  completion_estimate: number;
  profile: CompanyProfile;
  graph: CompanyGraph | null;
}

interface OnboardingCompleteResponse {
  session_id: string;
  profile: CompanyProfile;
  graph: CompanyGraph;
}

interface QuickStartResponse {
  session_id: string;
  profile: CompanyProfile;
  graph: CompanyGraph;
}

export function startOnboarding(): Promise<OnboardingStartResponse> {
  return request("/api/onboarding/start", { method: "POST" });
}

export function respondOnboarding(
  sessionId: string,
  message: string,
): Promise<OnboardingRespondResponse> {
  return request("/api/onboarding/respond", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, message }),
  });
}

export function skipOnboardingPhase(
  sessionId: string,
  currentPhase: string,
): Promise<OnboardingRespondResponse> {
  return request("/api/onboarding/skip-phase", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      current_phase: currentPhase,
    }),
  });
}

export function completeOnboarding(
  sessionId: string,
): Promise<OnboardingCompleteResponse> {
  return request("/api/onboarding/complete", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId }),
  });
}

export function quickStart(
  description: string,
): Promise<QuickStartResponse> {
  return request("/api/onboarding/quick-start", {
    method: "POST",
    body: JSON.stringify({ description }),
  });
}

// --- Documents ---

export async function uploadDocument(
  file: File,
): Promise<{ job_id: string; status: string }> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/documents/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Upload error ${res.status}: ${detail}`);
  }
  return res.json();
}

export function getDocumentStatus(
  jobId: string,
): Promise<{ job_id: string; status: string; category: string | null }> {
  return request(`/api/documents/status/${jobId}`);
}

export function getDocumentResult(
  jobId: string,
): Promise<{
  job_id: string;
  category: string | null;
  extraction: Record<string, unknown> | null;
  profile_fields: Record<string, unknown> | null;
}> {
  return request(`/api/documents/result/${jobId}`);
}

export function confirmDocument(
  jobId: string,
  sessionId: string,
): Promise<{ profile: CompanyProfile; merged_fields: string[] }> {
  return request(`/api/documents/confirm/${jobId}`, {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId }),
  });
}

// --- EDGAR ---

export function edgarLookup(ticker: string): Promise<{
  company_info: { name: string; ticker: string; cik: string; sic: string; industry: string; state: string } | null;
  financials: Record<string, unknown> | null;
  profile_preview: CompanyProfile | null;
  error: string | null;
}> {
  return request(`/api/documents/edgar/${ticker}`);
}

export function edgarConfirm(
  sessionId: string,
  ticker: string,
): Promise<{ profile: CompanyProfile; merged_fields: string[] }> {
  return request("/api/documents/edgar/confirm", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, ticker }),
  });
}

// --- Export ---

export function exportProfile(
  sessionId: string,
): Promise<OleoExport> {
  return request("/api/export/profile", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId }),
  });
}

export function exportSimulation(
  sessionId: string,
  simulationSessionId?: string,
): Promise<OleoExport> {
  return request("/api/export/simulation", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      simulation_session_id: simulationSessionId,
    }),
  });
}

// --- Import ---

interface ValidateImportResponse {
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

export function validateImport(
  data: Record<string, unknown>,
): Promise<ValidateImportResponse> {
  return request("/api/import/validate", {
    method: "POST",
    body: JSON.stringify({ data }),
  });
}

export function loadImport(
  data: Record<string, unknown>,
  targetSessionId?: string | null,
  mode = "replace",
): Promise<{
  session_id: string;
  profile: CompanyProfile;
  graph: CompanyGraph;
}> {
  return request("/api/import/load", {
    method: "POST",
    body: JSON.stringify({
      data,
      target_session_id: targetSessionId,
      mode,
    }),
  });
}
