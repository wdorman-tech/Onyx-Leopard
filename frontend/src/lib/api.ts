const API_BASE = "/api/backend";
const SSE_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

export interface Industry {
  slug: string;
  name: string;
  description: string;
  icon: string;
  playable: boolean;
  total_nodes: number;
  growth_stages: number;
  key_metrics: string[];
  example_nodes: string[];
  categories: Record<string, number>;
}

export function getIndustries(): Promise<Industry[]> {
  return request("/api/simulate/industries");
}

export function startSimulation(
  maxTicks = 0,
  industry = "restaurant",
): Promise<{ session_id: string }> {
  return request("/api/simulate/start", {
    method: "POST",
    body: JSON.stringify({ max_ticks: maxTicks, industry }),
  });
}

export type SimulationAction = "play" | "pause" | "set_speed" | "focus_company";

export function controlSimulation(
  sessionId: string,
  action: SimulationAction,
  speed?: number,
): Promise<{ status: string }> {
  return request(`/api/simulate/control/${sessionId}`, {
    method: "POST",
    body: JSON.stringify({ action, speed }),
  });
}

export function createSimulationStream(sessionId: string): EventSource {
  return new EventSource(`${SSE_BASE}/api/simulate/stream/${sessionId}`);
}

// ── Market simulation API ──

export interface MarketPresetResponse {
  slug: string;
  name: string;
  description: string;
  alpha: number;
  beta: number;
  delta: number;
  n_0: number;
}

export function getMarketPresets(): Promise<MarketPresetResponse[]> {
  return request("/api/simulate/market/presets");
}

export function startMarketSimulation(
  preset: string,
  maxTicks = 0,
): Promise<{ session_id: string }> {
  return request("/api/simulate/start", {
    method: "POST",
    body: JSON.stringify({ mode: "market", preset, max_ticks: maxTicks }),
  });
}

// ── Unified simulation API ──

export interface SpecDisplay {
  stage_labels: Record<string, string>;
  event_noise_filters: string[];
  duration_options: number[];
}

export interface UnifiedStartResponse {
  session_id: string;
  spec_display?: SpecDisplay;
  founder_type?: string;
}

export function startUnifiedSimulation(
  startMode = "identical",
  numCompanies = 4,
  maxTicks = 0,
  aiCeoEnabled = false,
  durationYears = 5,
  companyStrategies: Record<number, string> = {},
): Promise<UnifiedStartResponse> {
  return request("/api/simulate/start", {
    method: "POST",
    body: JSON.stringify({
      mode: "unified",
      start_mode: startMode,
      num_companies: numCompanies,
      max_ticks: maxTicks,
      ai_ceo_enabled: aiCeoEnabled,
      duration_years: durationYears,
      company_strategies: companyStrategies,
    }),
  });
}

export function focusCompany(
  sessionId: string,
  companyId: string,
): Promise<{ status: string }> {
  return request(`/api/simulate/control/${sessionId}`, {
    method: "POST",
    body: JSON.stringify({ action: "focus_company", company_id: companyId }),
  });
}

// ── Profile Builder API ──

export interface ProfileStartResponse {
  session_id: string;
  first_question: string;
}

export interface ProfileAnswerResponse {
  next_question: string;
  progress: number;
  is_complete: boolean;
  industry_spec?: Record<string, unknown>;
  error?: string;
}

export interface ProfileSessionResponse {
  session_id: string;
  status: string;
  transcript: Array<{ role: string; content: string }>;
  industry_spec: Record<string, unknown> | null;
  error: string | null;
}

export function startProfile(): Promise<ProfileStartResponse> {
  return request("/api/profile/start", { method: "POST" });
}

export function answerProfile(
  sessionId: string,
  answer: string,
): Promise<ProfileAnswerResponse> {
  return request(`/api/profile/${sessionId}/answer`, {
    method: "POST",
    body: JSON.stringify({ answer }),
  });
}

export function getProfileSession(
  sessionId: string,
): Promise<ProfileSessionResponse> {
  return request(`/api/profile/${sessionId}`);
}

export function confirmProfile(
  sessionId: string,
  slug: string,
): Promise<{ slug: string; path: string; message: string }> {
  return request(`/api/profile/${sessionId}/confirm`, {
    method: "POST",
    body: JSON.stringify({ slug }),
  });
}

export async function uploadDocument(
  sessionId: string,
  file: File,
): Promise<{ filename: string; summary: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`/api/backend/api/profile/${sessionId}/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Upload failed: ${detail}`);
  }
  return res.json();
}
