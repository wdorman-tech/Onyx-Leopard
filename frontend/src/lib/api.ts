import type { GraphData } from "@/types/graph";

const API_BASE = "/api/backend";
const SSE_BASE = "http://localhost:8000";

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

export function controlSimulation(
  sessionId: string,
  action: string,
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

export function startUnifiedSimulation(
  startMode = "identical",
  numCompanies = 4,
  maxTicks = 0,
): Promise<{ session_id: string }> {
  return request("/api/simulate/start", {
    method: "POST",
    body: JSON.stringify({
      mode: "unified",
      start_mode: startMode,
      num_companies: numCompanies,
      max_ticks: maxTicks,
    }),
  });
}

export function focusCompany(
  sessionId: string,
  companyId: string,
): Promise<{ status: string; graph?: GraphData }> {
  return request(`/api/simulate/control/${sessionId}`, {
    method: "POST",
    body: JSON.stringify({ action: "focus_company", company_id: companyId }),
  });
}
