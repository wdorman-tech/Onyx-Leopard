// Onyx Leopard API client. REST goes through the Next rewrite (/api/backend →
// :8000); the SSE stream connects to the backend directly so it is not buffered.

const API_BASE = "/api/backend";
const SSE_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = await res.text();
    try {
      const parsed = JSON.parse(detail);
      detail = typeof parsed.detail === "string" ? parsed.detail : JSON.stringify(parsed.detail);
    } catch {
      // not JSON; keep raw text
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ─── Models ───────────────────────────────────────────────────────────────

export interface ModelOption {
  id: string;
  name: string;
  prompt_per_million: number;
  completion_per_million: number;
  context_length: number;
  intelligence_index: number | null;
}

export type Tier = "bad" | "mediocre" | "good" | "super";

export interface ModelTier extends ModelOption {
  tier: Tier;
  label: string;
  blurb: string;
}

export interface ModelCatalog {
  default: string;
  live: boolean;
  tiers: ModelTier[];
  anthropic: ModelOption[];
}

export interface AppConfig {
  ai_available: boolean;
  key_env: string;
  jev_available: boolean;
}

export const getConfig = () => request<AppConfig>("/api/config");
export const getModels = () => request<ModelCatalog>("/api/models");

// Measured averages per AI CEO call (prompt ≈ 1.2–1.7k tokens; output includes
// low-effort reasoning). Used only for the cost estimate on the setup screens.
const TOKENS_IN_PER_CALL = 1600;
const TOKENS_OUT_PER_CALL = 900;
const AI_CADENCE_DAYS = 30;

export function estimateRunCost(model: ModelOption, companies: number, days: number): number {
  const calls = companies * (Math.floor(days / AI_CADENCE_DAYS) + 1); // +1: end-of-run report
  const perCall =
    (TOKENS_IN_PER_CALL * model.prompt_per_million +
      TOKENS_OUT_PER_CALL * model.completion_per_million) /
    1e6;
  return calls * perCall;
}

export function formatUsd(n: number): string {
  if (n === 0) return "$0";
  if (n < 0.01) return `$${n.toFixed(4)}`;
  if (n < 1) return `$${n.toFixed(3)}`;
  return `$${n.toFixed(2)}`;
}

// ─── Setup data ───────────────────────────────────────────────────────────

export interface CeoArchetype {
  id: string;
  label: string;
  summary: string;
}

export interface Archetypes {
  ceo: CeoArchetype[];
  company: string[];
}

/** Opaque to the UI except for the fields it displays; sent back verbatim. */
export interface CompanySeed {
  name: string;
  niche: string;
  economics_model: "physical" | "subscription" | "service";
  tam: number;
  [field: string]: unknown;
}

export interface CompanyProfile {
  filename: string;
  name: string;
  industry: string;
  seed: CompanySeed;
}

export const getArchetypes = () => request<Archetypes>("/api/archetypes");
export const getCompanies = () => request<CompanyProfile[]>("/api/companies");

// ─── Simulation ───────────────────────────────────────────────────────────

export interface StartRequest {
  seed?: CompanySeed;
  company_archetype: string;
  ceo_archetypes: string[];
  duration_days: number;
  ai_enabled: boolean;
  model?: string;
  budget_usd: number;
  shocks_enabled: boolean;
  jev_enabled: boolean;
  rng_seed?: number;
}

export interface StartResponse {
  session_id: string;
  max_ticks: number;
  model: string | null;
  companies: { company_id: string; name: string; archetype: string }[];
}

export const startSimulation = (req: StartRequest) =>
  request<StartResponse>("/api/simulate/start", { method: "POST", body: JSON.stringify(req) });

export type ControlAction = "play" | "pause" | "stop" | "set_speed";

export const controlSimulation = (sessionId: string, action: ControlAction, speed?: number) =>
  request<{ paused: boolean; speed: number }>(`/api/simulate/control/${sessionId}`, {
    method: "POST",
    body: JSON.stringify({ action, speed }),
  });

export const openStream = (sessionId: string) =>
  new EventSource(`${SSE_BASE}/api/simulate/stream/${sessionId}`);

// ─── Stream payloads ──────────────────────────────────────────────────────

export interface CeoDecision {
  day: number;
  /** "jev" = a weekly rule proposal Jev approved, or vetoed into a hold. */
  source: "heuristic" | "ai" | "jev";
  spawn: string[];
  retire: string[];
  open_locations: number;
  price: number | null;
  marketing_intensity: number | null;
  raise_amount: number | null;
  reasoning: string;
  references_stance: string[];
  jev_p: number | null;
  jev_approved: boolean | null;
}

export interface GraphPayload {
  nodes: {
    id: string;
    type: string;
    label: string;
    category: string;
    spawned_at: number;
    metrics: Record<string, number>;
  }[];
  edges: { source: string; target: string; relationship: string }[];
}

export interface CompanyState {
  company_id: string;
  name: string;
  archetype: string;
  alive: boolean;
  cash: number;
  daily_revenue: number;
  /** Sales to the core market (served × price), before revenue-stream uplift. */
  core_revenue: number;
  daily_costs: number;
  market_share: number;
  price: number;
  marketing_intensity: number;
  satisfaction: number;
  utilization: number;
  binding: "demand" | "capacity" | "production";
  employees: number;
  locations: number;
  runway_months: number | null;
  shocks: string[];
  decisions: CeoDecision[];
  graph: GraphPayload;
}

export interface AiStatus {
  model: string;
  spent_usd: number;
  budget_usd: number;
  calls: number;
  error: string | null;
}

export interface JevStatus {
  model: string;
  calls: number;
  approved: number;
  vetoed: number;
  errors: number;
  cost_usd: number;
  error: string | null;
}

export interface SavedRun {
  run_id: string;
  /** File name → API path, e.g. "report.pdf" → "/api/runs/<id>/report.pdf". */
  files?: Record<string, string>;
  error?: string;
}

/** Saved-run files are served through the same-origin rewrite. */
export const runFileUrl = (path: string) => `${API_BASE}${path}`;

export interface TickEvent {
  type: "tick";
  tick: number;
  max_ticks: number;
  tam: number;
  alive: number;
  market_shocks: string[];
  events: string[];
  ai: AiStatus | null;
  jev: JevStatus | null;
  companies: CompanyState[];
}

export interface CeoReport {
  company_id: string;
  company_name: string;
  strategy: string;
  source: "ai" | "heuristic";
  performance_summary: string;
  what_went_well: string;
  what_went_wrong: string;
  key_decisions: string[];
  final_assessment: string;
}

export type StreamEvent =
  | TickEvent
  | { type: "thinking"; tick: number }
  | { type: "reports"; reports: CeoReport[] }
  | ({ type: "saved" } & SavedRun)
  | { type: "complete"; tick: number; ai: AiStatus | null }
  | { type: "stopped" };
