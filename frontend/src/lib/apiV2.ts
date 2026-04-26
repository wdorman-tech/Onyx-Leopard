// Onyx Leopard v2 API client — seed + stance + library, no industry slugs.
// Lives in parallel with `api.ts` (v1) until the legacy routes are deleted.

const API_BASE = "/api/backend";
const SSE_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`v2 API error ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

// ─── CompanySeed (matches backend pydantic model) ────────────────────────

export type EconomicsModel = "physical" | "subscription" | "service";

export interface CompanySeed {
  name: string;
  niche: string;
  archetype: string;
  industry_keywords: string[];
  location_label: string;

  economics_model: EconomicsModel;
  starting_price: number;
  base_unit_cost: number;
  daily_fixed_costs: number;
  starting_cash: number;
  starting_employees: number;
  base_capacity_per_location: number;
  margin_target: number;
  revenue_per_employee_target: number;

  tam: number;
  competitor_density: number;
  market_growth_rate: number;
  customer_unit_label: string;
  seasonality_amplitude: number;

  initial_supplier_types: string[];
  initial_revenue_streams: string[];
  initial_cost_centers: string[];
}

// ─── CeoStance ────────────────────────────────────────────────────────────

export type HiringBias = "lean" | "balanced" | "build_bench";
export type TimeHorizon = "quarterly" | "annual" | "decade";

export interface CeoStance {
  archetype: string;
  risk_tolerance: number;
  growth_obsession: number;
  quality_floor: number;
  hiring_bias: HiringBias;
  time_horizon: TimeHorizon;
  cash_comfort: number;
  signature_moves: string[];
  voice: string;
}

// ─── Library ──────────────────────────────────────────────────────────────

export interface NodeLibraryEntry {
  label: string;
  category: string;
  hire_cost: number;
  daily_fixed_costs: number;
  modifier_keys: Record<string, number>;
  prerequisites: string[];
  applicable_economics: string[];
  soft_cap: number;
  hard_cap: number;
}

export interface LibrarySummary {
  node_count: number;
  categories: string[];
  nodes: Record<string, NodeLibraryEntry>;
}

export function getV2Library(): Promise<LibrarySummary> {
  return request("/api/v2/simulate/library");
}

export interface ArchetypesPayload {
  seed_archetypes: string[];
  stance_archetypes: string[];
}

export function getV2Archetypes(): Promise<ArchetypesPayload> {
  return request("/api/v2/simulate/archetypes");
}

// ─── Sampling helpers ─────────────────────────────────────────────────────

export function sampleSeed(archetype: string, rngSeed?: number): Promise<CompanySeed> {
  const qs = rngSeed !== undefined ? `?rng_seed=${rngSeed}` : "";
  return request(`/api/v2/simulate/seed/sample/${archetype}${qs}`, {
    method: "POST",
  });
}

export function sampleStance(archetype: string, rngSeed?: number): Promise<CeoStance> {
  const qs = rngSeed !== undefined ? `?rng_seed=${rngSeed}` : "";
  return request(`/api/v2/simulate/stance/sample/${archetype}${qs}`, {
    method: "POST",
  });
}

// ─── Start / control ──────────────────────────────────────────────────────

export interface StartV2Request {
  seed: CompanySeed;
  stance: CeoStance;
  num_companies?: number;
  duration_ticks?: number;
  tam_initial?: number;
  shock_lambdas?: Record<string, number>;
  cost_ceiling_usd?: number;
  rng_seed?: number;
}

export interface StartV2Response {
  session_id: string;
  sim_id: string;
  num_companies: number;
  max_ticks: number;
}

export function startV2(req: StartV2Request): Promise<StartV2Response> {
  return request("/api/v2/simulate/start", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export type ControlAction = "play" | "pause" | "stop" | "set_speed";

export function controlV2(
  sessionId: string,
  action: ControlAction,
  speed?: number,
): Promise<{ status: string; paused: boolean; stopped: boolean; speed: number }> {
  return request(`/api/v2/simulate/control/${sessionId}`, {
    method: "POST",
    body: JSON.stringify({ action, speed }),
  });
}

// ─── SSE stream ───────────────────────────────────────────────────────────

export function createV2Stream(sessionId: string): EventSource {
  return new EventSource(`${SSE_BASE}/api/v2/simulate/stream/${sessionId}`);
}

// ─── Tick payload (what /stream emits) ───────────────────────────────────

export interface ShockPayload {
  name: string;
  severity: "minor" | "moderate" | "severe";
  duration_ticks: number;
  impact: Record<string, number>;
  description: string;
  tick_started: number | null;
}

export interface CeoDecisionPayload {
  spawn_nodes: string[];
  retire_nodes: string[];
  adjust_params: Record<string, number>;
  open_locations: number;
  reasoning: string;
  references_stance: string[];
  tier: "heuristic" | "tactical" | "strategic";
  tick: number;
}

export interface GraphSnapshotV2 {
  nodes: Array<{
    id: string;
    type: string;
    label: string;
    category: string;
    spawned_at: number;
    metrics: Record<string, number>;
  }>;
  edges: Array<{
    source: string;
    target: string;
    relationship: string;
  }>;
}

export interface CompanyTickPayload {
  company_id: string;
  tick: number;
  cash: number;
  daily_revenue: number;
  daily_costs: number;
  monthly_revenue: number;
  capacity_utilization: number;
  avg_satisfaction: number;
  employee_count: number;
  spawned_nodes: Record<string, number>;
  bankrupt: boolean;
  active_shocks: ShockPayload[];
  arriving_shocks: ShockPayload[];
  decisions: CeoDecisionPayload[];
  graph: GraphSnapshotV2;
}

export interface TickEvent {
  type: "tick";
  tick: number;
  tam: number;
  alive: number;
  shares: number[];
  companies: CompanyTickPayload[];
}

export interface CompleteEvent {
  type: "complete";
  tick: number;
}

export interface StoppedEvent {
  type: "stopped";
}

export type V2StreamEvent = TickEvent | CompleteEvent | StoppedEvent;
