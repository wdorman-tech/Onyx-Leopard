import type { GraphData } from "./graph";

export type CEOStrategy =
  | "aggressive_growth"
  | "quality_focus"
  | "cost_leader"
  | "balanced"
  | "market_dominator"
  | "survivor";

export interface CEODecision {
  reasoning: string;
  price_adjustment: number;
  expansion_pace: "aggressive" | "normal" | "conservative";
  marketing_intensity: number;
  quality_investment: number;
  cost_target: number;
  max_locations_per_year: number;
}

export interface CEOReport {
  company_name: string;
  strategy: string;
  performance_summary: string;
  what_went_well: string;
  what_went_wrong: string;
  key_decisions: string[];
  final_assessment: string;
}

export interface CEODecisionEvent {
  company_name: string;
  tick: number;
  sim_year: number;
  strategy: string;
  decision: CEODecision;
}

export interface UnifiedAgentSnapshot {
  id: string;
  name: string;
  alive: boolean;
  color: string;
  quality: number;
  marketing: number;
  capacity: number;
  share: number;
  utilization: number;
  binding_constraint: "demand" | "capacity";
  cash: number;
  daily_revenue: number;
  daily_costs: number;
  stage: number;
  location_count: number;
  node_count: number;
  avg_satisfaction: number;
  total_employees: number;
  strategy?: string | null;
}

export interface UnifiedTickData {
  tick: number;
  status: string;
  mode: "unified";
  tam: number;
  captured: number;
  hhi: number;
  agent_count: number;
  agents: UnifiedAgentSnapshot[];
  focused_company_id: string;
  graphs: Record<string, GraphData>;
  events: string[];
}
