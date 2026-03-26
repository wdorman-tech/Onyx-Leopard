import type { GraphData } from "./graph";

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
  graph: GraphData;
  events: string[];
}
