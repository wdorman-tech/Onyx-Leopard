export interface MarketPreset {
  slug: string;
  name: string;
  description: string;
  alpha: number;
  beta: number;
  delta: number;
  n_0: number;
}

export interface AgentSnapshot {
  id: string;
  name: string;
  alive: boolean;
  revenue: number;
  cash: number;
  capacity: number;
  quality: number;
  marketing: number;
  share: number;
  utilization: number;
  binding_constraint: "demand" | "capacity";
  color: string;
}

export interface MarketTickData {
  tick: number;
  tam: number;
  captured: number;
  hhi: number;
  agent_count: number;
  agents: AgentSnapshot[];
  events: string[];
  status: string;
}
