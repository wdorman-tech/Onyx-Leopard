export type NodeType =
  | "department"
  | "team"
  | "role"
  | "revenue_stream"
  | "cost_center"
  | "external";

export type RelationshipType =
  | "reports_to"
  | "funds"
  | "supplies"
  | "collaborates"
  | "serves";

export interface NodeData {
  id: string;
  label: string;
  type: NodeType;
  metrics: Record<string, number>;
  agent_prompt: string;
}

export interface EdgeData {
  source: string;
  target: string;
  relationship: RelationshipType;
  label: string;
}

export interface CompanyGraph {
  name: string;
  description: string;
  nodes: NodeData[];
  edges: EdgeData[];
  global_metrics: Record<string, number>;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface AgentAction {
  agent_id: string;
  action_type: string;
  params: Record<string, unknown>;
  reasoning: string;
}

export interface BioNodeSummary {
  health_score: number;
  capacity_utilization: number;
  signal_activation: number;
  apoptosis_triggered: boolean;
  cell_cycle_phase?: string;
  adoption_r0?: number;
}

export interface TickEvent {
  type: "tick";
  tick: number;
  graph: CompanyGraph;
  actions: AgentAction[];
  global_metrics: Record<string, number>;
  bio_summary?: Record<string, BioNodeSummary>;
}

export interface SimulationEvent {
  type: "tick" | "complete" | "stopped";
  tick?: number;
  graph?: CompanyGraph;
  actions?: AgentAction[];
  global_metrics?: Record<string, number>;
  bio_summary?: Record<string, BioNodeSummary>;
}
