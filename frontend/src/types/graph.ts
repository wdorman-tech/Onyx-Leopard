export type NodeCategory = "location" | "corporate" | "external" | "revenue" | "root";

export interface SimNode {
  id: string;
  type: string;
  label: string;
  category: NodeCategory;
  spawned_at: number;
  metrics: Record<string, number>;
  [key: string]: unknown;
}

export interface SimEdge {
  source: string;
  target: string;
  relationship: string;
}

export interface GraphData {
  nodes: SimNode[];
  edges: SimEdge[];
}

export interface TickData {
  tick: number;
  stage: number;
  status: string;
  metrics: Record<string, number>;
  events: string[];
  graph: GraphData;
}
