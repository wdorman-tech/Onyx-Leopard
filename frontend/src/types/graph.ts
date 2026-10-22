export type NodeCategory =
  | "location"
  | "corporate"
  | "external"
  | "revenue"
  | "root"
  | "supplier"
  | "ops"
  | "sales"
  | "marketing"
  | "finance"
  | "exec"
  | "cost";

export interface SimNode {
  id: string;
  type: string;
  label: string;
  category: NodeCategory;
  spawned_at: number;
  metrics: Record<string, number>;
  companyId?: string;
  companyName?: string;
  companyColor?: string;
  alive?: boolean;
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
