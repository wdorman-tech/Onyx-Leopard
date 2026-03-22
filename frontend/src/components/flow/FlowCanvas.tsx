"use client";

import { useEffect } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import type { CompanyGraph, BioNodeSummary } from "@/types/graph";
import { getLayoutedElements } from "@/lib/layout";
import { BusinessNode } from "./BusinessNode";
import { Share2 } from "@/components/ui/icons";

const nodeTypes = { business: BusinessNode };

interface FlowCanvasProps {
  graph: CompanyGraph | null;
  previousGraph?: CompanyGraph | null;
  bioSummary?: Record<string, BioNodeSummary> | null;
}

function buildFlowElements(
  graph: CompanyGraph,
  previousGraph?: CompanyGraph | null,
  bioSummary?: Record<string, BioNodeSummary> | null,
): { nodes: Node[]; edges: Edge[] } {
  const prevMetrics = new Map<string, Record<string, number>>();
  if (previousGraph) {
    for (const node of previousGraph.nodes) {
      prevMetrics.set(node.id, { ...node.metrics });
    }
  }

  const rawNodes: Node[] = graph.nodes.map((n) => {
    const prev = prevMetrics.get(n.id);
    const delta: Record<string, number> = {};
    if (prev) {
      for (const [key, val] of Object.entries(n.metrics)) {
        delta[key] = val - (prev[key] ?? 0);
      }
    }
    return {
      id: n.id,
      type: "business",
      position: { x: 0, y: 0 },
      data: { nodeData: n, delta: prev ? delta : undefined },
    };
  });

  // Compute max flux for edge scaling (from bio_summary / resource_efficiency)
  const maxRevenue = Math.max(
    1,
    ...graph.nodes
      .filter((n) => n.type === "revenue_stream")
      .map((n) => n.metrics.revenue ?? 0),
  );

  const rawEdges: Edge[] = graph.edges.map((e, i) => {
    const isFunds = e.relationship === "funds";
    const isSupplies = e.relationship === "supplies";

    // Scale edge width by source node's revenue for funds edges
    let strokeWidth = 1.5;
    if (isFunds || isSupplies) {
      const sourceNode = graph.nodes.find((n) => n.id === e.source);
      if (sourceNode) {
        const value = sourceNode.metrics.revenue ?? sourceNode.metrics.budget ?? 0;
        strokeWidth = 1.5 + (value / maxRevenue) * 3;
      }
    }

    // Highlight bottleneck edges: if source node health is low, edge is stressed
    let strokeColor = "#cbd5e1";
    if (isFunds || isSupplies) {
      const sourceState = bioSummary?.[e.source];
      if (sourceState && sourceState.health_score < 0.3) {
        strokeColor = "#ef4444"; // red for bottleneck
      } else if (sourceState && sourceState.health_score < 0.6) {
        strokeColor = "#f59e0b"; // yellow for stressed
      }
    }

    return {
      id: `edge-${i}`,
      type: "smoothstep",
      source: e.source,
      target: e.target,
      label: e.label,
      animated: isFunds || isSupplies,
      style: { stroke: strokeColor, strokeWidth },
      labelStyle: { fill: "#64748b", fontSize: 10, fontWeight: 500 },
      labelBgStyle: { fill: "#ffffff", fillOpacity: 0.9 },
      labelBgPadding: [6, 3] as [number, number],
      labelBgBorderRadius: 6,
    };
  });

  return getLayoutedElements(rawNodes, rawEdges);
}

export function FlowCanvas({ graph, previousGraph, bioSummary }: FlowCanvasProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  useEffect(() => {
    if (!graph) return;
    const { nodes: ln, edges: le } = buildFlowElements(graph, previousGraph, bioSummary);
    setNodes(ln);
    setEdges(le);
  }, [graph, previousGraph, bioSummary, setNodes, setEdges]);

  return (
    <div className="h-full w-full bg-surface-0">
      {graph ? (
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.3 }}
          proOptions={{ hideAttribution: true }}
          minZoom={0.3}
          maxZoom={2}
        >
          <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="#e2e8f0" />
          <Controls showInteractive={false} />
          <MiniMap
            nodeColor="#cbd5e1"
            maskColor="rgba(255,255,255,0.8)"
            pannable
            zoomable
          />
        </ReactFlow>
      ) : (
        <div className="flex items-center justify-center h-full">
          <div className="text-center space-y-3">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-surface-50 border border-surface-200">
              <Share2 size={28} className="text-surface-400" />
            </div>
            <p className="text-surface-500 text-sm">Your company flowchart will appear here</p>
          </div>
        </div>
      )}
    </div>
  );
}
