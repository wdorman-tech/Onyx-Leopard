import dagre from "@dagrejs/dagre";
import type { Edge, Node } from "@xyflow/react";

const NODE_WIDTH = 200;
const NODE_HEIGHT = 80;

export function getLayoutedElements(
  nodes: Node[],
  edges: Edge[],
  direction: "LR" | "TB" = "LR"
): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: direction, nodesep: 80, ranksep: 200, acyclicer: "greedy" });

  nodes.forEach((node) => {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  });

  edges.forEach((edge) => {
    g.setEdge(edge.source, edge.target);
  });

  dagre.layout(g);

  const layoutedNodes = nodes.map((node) => {
    const pos = g.node(node.id);
    return {
      ...node,
      position: {
        x: pos.x - NODE_WIDTH / 2,
        y: pos.y - NODE_HEIGHT / 2,
      },
    };
  });

  // For edges going "backward" (source right of target in LR, or below in TB),
  // route through bottom→top handles to avoid long smoothstep detours.
  const posMap = new Map<string, { x: number; y: number }>();
  layoutedNodes.forEach((n) => posMap.set(n.id, n.position));

  const annotatedEdges = edges.map((edge) => {
    const src = posMap.get(edge.source);
    const tgt = posMap.get(edge.target);
    if (!src || !tgt) return edge;

    const isBackward =
      direction === "LR" ? src.x >= tgt.x : src.y >= tgt.y;

    if (isBackward) {
      return { ...edge, sourceHandle: "bottom", targetHandle: "top" };
    }
    return { ...edge, sourceHandle: "right", targetHandle: "left" };
  });

  return { nodes: layoutedNodes, edges: annotatedEdges };
}
