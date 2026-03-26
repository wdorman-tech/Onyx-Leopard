"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  forceSimulation,
  forceManyBody,
  forceLink,
  forceCenter,
  forceCollide,
  forceRadial,
  type Simulation,
  type SimulationNodeDatum,
  type SimulationLinkDatum,
} from "d3-force";
import type { GraphData, SimNode, SimEdge, NodeCategory } from "@/types/graph";
import { NodeTooltip } from "./NodeTooltip";

/* ── Visual constants ── */

const BG = "#ffffff";
const EDGE_COLOR = "rgba(148, 163, 184, 0.35)";

const CATEGORY_COLORS: Record<NodeCategory, string> = {
  root: "#ec4899",
  location: "#22c55e",
  corporate: "#3b82f6",
  external: "#f59e0b",
  revenue: "#8b5cf6",
};

const CATEGORY_RADIUS: Record<NodeCategory, number> = {
  root: 14,
  location: 8,
  corporate: 6,
  external: 5,
  revenue: 7,
};

/* ── Node type for d3-force ── */

interface ForceNode extends SimulationNodeDatum {
  id: string;
  label: string;
  type: string;
  category: NodeCategory;
  metrics: Record<string, number>;
  radius: number;
  color: string;
  depth: number;
}

interface ForceLink extends SimulationLinkDatum<ForceNode> {
  source: string | ForceNode;
  target: string | ForceNode;
}

/* ── Helpers ── */

function computeDepths(nodes: ForceNode[], edges: SimEdge[]): void {
  const adj = new Map<string, string[]>();
  for (const e of edges) {
    const s = typeof e.source === "string" ? e.source : (e.source as ForceNode).id;
    const t = typeof e.target === "string" ? e.target : (e.target as ForceNode).id;
    if (!adj.has(s)) adj.set(s, []);
    adj.get(s)!.push(t);
    if (!adj.has(t)) adj.set(t, []);
    adj.get(t)!.push(s);
  }
  const depthMap = new Map<string, number>();
  const root = nodes.find((n) => n.category === "root");
  if (!root) return;
  depthMap.set(root.id, 0);
  const queue = [root.id];
  while (queue.length > 0) {
    const cur = queue.shift()!;
    const d = depthMap.get(cur)!;
    for (const neighbor of adj.get(cur) ?? []) {
      if (!depthMap.has(neighbor)) {
        depthMap.set(neighbor, d + 1);
        queue.push(neighbor);
      }
    }
  }
  for (const n of nodes) {
    n.depth = depthMap.get(n.id) ?? 3;
  }
}

function injectRoot(graph: GraphData, companyName: string): { nodes: SimNode[]; edges: SimEdge[] } {
  const rootId = "__root__";
  const rootNode: SimNode = {
    id: rootId,
    type: "company_root",
    label: companyName,
    category: "root",
    spawned_at: 0,
    metrics: {},
  };

  const incomingSet = new Set(graph.edges.map((e) => e.target));
  const rootEdges: SimEdge[] = [];

  for (const n of graph.nodes) {
    if (n.type === "owner_operator" || !incomingSet.has(n.id)) {
      rootEdges.push({ source: rootId, target: n.id, relationship: "owns" });
    }
  }

  // If nothing connected (edge case), connect to first node
  if (rootEdges.length === 0 && graph.nodes.length > 0) {
    rootEdges.push({ source: rootId, target: graph.nodes[0].id, relationship: "owns" });
  }

  return {
    nodes: [rootNode, ...graph.nodes],
    edges: [...rootEdges, ...graph.edges],
  };
}

function toForceNodes(simNodes: SimNode[]): ForceNode[] {
  return simNodes.map((n) => ({
    id: n.id,
    label: n.label,
    type: n.type,
    category: n.category,
    metrics: n.metrics,
    radius: CATEGORY_RADIUS[n.category] ?? 5,
    color: CATEGORY_COLORS[n.category] ?? "#94a3b8",
    depth: 0,
  }));
}

/* ── Component ── */

interface ForceGraphProps {
  graph: GraphData | null;
  companyName?: string;
}

export function ForceGraph({ graph, companyName = "Company" }: ForceGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<Simulation<ForceNode, ForceLink> | null>(null);
  const nodesRef = useRef<ForceNode[]>([]);
  const linksRef = useRef<ForceLink[]>([]);
  const edgesRef = useRef<SimEdge[]>([]);
  const transformRef = useRef({ x: 0, y: 0, k: 1 });
  const dragRef = useRef<{ active: boolean; startX: number; startY: number; originX: number; originY: number }>({
    active: false, startX: 0, startY: 0, originX: 0, originY: 0,
  });
  const hoveredRef = useRef<ForceNode | null>(null);
  const rafRef = useRef<number>(0);
  const prevNodeCountRef = useRef(0);

  const [selected, setSelected] = useState<{ node: ForceNode; sx: number; sy: number } | null>(null);

  /* ── Screen ↔ Sim coordinate conversion ── */
  const screenToSim = useCallback((sx: number, sy: number) => {
    const t = transformRef.current;
    return { x: (sx - t.x) / t.k, y: (sy - t.y) / t.k };
  }, []);

  const simToScreen = useCallback((x: number, y: number) => {
    const t = transformRef.current;
    return { sx: x * t.k + t.x, sy: y * t.k + t.y };
  }, []);

  /* ── Render loop ── */
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);

    const t = transformRef.current;
    ctx.save();
    ctx.translate(t.x, t.y);
    ctx.scale(t.k, t.k);

    // Draw edges
    ctx.strokeStyle = EDGE_COLOR;
    ctx.lineWidth = 1 / t.k;
    for (const link of linksRef.current) {
      const s = link.source as ForceNode;
      const tgt = link.target as ForceNode;
      if (s.x == null || tgt.x == null) continue;
      ctx.beginPath();
      ctx.moveTo(s.x, s.y!);
      ctx.lineTo(tgt.x, tgt.y!);
      ctx.stroke();
    }

    // Draw nodes
    const hovered = hoveredRef.current;
    for (const node of nodesRef.current) {
      if (node.x == null) continue;
      const r = node.radius;
      const isHovered = hovered?.id === node.id;
      const isSelected = selected?.node.id === node.id;

      // Glow halo
      ctx.beginPath();
      ctx.arc(node.x, node.y!, r * (isHovered || isSelected ? 3 : 2.2), 0, Math.PI * 2);
      ctx.fillStyle = node.color + (isHovered || isSelected ? "30" : "18");
      ctx.fill();

      // Core dot
      ctx.beginPath();
      ctx.arc(node.x, node.y!, r, 0, Math.PI * 2);
      ctx.fillStyle = isHovered || isSelected ? "#0f172a" : node.color;
      ctx.fill();

      // Brighter ring on hover/select
      if (isHovered || isSelected) {
        ctx.beginPath();
        ctx.arc(node.x, node.y!, r + 2, 0, Math.PI * 2);
        ctx.strokeStyle = node.color;
        ctx.lineWidth = 1.5 / t.k;
        ctx.stroke();
      }
    }

    // Hover label
    if (hovered && hovered.x != null) {
      const fontSize = Math.max(10, 12 / t.k);
      ctx.font = `500 ${fontSize}px Inter, system-ui, sans-serif`;
      ctx.fillStyle = "#334155";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(hovered.label, hovered.x + hovered.radius + 6, hovered.y!);
    }

    ctx.restore();

    rafRef.current = requestAnimationFrame(draw);
  }, [selected]);

  /* ── Initialize and manage simulation ── */
  useEffect(() => {
    if (!graph || graph.nodes.length === 0) {
      // Clear everything
      nodesRef.current = [];
      linksRef.current = [];
      edgesRef.current = [];
      if (simRef.current) simRef.current.stop();
      return;
    }

    const { nodes: injectedNodes, edges: injectedEdges } = injectRoot(graph, companyName);
    const newNodeCount = injectedNodes.length;
    const nodeCountChanged = newNodeCount !== prevNodeCountRef.current;
    prevNodeCountRef.current = newNodeCount;

    const existingMap = new Map(nodesRef.current.map((n) => [n.id, n]));
    const forceNodes = toForceNodes(injectedNodes);

    // Preserve positions for existing nodes, initialize new ones near parent
    const container = containerRef.current;
    const cx = (container?.clientWidth ?? 800) / 2;
    const cy = (container?.clientHeight ?? 600) / 2;

    for (const fn of forceNodes) {
      const existing = existingMap.get(fn.id);
      if (existing) {
        fn.x = existing.x;
        fn.y = existing.y;
        fn.vx = existing.vx;
        fn.vy = existing.vy;
      } else if (fn.category === "root") {
        fn.x = cx;
        fn.y = cy;
        fn.fx = cx;
        fn.fy = cy;
      } else {
        // Find a connected existing node to spawn near
        const parentEdge = injectedEdges.find(
          (e) => e.target === fn.id && existingMap.has(e.source),
        ) ?? injectedEdges.find(
          (e) => e.source === fn.id && existingMap.has(e.target),
        );
        const parentId = parentEdge
          ? (parentEdge.source === fn.id ? parentEdge.target : parentEdge.source)
          : null;
        const parent = parentId ? existingMap.get(parentId) : null;
        if (parent && parent.x != null) {
          fn.x = parent.x + (Math.random() - 0.5) * 30;
          fn.y = (parent.y ?? cy) + (Math.random() - 0.5) * 30;
        } else {
          fn.x = cx + (Math.random() - 0.5) * 60;
          fn.y = cy + (Math.random() - 0.5) * 60;
        }
      }

      // Update metrics from latest tick
      const source = injectedNodes.find((n) => n.id === fn.id);
      if (source) fn.metrics = source.metrics;
    }

    computeDepths(forceNodes, injectedEdges);

    const forceLinks: ForceLink[] = injectedEdges
      .filter((e) => forceNodes.some((n) => n.id === e.source) && forceNodes.some((n) => n.id === e.target))
      .map((e) => ({ source: e.source, target: e.target }));

    nodesRef.current = forceNodes;
    linksRef.current = forceLinks;
    edgesRef.current = injectedEdges;

    // Update or create simulation
    if (simRef.current) {
      simRef.current.nodes(forceNodes);
      simRef.current
        .force("link", forceLink<ForceNode, ForceLink>(forceLinks).id((d) => d.id).distance(60).strength(0.4))
        .force("radial", forceRadial<ForceNode>((d) => (d.category === "root" ? 0 : 80 + d.depth * 50), cx, cy).strength(0.25));
      if (nodeCountChanged) {
        simRef.current.alpha(0.3).restart();
      }
    } else {
      const sim = forceSimulation(forceNodes)
        .force("charge", forceManyBody<ForceNode>().strength(-120))
        .force("link", forceLink<ForceNode, ForceLink>(forceLinks).id((d) => d.id).distance(60).strength(0.4))
        .force("center", forceCenter(cx, cy).strength(0.05))
        .force("collide", forceCollide<ForceNode>().radius((d) => d.radius + 4))
        .force("radial", forceRadial<ForceNode>((d) => (d.category === "root" ? 0 : 80 + d.depth * 50), cx, cy).strength(0.25))
        .alphaDecay(0.02)
        .velocityDecay(0.3);
      simRef.current = sim;
    }

    // Clear selected if its node no longer exists
    setSelected((prev) => {
      if (prev && !forceNodes.some((n) => n.id === prev.node.id)) return null;
      // Update metrics on selected node
      if (prev) {
        const updated = forceNodes.find((n) => n.id === prev.node.id);
        if (updated) return { ...prev, node: updated };
      }
      return prev;
    });
  }, [graph, companyName]);

  /* ── Canvas sizing ── */
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = container.clientWidth;
      const h = container.clientHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;

      // Re-center the transform if first render
      if (transformRef.current.x === 0 && transformRef.current.y === 0) {
        transformRef.current = { x: 0, y: 0, k: 1 };
      }

      // Update center force and root fix
      const cx = w / 2;
      const cy = h / 2;
      if (simRef.current) {
        simRef.current.force("center", forceCenter(cx, cy).strength(0.05));
        simRef.current.force("radial", forceRadial<ForceNode>(
          (d) => (d.category === "root" ? 0 : 80 + d.depth * 50), cx, cy,
        ).strength(0.25));
      }
      const root = nodesRef.current.find((n) => n.category === "root");
      if (root) {
        root.fx = cx;
        root.fy = cy;
      }
    };

    const ro = new ResizeObserver(resize);
    ro.observe(container);
    resize();
    return () => ro.disconnect();
  }, []);

  /* ── Animation frame loop ── */
  useEffect(() => {
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  /* ── Mouse interaction ── */
  const hitTest = useCallback((sx: number, sy: number): ForceNode | null => {
    const { x, y } = screenToSim(sx, sy);
    const t = transformRef.current;
    for (let i = nodesRef.current.length - 1; i >= 0; i--) {
      const n = nodesRef.current[i];
      if (n.x == null) continue;
      const dx = n.x - x;
      const dy = (n.y ?? 0) - y;
      const hitR = Math.max(n.radius + 4, 10 / t.k);
      if (dx * dx + dy * dy < hitR * hitR) return n;
    }
    return null;
  }, [screenToSim]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;

    if (dragRef.current.active) {
      const dx = e.clientX - dragRef.current.startX;
      const dy = e.clientY - dragRef.current.startY;
      transformRef.current.x = dragRef.current.originX + dx;
      transformRef.current.y = dragRef.current.originY + dy;
      return;
    }

    const hit = hitTest(sx, sy);
    hoveredRef.current = hit;
    if (canvasRef.current) {
      canvasRef.current.style.cursor = hit ? "pointer" : "grab";
    }
  }, [hitTest]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const hit = hitTest(sx, sy);

    if (hit) {
      const { sx: screenX, sy: screenY } = simToScreen(hit.x!, hit.y!);
      setSelected({ node: hit, sx: screenX, sy: screenY });
      return;
    }

    // Start pan
    dragRef.current = {
      active: true,
      startX: e.clientX,
      startY: e.clientY,
      originX: transformRef.current.x,
      originY: transformRef.current.y,
    };
    if (canvasRef.current) canvasRef.current.style.cursor = "grabbing";
  }, [hitTest, simToScreen]);

  const handleMouseUp = useCallback(() => {
    dragRef.current.active = false;
    if (canvasRef.current) canvasRef.current.style.cursor = hoveredRef.current ? "pointer" : "grab";
  }, []);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const t = transformRef.current;
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    const newK = Math.max(0.3, Math.min(3, t.k * factor));

    // Zoom toward cursor
    t.x = mx - (mx - t.x) * (newK / t.k);
    t.y = my - (my - t.y) * (newK / t.k);
    t.k = newK;
  }, []);

  const handleTooltipClose = useCallback(() => setSelected(null), []);

  return (
    <div ref={containerRef} className="relative w-full h-full overflow-hidden" style={{ background: BG }}>
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        className="block w-full h-full"
        style={{ cursor: "grab" }}
      />
      {selected && (
        <NodeTooltip
          label={selected.node.label}
          type={selected.node.type}
          category={selected.node.category}
          metrics={selected.node.metrics}
          x={selected.sx}
          y={selected.sy}
          onClose={handleTooltipClose}
        />
      )}
    </div>
  );
}
