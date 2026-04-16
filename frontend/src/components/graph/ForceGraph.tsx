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
const DEAD_COLOR = "#94a3b8";
const DEAD_EDGE_COLOR = "rgba(148, 163, 184, 0.15)";

/* ── Minimap constants ── */
const MINIMAP_W = 160;
const MINIMAP_H = 120;
const MINIMAP_PAD = 12;
const MINIMAP_BG = "rgba(248, 250, 252, 0.92)";
const MINIMAP_BORDER = "rgba(148, 163, 184, 0.4)";
const MINIMAP_VIEWPORT = "rgba(59, 130, 246, 0.25)";
const MINIMAP_VIEWPORT_STROKE = "rgba(59, 130, 246, 0.6)";

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
  companyId?: string;
  companyColor?: string;
  alive?: boolean;
  rootX?: number;
  rootY?: number;
}

interface ForceLinkDatum extends SimulationLinkDatum<ForceNode> {
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
  const roots = nodes.filter((n) => n.category === "root");
  if (roots.length === 0) return;

  for (const root of roots) {
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
  }
  for (const n of nodes) {
    n.depth = depthMap.get(n.id) ?? 3;
  }
}

function injectRoot(graph: GraphData, companyName: string, founderType = "owner_operator"): { nodes: SimNode[]; edges: SimEdge[] } {
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
    if (n.type === founderType || !incomingSet.has(n.id)) {
      rootEdges.push({ source: rootId, target: n.id, relationship: "owns" });
    }
  }

  if (rootEdges.length === 0 && graph.nodes.length > 0) {
    rootEdges.push({ source: rootId, target: graph.nodes[0].id, relationship: "owns" });
  }

  return {
    nodes: [rootNode, ...graph.nodes],
    edges: [...rootEdges, ...graph.edges],
  };
}

/** Multi-company: inject a root node per company and connect orphans to it. */
function injectMultiRoots(graph: GraphData, founderType = "owner_operator"): { nodes: SimNode[]; edges: SimEdge[] } {
  const companiesInGraph = new Set<string>();
  for (const n of graph.nodes) {
    if (n.companyId) companiesInGraph.add(n.companyId);
  }

  const allNodes: SimNode[] = [...graph.nodes];
  const allEdges: SimEdge[] = [...graph.edges];
  const incomingSet = new Set(graph.edges.map((e) => e.target));

  for (const companyId of companiesInGraph) {
    const rootId = `${companyId}::__root__`;
    const companyNodes = graph.nodes.filter((n) => n.companyId === companyId);
    if (companyNodes.length === 0) continue;

    const sample = companyNodes[0];
    const rootNode: SimNode = {
      id: rootId,
      type: "company_root",
      label: companyId,
      category: "root",
      spawned_at: 0,
      metrics: {},
      companyId,
      companyColor: sample.companyColor,
      alive: sample.alive,
    };
    allNodes.push(rootNode);

    for (const n of companyNodes) {
      if (n.type === founderType || !incomingSet.has(n.id)) {
        allEdges.push({ source: rootId, target: n.id, relationship: "owns" });
      }
    }
  }

  return { nodes: allNodes, edges: allEdges };
}

function toForceNodes(simNodes: SimNode[], multiCompany: boolean): ForceNode[] {
  return simNodes.map((n) => {
    const isDead = multiCompany && n.alive === false;
    let color: string;
    if (isDead) {
      color = DEAD_COLOR;
    } else if (multiCompany && n.category === "root" && n.companyColor) {
      color = n.companyColor;
    } else {
      color = CATEGORY_COLORS[n.category] ?? "#94a3b8";
    }

    return {
      id: n.id,
      label: n.label,
      type: n.type,
      category: n.category,
      metrics: n.metrics,
      radius: CATEGORY_RADIUS[n.category] ?? 5,
      color,
      depth: 0,
      companyId: n.companyId,
      companyColor: n.companyColor,
      alive: n.alive,
    };
  });
}

/** Custom force that pulls each node toward its own company root. */
function forceCluster(strength = 0.12) {
  let nodes: ForceNode[];

  function force(alpha: number) {
    for (const node of nodes) {
      if (node.category === "root" || node.rootX == null || node.rootY == null) continue;
      const targetR = 50 + (node.depth ?? 1) * 40;
      const dx = (node.x ?? 0) - node.rootX;
      const dy = (node.y ?? 0) - node.rootY;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const targetX = node.rootX + (dx / dist) * targetR;
      const targetY = node.rootY + (dy / dist) * targetR;
      node.vx! += (targetX - node.x!) * strength * alpha;
      node.vy! += (targetY - node.y!) * strength * alpha;
    }
  }

  force.initialize = (n: ForceNode[]) => {
    nodes = n;
  };
  return force;
}

/** Update rootX/rootY on non-root nodes to track their company root's current position. */
function syncRootPositions(nodes: ForceNode[]) {
  const rootPositions = new Map<string, { x: number; y: number }>();
  for (const n of nodes) {
    if (n.category === "root" && n.companyId && n.x != null) {
      rootPositions.set(n.companyId, { x: n.x, y: n.y ?? 0 });
    }
  }
  for (const n of nodes) {
    if (n.category !== "root" && n.companyId) {
      const pos = rootPositions.get(n.companyId);
      if (pos) {
        n.rootX = pos.x;
        n.rootY = pos.y;
      }
    }
  }
}

/* ── Component ── */

interface ForceGraphProps {
  graph: GraphData | null;
  companyName?: string;
  multiCompany?: boolean;
  founderType?: string;
  onFocusCompany?: (companyId: string) => void;
}

export function ForceGraph({
  graph,
  companyName = "Company",
  multiCompany = false,
  founderType = "owner_operator",
  onFocusCompany,
}: ForceGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<Simulation<ForceNode, ForceLinkDatum> | null>(null);
  const nodesRef = useRef<ForceNode[]>([]);
  const linksRef = useRef<ForceLinkDatum[]>([]);
  const edgesRef = useRef<SimEdge[]>([]);
  const transformRef = useRef({ x: 0, y: 0, k: 1 });
  const dragRef = useRef<{ active: boolean; startX: number; startY: number; originX: number; originY: number }>({
    active: false, startX: 0, startY: 0, originX: 0, originY: 0,
  });
  const hoveredRef = useRef<ForceNode | null>(null);
  const rafRef = useRef<number>(0);
  const prevNodeCountRef = useRef(0);
  const minimapRef = useRef<{
    x: number; y: number; w: number; h: number;
    simCX: number; simCY: number; scale: number;
    canvasW: number; canvasH: number;
  } | null>(null);
  const multiCompanyRef = useRef(multiCompany);
  multiCompanyRef.current = multiCompany;
  const onFocusCompanyRef = useRef(onFocusCompany);
  onFocusCompanyRef.current = onFocusCompany;

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

    // Sync root positions each frame so clustering force has up-to-date targets
    if (multiCompanyRef.current) {
      syncRootPositions(nodesRef.current);
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);

    const t = transformRef.current;
    ctx.save();
    ctx.translate(t.x, t.y);
    ctx.scale(t.k, t.k);

    // Draw edges
    ctx.lineWidth = 1 / t.k;
    for (const link of linksRef.current) {
      const s = link.source as ForceNode;
      const tgt = link.target as ForceNode;
      if (s.x == null || tgt.x == null) continue;
      const isDead = multiCompanyRef.current && (s.alive === false || tgt.alive === false);
      ctx.strokeStyle = isDead ? DEAD_EDGE_COLOR : EDGE_COLOR;
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
      const isDead = multiCompanyRef.current && node.alive === false;

      // Glow halo
      ctx.beginPath();
      ctx.arc(node.x, node.y!, r * (isHovered || isSelected ? 3 : 2.2), 0, Math.PI * 2);
      const haloColor = isDead ? DEAD_COLOR : node.color;
      ctx.fillStyle = haloColor + (isHovered || isSelected ? "30" : isDead ? "10" : "18");
      ctx.fill();

      // Core dot
      ctx.beginPath();
      ctx.arc(node.x, node.y!, r, 0, Math.PI * 2);
      ctx.fillStyle = isHovered || isSelected ? "#0f172a" : isDead ? DEAD_COLOR : node.color;
      ctx.globalAlpha = isDead ? 0.5 : 1;
      ctx.fill();
      ctx.globalAlpha = 1;

      // Brighter ring on hover/select
      if (isHovered || isSelected) {
        ctx.beginPath();
        ctx.arc(node.x, node.y!, r + 2, 0, Math.PI * 2);
        ctx.strokeStyle = node.color;
        ctx.lineWidth = 1.5 / t.k;
        ctx.stroke();
        ctx.lineWidth = 1 / t.k;
      }
    }

    // Company name labels under root nodes (multi-company mode)
    if (multiCompanyRef.current) {
      for (const node of nodesRef.current) {
        if (node.category !== "root" || node.x == null) continue;
        const isDead = node.alive === false;
        const fontSize = Math.max(10, 13 / t.k);
        ctx.font = `600 ${fontSize}px Inter, system-ui, sans-serif`;
        ctx.fillStyle = isDead ? DEAD_COLOR : (node.companyColor ?? "#334155");
        ctx.globalAlpha = isDead ? 0.5 : 0.9;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(node.label, node.x, node.y! + node.radius + 6);
        ctx.globalAlpha = 1;
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

    // ── Minimap (drawn in screen space) ──
    const nodes = nodesRef.current;
    if (nodes.length > 0) {
      const mmX = w - MINIMAP_W - MINIMAP_PAD;
      const mmY = h - MINIMAP_H - MINIMAP_PAD;

      // Compute bounding box of all nodes in sim space
      let minSX = Infinity, maxSX = -Infinity, minSY = Infinity, maxSY = -Infinity;
      for (const n of nodes) {
        if (n.x == null) continue;
        if (n.x < minSX) minSX = n.x;
        if (n.x > maxSX) maxSX = n.x;
        if (n.y! < minSY) minSY = n.y!;
        if (n.y! > maxSY) maxSY = n.y!;
      }
      const pad = 40;
      minSX -= pad; maxSX += pad; minSY -= pad; maxSY += pad;
      const rangeX = maxSX - minSX || 1;
      const rangeY = maxSY - minSY || 1;
      const mmScale = Math.min((MINIMAP_W - 8) / rangeX, (MINIMAP_H - 8) / rangeY);

      // Store geometry for click handling
      const simCX = (minSX + maxSX) / 2;
      const simCY = (minSY + maxSY) / 2;
      minimapRef.current = { x: mmX, y: mmY, w: MINIMAP_W, h: MINIMAP_H, simCX, simCY, scale: mmScale, canvasW: w, canvasH: h };

      // Background
      ctx.fillStyle = MINIMAP_BG;
      ctx.strokeStyle = MINIMAP_BORDER;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(mmX, mmY, MINIMAP_W, MINIMAP_H, 6);
      ctx.fill();
      ctx.stroke();

      ctx.save();
      ctx.beginPath();
      ctx.roundRect(mmX, mmY, MINIMAP_W, MINIMAP_H, 6);
      ctx.clip();

      const mmCX = mmX + MINIMAP_W / 2;
      const mmCY = mmY + MINIMAP_H / 2;

      // Draw nodes as dots
      for (const n of nodes) {
        if (n.x == null) continue;
        const dx = (n.x - simCX) * mmScale + mmCX;
        const dy = (n.y! - simCY) * mmScale + mmCY;
        const r = Math.max(1.5, n.radius * mmScale * 0.5);
        ctx.beginPath();
        ctx.arc(dx, dy, r, 0, Math.PI * 2);
        ctx.fillStyle = n.alive === false ? DEAD_COLOR : n.color;
        ctx.globalAlpha = n.alive === false ? 0.3 : 0.7;
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // Draw viewport rectangle
      const vl = (0 - t.x) / t.k;  // left edge in sim coords
      const vt = (0 - t.y) / t.k;  // top edge
      const vr = (w - t.x) / t.k;
      const vb = (h - t.y) / t.k;
      const rvl = (vl - simCX) * mmScale + mmCX;
      const rvt = (vt - simCY) * mmScale + mmCY;
      const rvr = (vr - simCX) * mmScale + mmCX;
      const rvb = (vb - simCY) * mmScale + mmCY;
      ctx.fillStyle = MINIMAP_VIEWPORT;
      ctx.strokeStyle = MINIMAP_VIEWPORT_STROKE;
      ctx.lineWidth = 1;
      ctx.fillRect(rvl, rvt, rvr - rvl, rvb - rvt);
      ctx.strokeRect(rvl, rvt, rvr - rvl, rvb - rvt);

      ctx.restore();
    }

    rafRef.current = requestAnimationFrame(draw);
  }, [selected]);

  /* ── Initialize and manage simulation ── */
  useEffect(() => {
    if (!graph || graph.nodes.length === 0) {
      nodesRef.current = [];
      linksRef.current = [];
      edgesRef.current = [];
      if (simRef.current) simRef.current.stop();
      return;
    }

    const { nodes: injectedNodes, edges: injectedEdges } = multiCompany
      ? injectMultiRoots(graph, founderType)
      : injectRoot(graph, companyName, founderType);
    const newNodeCount = injectedNodes.length;
    const nodeCountChanged = newNodeCount !== prevNodeCountRef.current;
    prevNodeCountRef.current = newNodeCount;

    const existingMap = new Map(nodesRef.current.map((n) => [n.id, n]));
    const forceNodes = toForceNodes(injectedNodes, multiCompany);

    const container = containerRef.current;
    const cx = (container?.clientWidth ?? 800) / 2;
    const cy = (container?.clientHeight ?? 600) / 2;

    // In multi-company mode, spread initial root positions around center
    const rootNodes = forceNodes.filter((n) => n.category === "root");
    const spreadRadius = Math.min(container?.clientWidth ?? 800, container?.clientHeight ?? 600) * 0.25;

    for (let i = 0; i < forceNodes.length; i++) {
      const fn = forceNodes[i];
      const existing = existingMap.get(fn.id);
      if (existing) {
        fn.x = existing.x;
        fn.y = existing.y;
        fn.vx = existing.vx;
        fn.vy = existing.vy;
      } else if (fn.category === "root") {
        if (multiCompany) {
          // Spread roots around center
          const rootIdx = rootNodes.indexOf(fn);
          const angle = (2 * Math.PI * rootIdx) / rootNodes.length - Math.PI / 2;
          fn.x = cx + spreadRadius * Math.cos(angle);
          fn.y = cy + spreadRadius * Math.sin(angle);
        } else {
          fn.x = cx;
          fn.y = cy;
          fn.fx = cx;
          fn.fy = cy;
        }
      } else {
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
        } else if (multiCompany && fn.companyId) {
          // Spawn near company root
          const root = rootNodes.find((r) => r.companyId === fn.companyId);
          if (root && root.x != null) {
            fn.x = root.x + (Math.random() - 0.5) * 40;
            fn.y = (root.y ?? cy) + (Math.random() - 0.5) * 40;
          } else {
            fn.x = cx + (Math.random() - 0.5) * 60;
            fn.y = cy + (Math.random() - 0.5) * 60;
          }
        } else {
          fn.x = cx + (Math.random() - 0.5) * 60;
          fn.y = cy + (Math.random() - 0.5) * 60;
        }
      }

      const source = injectedNodes.find((n) => n.id === fn.id);
      if (source) fn.metrics = source.metrics;
    }

    computeDepths(forceNodes, injectedEdges);

    // Set initial rootX/rootY
    if (multiCompany) {
      syncRootPositions(forceNodes);
    }

    const nodeIdSet = new Set(forceNodes.map((n) => n.id));
    const forceLinks: ForceLinkDatum[] = injectedEdges
      .filter((e) => nodeIdSet.has(e.source) && nodeIdSet.has(e.target))
      .map((e) => ({ source: e.source, target: e.target }));

    nodesRef.current = forceNodes;
    linksRef.current = forceLinks;
    edgesRef.current = injectedEdges;

    if (simRef.current) {
      simRef.current.nodes(forceNodes);
      if (multiCompany) {
        simRef.current
          .force("link", forceLink<ForceNode, ForceLinkDatum>(forceLinks).id((d) => d.id).distance(50).strength(0.3));
      } else {
        simRef.current
          .force("link", forceLink<ForceNode, ForceLinkDatum>(forceLinks).id((d) => d.id).distance(60).strength(0.4))
          .force("radial", forceRadial<ForceNode>((d) => (d.category === "root" ? 0 : 80 + d.depth * 50), cx, cy).strength(0.25));
      }
      if (nodeCountChanged) {
        simRef.current.alpha(0.3).restart();
      }
    } else {
      let sim: Simulation<ForceNode, ForceLinkDatum>;
      if (multiCompany) {
        sim = forceSimulation(forceNodes)
          .force("charge", forceManyBody<ForceNode>().strength((d) => {
            // Strong repulsion between roots to push companies apart
            return (d as ForceNode).category === "root" ? -600 : -80;
          }))
          .force("link", forceLink<ForceNode, ForceLinkDatum>(forceLinks).id((d) => d.id).distance(50).strength(0.3))
          .force("center", forceCenter(cx, cy).strength(0.02))
          .force("collide", forceCollide<ForceNode>().radius((d) => d.radius + 3))
          .force("cluster", forceCluster(0.12))
          .alphaDecay(0.05)
          .velocityDecay(0.5);
      } else {
        sim = forceSimulation(forceNodes)
          .force("charge", forceManyBody<ForceNode>().strength(-120))
          .force("link", forceLink<ForceNode, ForceLinkDatum>(forceLinks).id((d) => d.id).distance(60).strength(0.4))
          .force("center", forceCenter(cx, cy).strength(0.05))
          .force("collide", forceCollide<ForceNode>().radius((d) => d.radius + 4))
          .force("radial", forceRadial<ForceNode>((d) => (d.category === "root" ? 0 : 80 + d.depth * 50), cx, cy).strength(0.25))
          .alphaDecay(0.05)
          .velocityDecay(0.5);
      }
      simRef.current = sim;
    }

    setSelected((prev) => {
      if (prev && !forceNodes.some((n) => n.id === prev.node.id)) return null;
      if (prev) {
        const updated = forceNodes.find((n) => n.id === prev.node.id);
        if (updated) return { ...prev, node: updated };
      }
      return prev;
    });
  }, [graph, companyName, multiCompany]);

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

      if (transformRef.current.x === 0 && transformRef.current.y === 0) {
        transformRef.current = { x: 0, y: 0, k: 1 };
      }

      const cx = w / 2;
      const cy = h / 2;
      if (simRef.current) {
        if (!multiCompanyRef.current) {
          simRef.current.force("center", forceCenter(cx, cy).strength(0.05));
          simRef.current.force("radial", forceRadial<ForceNode>(
            (d) => (d.category === "root" ? 0 : 80 + d.depth * 50), cx, cy,
          ).strength(0.25));
        } else {
          simRef.current.force("center", forceCenter(cx, cy).strength(0.02));
        }
      }
      if (!multiCompanyRef.current) {
        const root = nodesRef.current.find((n) => n.category === "root");
        if (root) {
          root.fx = cx;
          root.fy = cy;
        }
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

    // Click-to-pan on minimap
    const mm = minimapRef.current;
    if (mm && sx >= mm.x && sx <= mm.x + mm.w && sy >= mm.y && sy <= mm.y + mm.h) {
      const mmCX = mm.x + mm.w / 2;
      const mmCY = mm.y + mm.h / 2;
      const simX = (sx - mmCX) / mm.scale + mm.simCX;
      const simY = (sy - mmCY) / mm.scale + mm.simCY;
      const t = transformRef.current;
      t.x = mm.canvasW / 2 - simX * t.k;
      t.y = mm.canvasH / 2 - simY * t.k;
      return;
    }

    const hit = hitTest(sx, sy);

    if (hit) {
      const { sx: screenX, sy: screenY } = simToScreen(hit.x!, hit.y!);
      setSelected({ node: hit, sx: screenX, sy: screenY });
      // In multi-company mode, clicking a node focuses that company
      if (multiCompanyRef.current && hit.companyId && onFocusCompanyRef.current) {
        onFocusCompanyRef.current(hit.companyId);
      }
      return;
    }

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
