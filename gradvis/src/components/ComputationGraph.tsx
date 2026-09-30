import React, { useMemo } from 'react';
import { GraphEdge, GraphNode } from '../engine/types';

interface ComputationGraphProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  isForwardAnimating: boolean;
  isBackwardAnimating: boolean;
  activeBackwardNodeId?: string;
  selectedNodeId?: string;
  onSelectNode: (node: GraphNode | null) => void;
}

export const ComputationGraph: React.FC<ComputationGraphProps> = ({
  nodes,
  edges,
  isForwardAnimating,
  isBackwardAnimating,
  activeBackwardNodeId,
  selectedNodeId,
  onSelectNode,
}) => {
  // Map nodes for fast coordinate lookup
  const nodeMap = useMemo(() => {
    const map = new Map<string, GraphNode>();
    nodes.forEach((n) => map.set(n.id, n));
    return map;
  }, [nodes]);

  // Node dimensions
  const nodeWidth = 114;
  const nodeHeight = 64;
  const halfW = nodeWidth / 2;
  const halfH = nodeHeight / 2;

  // ViewBox bounds calculated from nodes
  const { minX, minY, maxX, maxY } = useMemo(() => {
    let minX = 0;
    let minY = 0;
    let maxX = 1080;
    let maxY = 540;
    nodes.forEach((n) => {
      minX = Math.min(minX, n.x - 80);
      minY = Math.min(minY, n.y - 60);
      maxX = Math.max(maxX, n.x + 100);
      maxY = Math.max(maxY, n.y + 80);
    });
    return { minX, minY, maxX, maxY };
  }, [nodes]);

  const viewBoxWidth = maxX - minX + 40;
  const viewBoxHeight = maxY - minY + 40;

  return (
    <div className="relative w-full h-full bg-slate-950/60 rounded-xl border border-slate-800/80 overflow-hidden flex flex-col">
      {/* Visual Legend / Header */}
      <div className="px-4 py-2.5 bg-slate-900/70 border-b border-slate-800/80 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-3">
          <span className="font-medium text-slate-300">Computation Graph</span>
          <span className="text-slate-500 font-mono text-[11px] hidden sm:inline">
            Directed Acyclic Graph (DAG) · Left-to-Right Forward
          </span>
        </div>

        <div className="flex items-center gap-4 text-[11px] text-slate-400">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-xs bg-emerald-500/80"></span>
            <span>∂L &gt; 0</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-xs bg-rose-500/80"></span>
            <span>∂L &lt; 0</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-xs bg-amber-500/80"></span>
            <span>Trainable</span>
          </div>
        </div>
      </div>

      {/* SVG Canvas */}
      <div className="relative flex-1 w-full overflow-x-auto overflow-y-hidden select-none flex items-center justify-center p-2">
        <svg
          viewBox={`${minX} ${minY} ${viewBoxWidth} ${viewBoxHeight}`}
          className="w-full h-full min-h-[440px] max-h-[580px]"
          style={{ minWidth: '820px' }}
        >
          <defs>
            {/* Arrow marker for forward edges */}
            <marker
              id="arrow-forward"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569" />
            </marker>

            {/* Glowing filter for active backward nodes */}
            <filter id="glow-emerald" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>

            <filter id="glow-rose" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>

            <filter id="glow-cyan" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>

            {/* Animated gradients for forward pulse */}
            <linearGradient id="grad-pulse-fwd" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.1" />
              <stop offset="50%" stopColor="#38bdf8" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.1" />
            </linearGradient>

            {/* Animated gradients for backward pulse */}
            <linearGradient id="grad-pulse-bwd" x1="100%" y1="0%" x2="0%" y2="0%">
              <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.1" />
              <stop offset="50%" stopColor="#fbbf24" stopOpacity="1" />
              <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.1" />
            </linearGradient>
          </defs>

          {/* EDGES LAYER */}
          <g className="edges-layer">
            {edges.map((edge) => {
              const fromNode = nodeMap.get(edge.from);
              const toNode = nodeMap.get(edge.to);
              if (!fromNode || !toNode) return null;

              const x1 = fromNode.x + halfW;
              const y1 = fromNode.y;
              const x2 = toNode.x - halfW;
              const y2 = toNode.y;

              const dx = x2 - x1;
              const cx1 = x1 + dx * 0.45;
              const cy1 = y1;
              const cx2 = x1 + dx * 0.55;
              const cy2 = y2;
              const pathD = `M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`;

              const isEdgeConnectedToActive =
                activeBackwardNodeId === edge.to || activeBackwardNodeId === edge.from;

              return (
                <g key={edge.id} className="transition-all duration-300">
                  {/* Background base path */}
                  <path
                    d={pathD}
                    fill="none"
                    stroke="#1e293b"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    markerEnd="url(#arrow-forward)"
                  />

                  {/* Secondary subtle line */}
                  <path
                    d={pathD}
                    fill="none"
                    stroke="#334155"
                    strokeWidth="1.2"
                    strokeLinecap="round"
                  />

                  {/* Forward Animation Pulse */}
                  {isForwardAnimating && (
                    <path
                      d={pathD}
                      fill="none"
                      stroke="#38bdf8"
                      strokeWidth="3"
                      strokeDasharray="12 24"
                      className="animate-forward-dash"
                      opacity="0.85"
                    />
                  )}

                  {/* Backward Animation Pulse */}
                  {isBackwardAnimating && (
                    <path
                      d={pathD}
                      fill="none"
                      stroke="#fbbf24"
                      strokeWidth="3.5"
                      strokeDasharray="14 20"
                      className="animate-backward-dash"
                      opacity={isEdgeConnectedToActive ? '1' : '0.6'}
                    />
                  )}

                  {/* Optional Edge Label (e.g. for MLP weights) */}
                  {edge.label && (
                    <g transform={`translate(${(x1 + x2) / 2}, ${(y1 + y2) / 2 - 8})`}>
                      <rect
                        x="-24"
                        y="-9"
                        width="48"
                        height="16"
                        rx="4"
                        fill="#090d16"
                        stroke="#334155"
                        strokeWidth="0.8"
                      />
                      <text
                        textAnchor="middle"
                        dominantBaseline="central"
                        className="font-mono text-[9px] fill-slate-300"
                      >
                        {edge.label}
                      </text>
                    </g>
                  )}
                </g>
              );
            })}
          </g>

          {/* NODES LAYER */}
          <g className="nodes-layer">
            {nodes.map((node) => {
              const isSelected = selectedNodeId === node.id;
              const isActiveBackward = activeBackwardNodeId === node.id;

              // Color styling based on gradient
              const gradAbs = Math.abs(node.grad);
              const isPositive = node.grad > 0.00005;
              const isNegative = node.grad < -0.00005;

              let gradTextColor = '#94a3b8'; // slate-400
              let borderStroke = '#334155'; // slate-700
              let nodeBg = '#0f172a'; // slate-900

              if (isActiveBackward) {
                borderStroke = '#38bdf8'; // cyan-400
                nodeBg = '#082f49'; // cyan-950
              } else if (isPositive) {
                gradTextColor = '#34d399'; // emerald-400
                const alpha = Math.min(0.85, 0.2 + gradAbs * 0.4);
                borderStroke = `rgba(16, 185, 129, ${alpha})`;
                nodeBg = `rgba(6, 78, 59, 0.25)`;
              } else if (isNegative) {
                gradTextColor = '#fb7185'; // rose-400
                const alpha = Math.min(0.85, 0.2 + gradAbs * 0.4);
                borderStroke = `rgba(244, 63, 94, ${alpha})`;
                nodeBg = `rgba(136, 19, 55, 0.25)`;
              }

              if (node.isTrainable && !isActiveBackward && !isPositive && !isNegative) {
                borderStroke = '#d97706'; // amber-600
              }

              return (
                <g
                  key={node.id}
                  transform={`translate(${node.x}, ${node.y})`}
                  className="cursor-pointer transition-transform duration-200 hover:scale-105"
                  onClick={() => onSelectNode(isSelected ? null : node)}
                  filter={isActiveBackward ? 'url(#glow-cyan)' : undefined}
                >
                  {/* Card Background Container */}
                  <rect
                    x={-halfW}
                    y={-halfH}
                    width={nodeWidth}
                    height={nodeHeight}
                    rx="8"
                    fill={nodeBg}
                    stroke={isSelected ? '#38bdf8' : borderStroke}
                    strokeWidth={isSelected ? '2.5' : isActiveBackward ? '2' : '1.2'}
                    className="transition-colors duration-200"
                  />

                  {/* Header Row: Op Badge and Node Name */}
                  <g transform={`translate(${-halfW + 8}, ${-halfH + 14})`}>
                    {node.op ? (
                      <rect
                        x="0"
                        y="-8"
                        width="20"
                        height="14"
                        rx="3"
                        fill="#1e293b"
                        stroke="#475569"
                        strokeWidth="0.5"
                      />
                    ) : null}
                    {node.op ? (
                      <text
                        x="10"
                        y="-1"
                        textAnchor="middle"
                        dominantBaseline="central"
                        className="font-mono text-[9px] font-semibold fill-cyan-300"
                      >
                        {node.op === 'sqr' ? 'x²' : node.op}
                      </text>
                    ) : null}

                    <text
                      x={node.op ? 26 : 0}
                      y="-1"
                      dominantBaseline="central"
                      className="font-sans font-bold text-[11px] fill-slate-200 tracking-tight"
                    >
                      {node.name}
                    </text>

                    {node.isTrainable && (
                      <circle cx={nodeWidth - 24} cy="-1" r="3" fill="#f59e0b" />
                    )}
                    {node.isLoss && (
                      <circle cx={nodeWidth - 24} cy="-1" r="3" fill="#ec4899" />
                    )}
                  </g>

                  {/* Divider line inside card */}
                  <line
                    x1={-halfW + 6}
                    y1={-halfH + 22}
                    x2={halfW - 6}
                    y2={-halfH + 22}
                    stroke="#1e293b"
                    strokeWidth="1"
                  />

                  {/* Body: data stacked on top of grad */}
                  {/* Data value */}
                  <g transform={`translate(${-halfW + 8}, ${-halfH + 34})`}>
                    <text
                      x="0"
                      y="0"
                      className="font-mono text-[9.5px] fill-slate-400 select-none"
                    >
                      data:
                    </text>
                    <text
                      x={nodeWidth - 16}
                      y="0"
                      textAnchor="end"
                      className="font-mono text-[10px] font-semibold fill-slate-100 tabular-nums"
                    >
                      {node.data >= 0 ? ' ' : ''}
                      {node.data.toFixed(4)}
                    </text>
                  </g>

                  {/* Grad value */}
                  <g transform={`translate(${-halfW + 8}, ${-halfH + 48})`}>
                    <text
                      x="0"
                      y="0"
                      className="font-mono text-[9.5px] fill-slate-400 select-none"
                    >
                      grad:
                    </text>
                    <text
                      x={nodeWidth - 16}
                      y="0"
                      textAnchor="end"
                      style={{ fill: gradTextColor }}
                      className="font-mono text-[10px] font-bold tabular-nums"
                    >
                      {node.grad >= 0 ? '+' : ''}
                      {node.grad.toFixed(4)}
                    </text>
                  </g>

                  {/* Active backprop pulse ring */}
                  {isActiveBackward && (
                    <rect
                      x={-halfW - 3}
                      y={-halfH - 3}
                      width={nodeWidth + 6}
                      height={nodeHeight + 6}
                      rx="11"
                      fill="none"
                      stroke="#38bdf8"
                      strokeWidth="1.5"
                      strokeDasharray="4 4"
                      className="animate-spin-slow"
                    />
                  )}
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      {/* Floating Instructions/Help Footnote */}
      <div className="px-4 py-2 bg-slate-950/90 border-t border-slate-900 flex items-center justify-between text-[11px] text-slate-400">
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 font-semibold">Tip:</span>
          <span>Click any node to highlight its parents and inspect mathematical dependencies.</span>
        </div>
        <div className="hidden md:flex items-center gap-2 font-mono text-[10px] text-slate-500">
          <span>w = w - η · ∂L/∂w</span>
        </div>
      </div>
    </div>
  );
};
