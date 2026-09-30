import React, { useMemo, useState } from 'react';
import { AlertTriangle, TrendingDown, Trash2 } from 'lucide-react';

interface LossPoint {
  step: number;
  loss: number;
}

interface LossChartProps {
  history: LossPoint[];
  currentLoss: number;
  onClearHistory: () => void;
  learningRate: number;
}

export const LossChart: React.FC<LossChartProps> = ({
  history,
  currentLoss,
  onClearHistory,
  learningRate,
}) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Check for divergence
  const isDiverging = useMemo(() => {
    if (history.length < 3) return false;
    const recent = history.slice(-3);
    const last = recent[recent.length - 1].loss;
    const first = history[0].loss;
    return (
      last > first * 3 && last > 2.0 // loss grew 3x and is notable
    );
  }, [history]);

  // Compute SVG coordinates
  const { pathData, areaData, points, minLoss, maxLoss } = useMemo(() => {
    if (history.length === 0) {
      return { pathData: '', areaData: '', points: [], minLoss: 0, maxLoss: 1 };
    }

    const losses = history.map((h) => h.loss);
    let minLoss = Math.min(...losses);
    let maxLoss = Math.max(...losses);

    // Give some breathing room
    if (maxLoss - minLoss < 0.001) {
      maxLoss += 0.05;
      minLoss = Math.max(0, minLoss - 0.05);
    } else {
      const padding = (maxLoss - minLoss) * 0.1;
      maxLoss += padding;
      minLoss = Math.max(0, minLoss - padding);
    }

    const width = 360;
    const height = 110;
    const padL = 35;
    const padR = 15;
    const padT = 15;
    const padB = 25;

    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    const pts = history.map((pt, i) => {
      const x = history.length === 1 ? padL + plotW / 2 : padL + (i / (history.length - 1)) * plotW;
      const normalizedY = (pt.loss - minLoss) / (maxLoss - minLoss);
      const y = padT + (1 - Math.min(1, Math.max(0, normalizedY))) * plotH;
      return { x, y, step: pt.step, loss: pt.loss };
    });

    let path = `M ${pts[0].x} ${pts[0].y}`;
    for (let i = 1; i < pts.length; i++) {
      path += ` L ${pts[i].x} ${pts[i].y}`;
    }

    const baselineY = padT + plotH;
    const area = `${path} L ${pts[pts.length - 1].x} ${baselineY} L ${pts[0].x} ${baselineY} Z`;

    return { pathData: path, areaData: area, points: pts, minLoss, maxLoss };
  }, [history]);

  return (
    <div className="bg-slate-900/60 rounded-xl border border-slate-800/80 p-4 flex flex-col gap-3">
      {/* Header */}
      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <TrendingDown className="w-4 h-4 text-emerald-400" />
          <span className="text-xs sm:text-sm font-semibold text-white">Loss History</span>
          <span className="text-xs font-mono text-cyan-300 tabular-nums">
            L = {currentLoss.toFixed(5)}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {history.length > 1 && (
            <button
              onClick={onClearHistory}
              className="text-slate-400 hover:text-slate-200 transition-colors p-1"
              title="Clear loss history"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Divergence banner */}
      {isDiverging && (
        <div className="p-2.5 rounded-lg bg-rose-950/50 border border-rose-800/80 flex items-start gap-2 text-xs text-rose-300">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-rose-200">Divergence Detected! </span>
            Loss is exploding. The learning rate (η={learningRate}) is likely too high, causing gradient descent to overshoot the valley. Try lowering η to 0.05 or 0.01.
          </div>
        </div>
      )}

      {/* Line Chart */}
      <div className="relative w-full h-[120px] bg-slate-950/70 rounded-lg border border-slate-800/70 overflow-hidden flex items-center justify-center">
        {history.length === 0 ? (
          <div className="text-xs text-slate-500 font-mono">
            Click &quot;Step (1x GD)&quot; to begin tracking loss
          </div>
        ) : (
          <svg viewBox="0 0 360 110" className="w-full h-full select-none">
            <defs>
              <linearGradient id="loss-fill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.3" />
                <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {/* Grid horizontal lines */}
            <line x1="35" y1="15" x2="345" y2="15" stroke="#1e293b" strokeDasharray="3 3" />
            <line x1="35" y1="52" x2="345" y2="52" stroke="#1e293b" strokeDasharray="3 3" />
            <line x1="35" y1="90" x2="345" y2="90" stroke="#1e293b" strokeWidth="1" />

            {/* Y-axis labels */}
            <text x="30" y="18" textAnchor="end" className="font-mono text-[9px] fill-slate-500">
              {maxLoss > 99 ? maxLoss.toExponential(1) : maxLoss.toFixed(2)}
            </text>
            <text x="30" y="93" textAnchor="end" className="font-mono text-[9px] fill-slate-500">
              {minLoss > 99 ? minLoss.toExponential(1) : minLoss.toFixed(2)}
            </text>

            {/* Area under curve */}
            <path d={areaData} fill="url(#loss-fill)" />

            {/* Path line */}
            <path
              d={pathData}
              fill="none"
              stroke="#06b6d4"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Points */}
            {points.map((pt, i) => (
              <circle
                key={i}
                cx={pt.x}
                cy={pt.y}
                r={hoveredIndex === i ? 4 : 2}
                fill={hoveredIndex === i ? '#38bdf8' : '#0891b2'}
                stroke="#0f172a"
                strokeWidth="1.5"
                className="cursor-pointer transition-all"
                onMouseEnter={() => setHoveredIndex(i)}
                onMouseLeave={() => setHoveredIndex(null)}
              />
            ))}
          </svg>
        )}

        {/* Hover Tooltip */}
        {hoveredIndex !== null && points[hoveredIndex] && (
          <div
            className="absolute top-2 right-2 bg-slate-900 border border-cyan-500/40 rounded px-2 py-1 text-[10px] font-mono text-cyan-300 shadow-lg pointer-events-none"
          >
            Step {points[hoveredIndex].step}: Loss = {points[hoveredIndex].loss.toFixed(5)}
          </div>
        )}
      </div>

      <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
        <span>Steps recorded: {history.length}</span>
        <span>
          {history.length > 0 &&
            `ΔLoss: ${(history[history.length - 1].loss - history[0].loss).toFixed(4)}`}
        </span>
      </div>
    </div>
  );
};
