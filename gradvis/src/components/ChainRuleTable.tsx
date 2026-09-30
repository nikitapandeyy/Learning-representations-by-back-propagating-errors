import React from 'react';
import { DerivativeStep } from '../engine/types';
import { Calculator, HelpCircle } from 'lucide-react';

interface ChainRuleTableProps {
  steps: DerivativeStep[];
  activeNodeId?: string;
  onHoverNode?: (nodeId: string | null) => void;
}

export const ChainRuleTable: React.FC<ChainRuleTableProps> = ({
  steps,
  activeNodeId,
  onHoverNode,
}) => {
  return (
    <div className="bg-slate-900/60 rounded-xl border border-slate-800/80 overflow-hidden flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 bg-slate-900/80 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Calculator className="w-4 h-4 text-cyan-400" />
          <h3 className="text-xs sm:text-sm font-semibold text-white">
            Reverse-Mode Chain Rule Breakdown
          </h3>
        </div>
        <span className="text-[11px] text-slate-400 font-mono hidden sm:inline">
          Symbolic ↔ Numeric Verification
        </span>
      </div>

      {/* Table Container */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="bg-slate-950/60 text-slate-400 font-mono text-[11px] border-b border-slate-800">
              <th className="py-2.5 px-3 font-semibold">Derivative Target</th>
              <th className="py-2.5 px-3 font-semibold">Analytical Formula</th>
              <th className="py-2.5 px-3 font-semibold">Substitution (Current Values)</th>
              <th className="py-2.5 px-3 font-semibold text-right">Computed Gradient</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50">
            {steps.map((step, idx) => {
              const isActive = activeNodeId === step.nodeId;
              const isPositive = step.accumulatedGrad > 0.00005;
              const isNegative = step.accumulatedGrad < -0.00005;

              return (
                <tr
                  key={idx}
                  onMouseEnter={() => onHoverNode && onHoverNode(step.nodeId)}
                  onMouseLeave={() => onHoverNode && onHoverNode(null)}
                  className={`transition-colors duration-150 cursor-pointer ${
                    isActive
                      ? 'bg-cyan-950/40 text-cyan-200'
                      : 'hover:bg-slate-800/40 text-slate-300'
                  }`}
                >
                  {/* Target Column */}
                  <td className="py-2.5 px-3 font-mono font-bold text-slate-200 whitespace-nowrap">
                    <div className="flex items-center gap-2">
                      {isActive && (
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse"></span>
                      )}
                      <span>{step.target}</span>
                    </div>
                  </td>

                  {/* Symbolic Formula */}
                  <td className="py-2.5 px-3 font-mono text-cyan-300/90 whitespace-nowrap">
                    {step.formula}
                  </td>

                  {/* Numeric Substitution */}
                  <td className="py-2.5 px-3 font-mono text-slate-300 text-[11px] whitespace-nowrap">
                    {step.substitution}
                  </td>

                  {/* Computed Grad */}
                  <td className="py-2.5 px-3 font-mono font-bold text-right tabular-nums whitespace-nowrap">
                    <span
                      className={`inline-block px-2 py-0.5 rounded text-xs ${
                        isPositive
                          ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-800/50'
                          : isNegative
                          ? 'bg-rose-950/60 text-rose-400 border border-rose-800/50'
                          : 'bg-slate-800/60 text-slate-400'
                      }`}
                    >
                      {step.accumulatedGrad >= 0 ? '+' : ''}
                      {step.accumulatedGrad.toFixed(4)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Explanatory footer */}
      <div className="px-4 py-2 bg-slate-950/80 border-t border-slate-900 text-[11px] text-slate-400 flex items-center justify-between">
        <span className="flex items-center gap-1.5">
          <HelpCircle className="w-3.5 h-3.5 text-slate-500" />
          <span>Reverse-mode computes backward from loss to inputs in $O(1)$ passes via topological sort.</span>
        </span>
        <span className="font-mono text-slate-500 text-[10px] hidden md:inline">
          ∂L/∂w = ∂L/∂z · ∂z/∂w
        </span>
      </div>
    </div>
  );
};
