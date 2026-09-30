import React from 'react';
import { NumericalCheckResult } from '../engine/types';
import { CheckCircle2, X, AlertTriangle, ArrowRight } from 'lucide-react';

interface NumericalCheckModalProps {
  isOpen: boolean;
  onClose: () => void;
  results: NumericalCheckResult[];
}

export const NumericalCheckModal: React.FC<NumericalCheckModalProps> = ({
  isOpen,
  onClose,
  results,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-xs animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-xl w-full p-5 flex flex-col gap-4">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center border border-emerald-500/30">
              <CheckCircle2 className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">
                Finite Difference Gradient Verification
              </h3>
              <p className="text-xs text-slate-400 font-mono">
                Numerical vs. Analytical Reverse-Mode Autodiff
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded-md hover:bg-slate-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Mathematical Explanation */}
        <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800/80 text-xs text-slate-300 leading-relaxed flex flex-col gap-2">
          <div className="font-semibold text-cyan-300">How Finite Differences Verify Autograd:</div>
          <p>
            By definition of the two-sided numerical derivative, for any parameter $\theta$ with perturbation $\epsilon = 10^{-4}$:
          </p>
          <div className="font-mono text-[11px] text-center bg-slate-900/90 py-1.5 px-3 rounded border border-slate-800 text-slate-200">
            ∂L / ∂θ ≈ [ L(θ + ε) - L(θ - ε) ] / (2 · ε)
          </div>
          <p className="text-slate-400 text-[11px]">
            If our analytical chain rule is implemented correctly, the difference between the analytical and numerical gradients will be approximately ≤ 10⁻⁴ (limited only by floating-point precision).
          </p>
        </div>

        {/* Results Table */}
        <div className="overflow-x-auto rounded-lg border border-slate-800">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="bg-slate-950 text-slate-400 font-mono text-[11px] border-b border-slate-800">
                <th className="py-2 px-3">Parameter</th>
                <th className="py-2 px-3 text-right">Analytic (Autograd)</th>
                <th className="py-2 px-3 text-right">Finite Diff (2-sided)</th>
                <th className="py-2 px-3 text-right">Absolute Error (|Δ|)</th>
                <th className="py-2 px-3 text-center">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 font-mono">
              {results.map((r) => (
                <tr key={r.paramName} className="hover:bg-slate-800/30">
                  <td className="py-2 px-3 font-bold text-white">{r.paramName}</td>
                  <td className="py-2 px-3 text-right text-cyan-300 tabular-nums">
                    {r.analyticGrad.toFixed(6)}
                  </td>
                  <td className="py-2 px-3 text-right text-emerald-300 tabular-nums">
                    {r.numericGrad.toFixed(6)}
                  </td>
                  <td className="py-2 px-3 text-right text-slate-300 tabular-nums">
                    {r.diff.toExponential(2)}
                  </td>
                  <td className="py-2 px-3 text-center">
                    {r.matches ? (
                      <span className="inline-flex items-center gap-1 text-[11px] text-emerald-400 font-sans font-medium">
                        <CheckCircle2 className="w-3.5 h-3.5" /> Match
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-[11px] text-rose-400 font-sans font-medium">
                        <AlertTriangle className="w-3.5 h-3.5" /> Mismatch
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="flex justify-end pt-2">
          <button
            onClick={onClose}
            className="px-4 py-1.5 text-xs font-medium text-white bg-cyan-600 hover:bg-cyan-500 rounded-lg transition-colors"
          >
            Close Inspector
          </button>
        </div>
      </div>
    </div>
  );
};
