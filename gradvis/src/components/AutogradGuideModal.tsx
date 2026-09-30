import React from 'react';
import { X, BookOpen, GitBranch, ArrowLeftRight, Activity } from 'lucide-react';

interface AutogradGuideModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const AutogradGuideModal: React.FC<AutogradGuideModalProps> = ({
  isOpen,
  onClose,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-2xl w-full max-h-[85vh] overflow-y-auto p-6 flex flex-col gap-5">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/20 text-cyan-400 flex items-center justify-center border border-cyan-500/30">
              <BookOpen className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">
                Understanding Reverse-Mode Automatic Differentiation
              </h3>
              <p className="text-xs text-slate-400 font-mono">
                The core engine powering PyTorch, JAX, and Micrograd
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded-md hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content sections */}
        <div className="space-y-4 text-xs text-slate-300 leading-relaxed">
          <div className="bg-slate-950/70 p-4 rounded-lg border border-slate-800 space-y-2">
            <h4 className="font-semibold text-cyan-300 text-sm flex items-center gap-1.5">
              <Activity className="w-4 h-4" /> What is Autograd?
            </h4>
            <p>
              Automatic differentiation (autograd) is an algorithmic technique for evaluating exact derivatives of numerical functions specified by computer programs. Unlike symbolic differentiation (which produces unwieldy expression formulas) or finite differences (which suffer from numerical floating-point truncation error), autograd executes in constant-factor time relative to the forward evaluation.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="font-semibold text-white text-sm flex items-center gap-1.5">
              <GitBranch className="w-4 h-4 text-amber-400" />
              Why Reverse Mode (Backpropagation)?
            </h4>
            <p>
              Neural networks typically have millions of inputs/weights ($N \gg 1$) but only a single scalar output (the Loss $L$).
            </p>
            <ul className="list-disc pl-5 space-y-1 text-slate-400">
              <li>
                <strong className="text-slate-200">Forward mode</strong> requires $N$ separate forward passes to calculate all $\partial L / \partial w_i$.
              </li>
              <li>
                <strong className="text-slate-200">Reverse mode</strong> needs only <strong>ONE forward pass</strong> to compute data values and build the DAG, followed by <strong>ONE backward pass</strong> traversing the graph in reverse topological order, giving gradients for all parameters simultaneously!
              </li>
            </ul>
          </div>

          <div className="space-y-2">
            <h4 className="font-semibold text-white text-sm flex items-center gap-1.5">
              <ArrowLeftRight className="w-4 h-4 text-emerald-400" />
              The Computation Node Contract
            </h4>
            <p>
              In Andrej Karpathy&apos;s famous <code className="font-mono text-cyan-300">micrograd</code> engine, every scalar is wrapped in a <code className="font-mono text-cyan-300">Value</code> object holding two numbers:
            </p>
            <div className="font-mono bg-slate-950 p-3 rounded-lg border border-slate-800 text-[11px] text-slate-300 space-y-1">
              <div>class Value:</div>
              <div className="pl-4">data: float  <span className="text-slate-500"># value produced during forward pass</span></div>
              <div className="pl-4">grad: float  <span className="text-slate-500"># dL / d(this node), accumulated backward</span></div>
              <div className="pl-4">_prev: set(Value) <span className="text-slate-500"># children in the computation graph</span></div>
              <div className="pl-4">_backward: callable <span className="text-slate-500"># applies local chain rule</span></div>
            </div>
          </div>
        </div>

        <div className="flex justify-end pt-2 border-t border-slate-800">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-semibold text-slate-900 bg-cyan-400 hover:bg-cyan-300 rounded-lg transition-colors"
          >
            Got it, let&apos;s explore!
          </button>
        </div>
      </div>
    </div>
  );
};
