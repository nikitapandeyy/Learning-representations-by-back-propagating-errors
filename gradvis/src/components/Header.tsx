import React from 'react';
import { ActivationType, ArchitectureType } from '../engine/types';
import { BookOpen, CheckCircle2, RotateCcw, Sparkles } from 'lucide-react';

interface HeaderProps {
  architecture: ArchitectureType;
  onArchitectureChange: (arch: ArchitectureType) => void;
  activation: ActivationType;
  onActivationChange: (act: ActivationType) => void;
  onOpenAutogradInfo: () => void;
  onToggleNumericalCheck: () => void;
  isNumericalCheckOpen: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  architecture,
  onArchitectureChange,
  activation,
  onActivationChange,
  onOpenAutogradInfo,
  onToggleNumericalCheck,
  isNumericalCheckOpen,
}) => {
  return (
    <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
        {/* Zone 1: Single text element wordmark */}
        <div className="flex items-center gap-3 shrink-0">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 text-white font-mono font-bold text-sm">
            ∂
          </div>
          <div>
            <h1 className="text-base sm:text-lg font-bold tracking-tight text-white flex items-center gap-2">
              Micrograd Autodiff
              <span className="text-xs font-normal text-cyan-400 font-mono hidden md:inline">
                reverse-mode engine
              </span>
            </h1>
          </div>
        </div>

        {/* Zone 2: Model & Activation selectors */}
        <div className="hidden sm:flex items-center gap-4">
          {/* Architecture Switcher */}
          <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs font-medium">
            <button
              onClick={() => onArchitectureChange('single-neuron')}
              className={`px-3 py-1.5 rounded-md transition-all whitespace-nowrap ${
                architecture === 'single-neuron'
                  ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/30 shadow-xs'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Single Neuron
            </button>
            <button
              onClick={() => onArchitectureChange('mlp-2-2-1')}
              className={`px-3 py-1.5 rounded-md transition-all whitespace-nowrap ${
                architecture === 'mlp-2-2-1'
                  ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/30 shadow-xs'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              2-Layer MLP
            </button>
          </div>

          {/* Activation Function Switcher */}
          <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs font-medium">
            <span className="text-slate-500 px-2 font-mono">f(z):</span>
            {(['tanh', 'sigmoid', 'relu'] as ActivationType[]).map((act) => (
              <button
                key={act}
                onClick={() => onActivationChange(act)}
                className={`px-2.5 py-1 rounded-md transition-all uppercase font-mono text-[11px] ${
                  activation === act
                    ? 'bg-blue-600/30 text-blue-300 font-semibold border border-blue-500/40 shadow-xs'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {act}
              </button>
            ))}
          </div>
        </div>

        {/* Zone 3: Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={onToggleNumericalCheck}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-all flex items-center gap-1.5 whitespace-nowrap ${
              isNumericalCheckOpen
                ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300'
                : 'bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-800 hover:text-white'
            }`}
            title="Verify analytic gradients with finite differences"
          >
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span className="hidden lg:inline">Finite Diff</span> Check
          </button>

          <button
            onClick={onOpenAutogradInfo}
            className="px-3 py-1.5 text-xs font-medium text-slate-300 bg-slate-900 border border-slate-800 rounded-lg hover:bg-slate-800 hover:text-white transition-all flex items-center gap-1.5 whitespace-nowrap"
          >
            <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
            <span>Guide</span>
          </button>
        </div>
      </div>

      {/* Mobile architecture & activation row */}
      <div className="sm:hidden flex items-center justify-between px-4 py-2 border-t border-slate-900 bg-slate-950 text-xs">
        <div className="flex gap-1">
          <button
            onClick={() => onArchitectureChange('single-neuron')}
            className={`px-2 py-1 rounded text-xs ${
              architecture === 'single-neuron' ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400'
            }`}
          >
            Single Neuron
          </button>
          <button
            onClick={() => onArchitectureChange('mlp-2-2-1')}
            className={`px-2 py-1 rounded text-xs ${
              architecture === 'mlp-2-2-1' ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400'
            }`}
          >
            2-Layer MLP
          </button>
        </div>

        <div className="flex gap-1">
          {(['tanh', 'sigmoid', 'relu'] as ActivationType[]).map((act) => (
            <button
              key={act}
              onClick={() => onActivationChange(act)}
              className={`px-2 py-1 rounded font-mono text-[10px] uppercase ${
                activation === act ? 'bg-blue-600/30 text-blue-300' : 'text-slate-400'
              }`}
            >
              {act}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
};
