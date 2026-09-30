import React from 'react';
import { AlertCircle, ZapOff, Sparkles } from 'lucide-react';
import { ActivationType } from '../engine/types';

interface VanishingGradientAlertProps {
  isVanishing: boolean;
  isReluDead: boolean;
  z: number;
  da_dz: number;
  activation: ActivationType;
  onSwitchToRelu: () => void;
}

export const VanishingGradientAlert: React.FC<VanishingGradientAlertProps> = ({
  isVanishing,
  isReluDead,
  z,
  da_dz,
  activation,
  onSwitchToRelu,
}) => {
  if (!isVanishing && !isReluDead) return null;

  if (isReluDead) {
    return (
      <div className="bg-amber-950/40 border border-amber-800/70 rounded-xl p-3.5 flex items-start gap-3 text-xs">
        <ZapOff className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        <div className="flex flex-col gap-1">
          <div className="font-semibold text-amber-200">
            Dying ReLU Detected! (z = {z.toFixed(4)} ≤ 0)
          </div>
          <p className="text-amber-300/90 leading-relaxed">
            Because pre-activation <code className="font-mono bg-amber-900/40 px-1 py-0.5 rounded">z ≤ 0</code>, the ReLU derivative is strictly <code className="font-mono">∂a/∂z = 0</code>. Gradients cannot flow backward through this neuron, freezing weight updates until <code className="font-mono">z</code> becomes positive again.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-rose-950/40 border border-rose-800/70 rounded-xl p-3.5 flex items-start gap-3 text-xs">
      <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
      <div className="flex flex-col gap-1.5 flex-1">
        <div className="flex items-center justify-between">
          <span className="font-semibold text-rose-200">
            Why does the gradient vanish? Saturation at z = {z.toFixed(4)}
          </span>
          <span className="font-mono text-[10px] bg-rose-900/60 text-rose-300 px-1.5 py-0.5 rounded border border-rose-700/50">
            da/dz = {da_dz.toFixed(5)} ≈ 0
          </span>
        </div>
        <p className="text-rose-200/90 leading-relaxed">
          The activation function <span className="font-semibold text-white">{activation}</span> is saturated. As $|z|$ becomes large, the slope <code className="font-mono bg-rose-900/40 px-1 py-0.5 rounded">da/dz = {activation === 'tanh' ? '1 - a²' : 'a·(1-a)'}</code> approaches zero. Because of the chain rule:
        </p>
        <div className="font-mono bg-slate-950/70 p-2 rounded border border-rose-900/50 text-[11px] text-cyan-300">
          ∂L/∂w = (∂L/∂a) · <span className="text-rose-400 font-bold">(∂a/∂z)</span> · x → 0.0000
        </div>
        <p className="text-slate-300 text-[11px]">
          Even if Loss is large, weights will barely update! In deep neural networks, this multiplication happens across dozens of layers, causing earlier layers to completely stop learning. This is why non-saturating functions like <strong>ReLU</strong> became standard in deep learning.
        </p>
        {activation !== 'relu' && (
          <button
            onClick={onSwitchToRelu}
            className="self-start mt-1 px-2.5 py-1 bg-rose-900/50 hover:bg-rose-900/80 border border-rose-700 text-rose-200 rounded text-[11px] font-medium transition-colors flex items-center gap-1.5"
          >
            <Sparkles className="w-3 h-3 text-rose-300" />
            Switch to ReLU to compare
          </button>
        )}
      </div>
    </div>
  );
};
