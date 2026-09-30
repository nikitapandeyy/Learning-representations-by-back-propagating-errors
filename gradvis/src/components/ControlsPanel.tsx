import React from 'react';
import { ArchitectureType, MLPParams, SingleNeuronParams } from '../engine/types';
import {
  Play,
  Pause,
  RotateCcw,
  Sparkles,
  ArrowRight,
  TrendingDown,
  Layers,
  Sliders,
  Dices,
} from 'lucide-react';

interface ControlsPanelProps {
  architecture: ArchitectureType;
  singleParams: SingleNeuronParams;
  mlpParams: MLPParams;
  onUpdateSingleParam: (key: keyof SingleNeuronParams, val: number) => void;
  onUpdateMLPParam: (key: keyof MLPParams, val: number) => void;
  onComputeGradients: () => void;
  onStep: () => void;
  onAutoPlayToggle: () => void;
  isAutoPlaying: boolean;
  onReset: () => void;
  onRandomize: () => void;
  isBackwardAnimating: boolean;
  stepCount: number;
}

export const ControlsPanel: React.FC<ControlsPanelProps> = ({
  architecture,
  singleParams,
  mlpParams,
  onUpdateSingleParam,
  onUpdateMLPParam,
  onComputeGradients,
  onStep,
  onAutoPlayToggle,
  isAutoPlaying,
  onReset,
  onRandomize,
  isBackwardAnimating,
  stepCount,
}) => {
  // Convert learning rate between log slider [0..100] and actual value [0.001..1.0]
  // log scale: val = 10^( -3 + 3 * (slider/100) )
  const lrToSlider = (lr: number) => {
    const clamped = Math.max(0.001, Math.min(1.0, lr));
    return Math.round(((Math.log10(clamped) + 3) / 3) * 100);
  };

  const sliderToLr = (sliderVal: number) => {
    const exp = -3 + (3 * sliderVal) / 100;
    const raw = Math.pow(10, exp);
    // round to 4 decimal places
    return Math.round(raw * 1000) / 1000;
  };

  const currentLr =
    architecture === 'single-neuron'
      ? singleParams.learningRate
      : mlpParams.learningRate;

  const handleLrSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newLr = sliderToLr(parseFloat(e.target.value));
    if (architecture === 'single-neuron') {
      onUpdateSingleParam('learningRate', newLr);
    } else {
      onUpdateMLPParam('learningRate', newLr);
    }
  };

  const handleLrNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    if (!isNaN(val) && val >= 0.0001 && val <= 5.0) {
      if (architecture === 'single-neuron') {
        onUpdateSingleParam('learningRate', val);
      } else {
        onUpdateMLPParam('learningRate', val);
      }
    }
  };

  return (
    <div className="bg-slate-900/60 rounded-xl border border-slate-800/80 p-4 flex flex-col gap-4">
      {/* Top action row */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Sliders className="w-4 h-4 text-cyan-400" />
          <span className="text-sm font-semibold text-white">Network Parameters</span>
          <span className="text-xs text-slate-400 font-mono">
            (Step #{stepCount})
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onRandomize}
            className="px-2.5 py-1 text-xs text-slate-300 hover:text-white bg-slate-800/80 hover:bg-slate-800 border border-slate-700/60 rounded-md transition-colors flex items-center gap-1.5"
            title="Randomize weights"
          >
            <Dices className="w-3.5 h-3.5" />
            <span>Randomize</span>
          </button>
          <button
            onClick={onReset}
            className="px-2.5 py-1 text-xs text-slate-300 hover:text-white bg-slate-800/80 hover:bg-slate-800 border border-slate-700/60 rounded-md transition-colors flex items-center gap-1.5"
            title="Reset to default initial weights and inputs"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Main Execution Controls: Compute Gradients, Step, Auto-Train */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
        <button
          onClick={onComputeGradients}
          disabled={isBackwardAnimating}
          className={`py-2 px-3 rounded-lg border font-medium text-xs flex items-center justify-center gap-2 transition-all ${
            isBackwardAnimating
              ? 'bg-cyan-950/60 border-cyan-800 text-cyan-400 cursor-wait'
              : 'bg-gradient-to-r from-cyan-600/30 to-blue-600/30 border-cyan-500/50 hover:border-cyan-400 text-cyan-200 hover:bg-cyan-500/20 shadow-xs'
          }`}
        >
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          <span>Compute Gradients</span>
        </button>

        <button
          onClick={onStep}
          disabled={isAutoPlaying}
          className="py-2 px-3 rounded-lg border border-emerald-500/50 bg-emerald-600/25 hover:bg-emerald-600/35 hover:border-emerald-400 text-emerald-200 font-semibold text-xs flex items-center justify-center gap-2 transition-all shadow-xs"
          title="Update weights: w = w - lr * dL/dw"
        >
          <ArrowRight className="w-3.5 h-3.5 text-emerald-400" />
          <span>Step (1x GD)</span>
        </button>

        <button
          onClick={onAutoPlayToggle}
          className={`py-2 px-3 rounded-lg border font-medium text-xs flex items-center justify-center gap-2 transition-all shadow-xs ${
            isAutoPlaying
              ? 'bg-amber-600/25 border-amber-500 text-amber-200'
              : 'bg-slate-800/90 border-slate-700 text-slate-200 hover:bg-slate-800 hover:text-white'
          }`}
        >
          {isAutoPlaying ? (
            <>
              <Pause className="w-3.5 h-3.5 text-amber-400" />
              <span>Pause Training</span>
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 text-emerald-400" />
              <span>Auto Train</span>
            </>
          )}
        </button>
      </div>

      {/* Learning Rate Slider */}
      <div className="bg-slate-950/40 p-3 rounded-lg border border-slate-800/70">
        <div className="flex items-center justify-between text-xs mb-1.5">
          <label htmlFor="learning-rate-slider" className="text-slate-300 font-medium flex items-center gap-1.5">
            <span>Learning Rate (η)</span>
            <span className="text-[10px] text-slate-500 font-mono">(log scale)</span>
          </label>
          <div className="flex items-center gap-1">
            <input
              id="learning-rate-input"
              type="number"
              min="0.0001"
              max="5.0"
              step="0.001"
              value={currentLr}
              onChange={handleLrNumberChange}
              aria-label="Learning rate numeric value"
              className="w-20 bg-slate-900 border border-slate-700 rounded px-1.5 py-0.5 text-xs font-mono text-cyan-300 text-right focus:outline-none focus:border-cyan-500"
            />
          </div>
        </div>
        <input
          id="learning-rate-slider"
          type="range"
          min="0"
          max="100"
          value={lrToSlider(currentLr)}
          onChange={handleLrSliderChange}
          aria-label="Learning rate log scale slider"
          className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
        />
        <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
          <span>0.001</span>
          <span>0.01</span>
          <span>0.1</span>
          <span>1.0</span>
        </div>
      </div>

      {/* Sliders Grid */}
      {architecture === 'single-neuron' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {/* Inputs Section */}
          <div className="bg-slate-950/40 p-3 rounded-lg border border-slate-800/70 flex flex-col gap-2.5">
            <div className="flex items-center justify-between text-xs font-semibold text-slate-300 pb-1 border-b border-slate-800/60">
              <span>Input Features &amp; Target</span>
              <span className="text-[10px] text-slate-500 font-mono">constant per step</span>
            </div>

            {/* x1 */}
            <ParamSlider
              label="Input x₁"
              value={singleParams.x1}
              min={-5}
              max={5}
              step={0.1}
              onChange={(val) => onUpdateSingleParam('x1', val)}
            />

            {/* x2 */}
            <ParamSlider
              label="Input x₂"
              value={singleParams.x2}
              min={-5}
              max={5}
              step={0.1}
              onChange={(val) => onUpdateSingleParam('x2', val)}
            />

            {/* yTarget */}
            <ParamSlider
              label="Target y_target"
              value={singleParams.yTarget}
              min={-1}
              max={1}
              step={0.05}
              color="text-pink-400"
              onChange={(val) => onUpdateSingleParam('yTarget', val)}
            />
          </div>

          {/* Trainable Parameters Section */}
          <div className="bg-slate-950/40 p-3 rounded-lg border border-amber-900/30 flex flex-col gap-2.5">
            <div className="flex items-center justify-between text-xs font-semibold text-amber-300 pb-1 border-b border-amber-900/40">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-amber-400"></span>
                <span>Trainable Weights &amp; Bias</span>
              </span>
              <span className="text-[10px] text-amber-500/80 font-mono">updated by GD</span>
            </div>

            {/* w1 */}
            <ParamSlider
              label="Weight w₁"
              value={singleParams.w1}
              min={-5}
              max={5}
              step={0.05}
              color="text-amber-300"
              onChange={(val) => onUpdateSingleParam('w1', val)}
            />

            {/* w2 */}
            <ParamSlider
              label="Weight w₂"
              value={singleParams.w2}
              min={-5}
              max={5}
              step={0.05}
              color="text-amber-300"
              onChange={(val) => onUpdateSingleParam('w2', val)}
            />

            {/* b */}
            <ParamSlider
              label="Bias b"
              value={singleParams.b}
              min={-5}
              max={5}
              step={0.05}
              color="text-amber-300"
              onChange={(val) => onUpdateSingleParam('b', val)}
            />
          </div>
        </div>
      ) : (
        /* MLP Parameters View */
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          {/* Inputs & Target */}
          <div className="bg-slate-950/40 p-3 rounded-lg border border-slate-800/70 flex flex-col gap-2">
            <span className="font-semibold text-slate-300 pb-1 border-b border-slate-800">
              Inputs &amp; Target
            </span>
            <ParamSlider
              label="x₁"
              value={mlpParams.x1}
              min={-3}
              max={3}
              step={0.1}
              onChange={(v) => onUpdateMLPParam('x1', v)}
            />
            <ParamSlider
              label="x₂"
              value={mlpParams.x2}
              min={-3}
              max={3}
              step={0.1}
              onChange={(v) => onUpdateMLPParam('x2', v)}
            />
            <ParamSlider
              label="Target y"
              value={mlpParams.yTarget}
              min={-1}
              max={1}
              step={0.05}
              color="text-pink-400"
              onChange={(v) => onUpdateMLPParam('yTarget', v)}
            />
          </div>

          {/* Hidden Layer (Layer 1) */}
          <div className="bg-slate-950/40 p-3 rounded-lg border border-amber-900/30 flex flex-col gap-2">
            <span className="font-semibold text-amber-300 pb-1 border-b border-amber-900/40">
              Layer 1 (Hidden Weights)
            </span>
            <ParamSlider
              label="w₁₁"
              value={mlpParams.w11}
              min={-3}
              max={3}
              step={0.05}
              onChange={(v) => onUpdateMLPParam('w11', v)}
            />
            <ParamSlider
              label="w₁₂"
              value={mlpParams.w12}
              min={-3}
              max={3}
              step={0.05}
              onChange={(v) => onUpdateMLPParam('w12', v)}
            />
            <ParamSlider
              label="w₂₁"
              value={mlpParams.w21}
              min={-3}
              max={3}
              step={0.05}
              onChange={(v) => onUpdateMLPParam('w21', v)}
            />
            <ParamSlider
              label="w₂₂"
              value={mlpParams.w22}
              min={-3}
              max={3}
              step={0.05}
              onChange={(v) => onUpdateMLPParam('w22', v)}
            />
          </div>

          {/* Output Layer (Layer 2) */}
          <div className="bg-slate-950/40 p-3 rounded-lg border border-amber-900/30 flex flex-col gap-2">
            <span className="font-semibold text-amber-300 pb-1 border-b border-amber-900/40">
              Layer 2 (Output Weights)
            </span>
            <ParamSlider
              label="v₁ (a₁→out)"
              value={mlpParams.v1}
              min={-3}
              max={3}
              step={0.05}
              onChange={(v) => onUpdateMLPParam('v1', v)}
            />
            <ParamSlider
              label="v₂ (a₂→out)"
              value={mlpParams.v2}
              min={-3}
              max={3}
              step={0.05}
              onChange={(v) => onUpdateMLPParam('v2', v)}
            />
            <ParamSlider
              label="b_out"
              value={mlpParams.bOut}
              min={-3}
              max={3}
              step={0.05}
              onChange={(v) => onUpdateMLPParam('bOut', v)}
            />
          </div>
        </div>
      )}
    </div>
  );
};

interface ParamSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  color?: string;
  onChange: (val: number) => void;
}

const ParamSlider: React.FC<ParamSliderProps> = ({
  label,
  value,
  min,
  max,
  step,
  color = 'text-slate-200',
  onChange,
}) => {
  const sliderId = `slider-${label.replace(/[^a-zA-Z0-9]/g, '-')}`;
  const inputId = `input-${label.replace(/[^a-zA-Z0-9]/g, '-')}`;

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-xs">
        <label htmlFor={sliderId} className="text-slate-300 font-medium">
          {label}
        </label>
        <div className="flex items-center gap-1">
          <input
            id={inputId}
            type="number"
            min={min}
            max={max}
            step={step}
            value={value}
            aria-label={`${label} numeric value`}
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              if (!isNaN(val)) onChange(val);
            }}
            className={`w-16 bg-slate-900 border border-slate-700 rounded px-1 py-0.5 text-xs font-mono text-right focus:outline-none focus:border-cyan-500 tabular-nums ${color}`}
          />
        </div>
      </div>
      <input
        id={sliderId}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={`${label} slider`}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
      />
    </div>
  );
};
