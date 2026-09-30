import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  ActivationType,
  ArchitectureType,
  GraphNode,
  MLPParams,
  SingleNeuronParams,
} from './engine/types';
import {
  computeNumericalGradientSingleNeuron,
  evaluateMLP,
  evaluateSingleNeuron,
} from './engine/autodiff';
import { Header } from './components/Header';
import { ComputationGraph } from './components/ComputationGraph';
import { ControlsPanel } from './components/ControlsPanel';
import { ChainRuleTable } from './components/ChainRuleTable';
import { LossChart } from './components/LossChart';
import { VanishingGradientAlert } from './components/VanishingGradientAlert';
import { NumericalCheckModal } from './components/NumericalCheckModal';
import { AutogradGuideModal } from './components/AutogradGuideModal';
import { Info, Gauge, Crosshair, Check, ChevronRight } from 'lucide-react';

export default function App() {
  // Global settings
  const [architecture, setArchitecture] = useState<ArchitectureType>('single-neuron');
  const [activation, setActivation] = useState<ActivationType>('tanh');

  // Single Neuron Parameters
  const initialSingleParams: SingleNeuronParams = {
    x1: 2.0,
    x2: -1.0,
    w1: -1.5,
    w2: 1.0,
    b: 0.5,
    yTarget: 1.0,
    activation: 'tanh',
    learningRate: 0.1,
  };
  const [singleParams, setSingleParams] = useState<SingleNeuronParams>(initialSingleParams);

  // MLP Parameters
  const initialMLPParams: MLPParams = {
    x1: 1.0,
    x2: -1.0,
    w11: 0.8,
    w12: -0.5,
    b1: 0.2,
    w21: -0.4,
    w22: 0.9,
    b2: -0.1,
    v1: 1.2,
    v2: -0.8,
    bOut: 0.1,
    yTarget: 0.8,
    activation: 'tanh',
    learningRate: 0.1,
  };
  const [mlpParams, setMLPParams] = useState<MLPParams>(initialMLPParams);

  // Sync activation change into params
  useEffect(() => {
    setSingleParams((prev) => ({ ...prev, activation }));
    setMLPParams((prev) => ({ ...prev, activation }));
  }, [activation]);

  // Backward Animation State
  // 1.0 = fully calculated, 0..1 = animation progress
  const [backwardProgress, setBackwardProgress] = useState<number>(1.0);
  const [isBackwardAnimating, setIsBackwardAnimating] = useState<boolean>(false);
  const [activeBackwardNodeId, setActiveBackwardNodeId] = useState<string | undefined>(undefined);
  const backwardAnimTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Forward animation pulse
  const [isForwardAnimating, setIsForwardAnimating] = useState<boolean>(false);
  const forwardAnimTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Selected node inspection
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  // Loss History Tracking
  const [lossHistory, setLossHistory] = useState<{ step: number; loss: number }[]>([]);
  const [stepCount, setStepCount] = useState<number>(0);

  // Auto-play training loop
  const [isAutoPlaying, setIsAutoPlaying] = useState<boolean>(false);
  const autoPlayIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Modals
  const [isAutogradInfoOpen, setIsAutogradInfoOpen] = useState<boolean>(false);
  const [isNumericalCheckOpen, setIsNumericalCheckOpen] = useState<boolean>(false);

  // Forward pulse trigger on slider inputs
  const triggerForwardPulse = useCallback(() => {
    setIsForwardAnimating(true);
    if (forwardAnimTimeoutRef.current) clearTimeout(forwardAnimTimeoutRef.current);
    forwardAnimTimeoutRef.current = setTimeout(() => {
      setIsForwardAnimating(false);
    }, 450);
  }, []);

  // Compute graph data
  const currentEvaluation = useMemo(() => {
    if (architecture === 'single-neuron') {
      return evaluateSingleNeuron(singleParams, backwardProgress);
    } else {
      return evaluateMLP(mlpParams, backwardProgress);
    }
  }, [architecture, singleParams, mlpParams, backwardProgress]);

  // Record initial loss point if history is empty
  useEffect(() => {
    if (lossHistory.length === 0) {
      setLossHistory([{ step: 0, loss: currentEvaluation.loss }]);
    }
  }, [currentEvaluation.loss, lossHistory.length]);

  // Update single neuron parameter handler
  const handleUpdateSingleParam = (key: keyof SingleNeuronParams, val: number) => {
    setSingleParams((prev) => ({ ...prev, [key]: val }));
    triggerForwardPulse();
  };

  // Update MLP parameter handler
  const handleUpdateMLPParam = (key: keyof MLPParams, val: number) => {
    setMLPParams((prev) => ({ ...prev, [key]: val }));
    triggerForwardPulse();
  };

  // Trigger backward pass animation
  const handleComputeGradients = useCallback(() => {
    if (isBackwardAnimating) return;
    setIsBackwardAnimating(true);
    setBackwardProgress(0);

    const stepsSingle = [
      { progress: 0.1, nodeId: 'L' },
      { progress: 0.25, nodeId: 'diff' },
      { progress: 0.45, nodeId: 'a' },
      { progress: 0.65, nodeId: 'z' },
      { progress: 0.8, nodeId: 'sumWx' },
      { progress: 0.9, nodeId: 'w1x1' },
      { progress: 1.0, nodeId: 'w1' },
    ];

    const stepsMLP = [
      { progress: 0.15, nodeId: 'L' },
      { progress: 0.35, nodeId: 'aOut' },
      { progress: 0.55, nodeId: 'zOut' },
      { progress: 0.75, nodeId: 'a1' },
      { progress: 0.9, nodeId: 'z1' },
      { progress: 1.0, nodeId: 'x1' },
    ];

    const seq = architecture === 'single-neuron' ? stepsSingle : stepsMLP;
    let index = 0;

    const runNext = () => {
      if (index < seq.length) {
        const item = seq[index];
        setBackwardProgress(item.progress);
        setActiveBackwardNodeId(item.nodeId);
        index++;
        backwardAnimTimeoutRef.current = setTimeout(runNext, 380);
      } else {
        setBackwardProgress(1.0);
        setActiveBackwardNodeId(undefined);
        setIsBackwardAnimating(false);
      }
    };

    runNext();
  }, [architecture, isBackwardAnimating]);

  // Single Gradient Descent Step
  const handleStep = useCallback(() => {
    if (architecture === 'single-neuron') {
      // Evaluate full gradients
      const evalFull = evaluateSingleNeuron(singleParams, 1);
      const lr = singleParams.learningRate;

      // Gradient descent update rule: w = w - lr * grad
      const newW1 = singleParams.w1 - lr * evalFull.dL_dw1;
      const newW2 = singleParams.w2 - lr * evalFull.dL_dw2;
      const newB = singleParams.b - lr * evalFull.dL_db;

      const nextParams = {
        ...singleParams,
        w1: newW1,
        w2: newW2,
        b: newB,
      };

      const nextEval = evaluateSingleNeuron(nextParams, 1);
      const nextStep = stepCount + 1;

      setSingleParams(nextParams);
      setStepCount(nextStep);
      setLossHistory((prev) => [...prev, { step: nextStep, loss: nextEval.loss }]);
    } else {
      // MLP Update
      const evalMLP = evaluateMLP(mlpParams, 1);
      const lr = mlpParams.learningRate;

      const nextMLP: MLPParams = {
        ...mlpParams,
        w11: mlpParams.w11 - lr * evalMLP.dL_dw11,
        w12: mlpParams.w12 - lr * evalMLP.dL_dw12,
        b1: mlpParams.b1 - lr * evalMLP.dL_db1,
        w21: mlpParams.w21 - lr * evalMLP.dL_dw21,
        w22: mlpParams.w22 - lr * evalMLP.dL_dw22,
        b2: mlpParams.b2 - lr * evalMLP.dL_db2,
        v1: mlpParams.v1 - lr * evalMLP.dL_dv1,
        v2: mlpParams.v2 - lr * evalMLP.dL_dv2,
        bOut: mlpParams.bOut - lr * evalMLP.dL_dbOut,
      };

      const nextEval = evaluateMLP(nextMLP, 1);
      const nextStep = stepCount + 1;

      setMLPParams(nextMLP);
      setStepCount(nextStep);
      setLossHistory((prev) => [...prev, { step: nextStep, loss: nextEval.loss }]);
    }

    triggerForwardPulse();
  }, [architecture, singleParams, mlpParams, stepCount, triggerForwardPulse]);

  // Auto-play training loop
  useEffect(() => {
    if (isAutoPlaying) {
      autoPlayIntervalRef.current = setInterval(() => {
        handleStep();
      }, 160);
    } else {
      if (autoPlayIntervalRef.current) clearInterval(autoPlayIntervalRef.current);
    }
    return () => {
      if (autoPlayIntervalRef.current) clearInterval(autoPlayIntervalRef.current);
    };
  }, [isAutoPlaying, handleStep]);

  // Reset function
  const handleReset = () => {
    setIsAutoPlaying(false);
    if (architecture === 'single-neuron') {
      setSingleParams(initialSingleParams);
      const evalInit = evaluateSingleNeuron(initialSingleParams, 1);
      setLossHistory([{ step: 0, loss: evalInit.loss }]);
    } else {
      setMLPParams(initialMLPParams);
      const evalInit = evaluateMLP(initialMLPParams, 1);
      setLossHistory([{ step: 0, loss: evalInit.loss }]);
    }
    setStepCount(0);
    setBackwardProgress(1.0);
    setActiveBackwardNodeId(undefined);
    triggerForwardPulse();
  };

  // Randomize weights
  const handleRandomize = () => {
    const rand = () => Math.round((Math.random() * 4 - 2) * 100) / 100;
    if (architecture === 'single-neuron') {
      const updated = {
        ...singleParams,
        w1: rand(),
        w2: rand(),
        b: rand() / 2,
      };
      setSingleParams(updated);
      const evalR = evaluateSingleNeuron(updated, 1);
      setLossHistory([{ step: 0, loss: evalR.loss }]);
    } else {
      const updated: MLPParams = {
        ...mlpParams,
        w11: rand(),
        w12: rand(),
        b1: rand() / 2,
        w21: rand(),
        w22: rand(),
        b2: rand() / 2,
        v1: rand(),
        v2: rand(),
        bOut: rand() / 2,
      };
      setMLPParams(updated);
      const evalR = evaluateMLP(updated, 1);
      setLossHistory([{ step: 0, loss: evalR.loss }]);
    }
    setStepCount(0);
    triggerForwardPulse();
  };

  // Finite difference numerical check results
  const numericalCheckResults = useMemo(() => {
    return computeNumericalGradientSingleNeuron(singleParams);
  }, [singleParams]);

  // Clean up timers on unmount
  useEffect(() => {
    return () => {
      if (backwardAnimTimeoutRef.current) clearTimeout(backwardAnimTimeoutRef.current);
      if (forwardAnimTimeoutRef.current) clearTimeout(forwardAnimTimeoutRef.current);
      if (autoPlayIntervalRef.current) clearInterval(autoPlayIntervalRef.current);
    };
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* 3-zone Header */}
      <Header
        architecture={architecture}
        onArchitectureChange={(arch) => {
          setArchitecture(arch);
          setStepCount(0);
          setLossHistory([]);
        }}
        activation={activation}
        onActivationChange={setActivation}
        onOpenAutogradInfo={() => setIsAutogradInfoOpen(true)}
        onToggleNumericalCheck={() => setIsNumericalCheckOpen((prev) => !prev)}
        isNumericalCheckOpen={isNumericalCheckOpen}
      />

      {/* Main Container */}
      <main className="max-w-7xl w-full mx-auto px-4 sm:px-6 py-5 flex-1 flex flex-col gap-5">
        {/* Static Educational Panel: What is autograd? (Prompt requirement) */}
        <section className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs shadow-xs">
          <div className="flex items-start gap-3">
            <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 shrink-0 mt-0.5">
              <Info className="w-4 h-4" />
            </div>
            <div className="space-y-1">
              <div className="font-semibold text-slate-200 flex items-center gap-2">
                <span>What is Automatic Differentiation (Autograd)?</span>
                <span className="text-[11px] font-normal text-slate-400 font-mono hidden md:inline">
                  Reverse-mode computation graph
                </span>
              </div>
              <p className="text-slate-300 leading-relaxed max-w-4xl">
                Automatic differentiation builds a computation graph during the forward pass, where every node stores a value and its operation parents. Reverse-mode backpropagation then traverses this graph backwards from the scalar loss to every leaf node, multiplying local chain-rule derivatives to calculate exact parameter gradients in a single sweep.
              </p>
            </div>
          </div>

          <button
            onClick={() => setIsAutogradInfoOpen(true)}
            className="shrink-0 text-cyan-400 hover:text-cyan-300 font-medium flex items-center gap-1 hover:underline text-xs"
          >
            <span>Learn math</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </section>

        {/* Vanishing Gradient Callout (Prompt requirement) */}
        {architecture === 'single-neuron' && (
          <VanishingGradientAlert
            isVanishing={Boolean((currentEvaluation as any).isVanishing)}
            isReluDead={Boolean((currentEvaluation as any).isReluDead)}
            z={(currentEvaluation as any).z}
            da_dz={(currentEvaluation as any).da_dz}
            activation={activation}
            onSwitchToRelu={() => setActivation('relu')}
          />
        )}

        {/* High-Level Telemetry Cards */}
        <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3 flex flex-col">
            <span className="text-[11px] font-medium text-slate-400">Current Loss (L)</span>
            <div className="text-lg sm:text-xl font-bold font-mono text-cyan-300 tabular-nums">
              {currentEvaluation.loss.toFixed(5)}
            </div>
            <span className="text-[10px] text-slate-500 font-mono">
              L = (a - y)²
            </span>
          </div>

          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3 flex flex-col">
            <span className="text-[11px] font-medium text-slate-400">Neuron Output (a)</span>
            <div className="text-lg sm:text-xl font-bold font-mono text-slate-100 tabular-nums">
              {currentEvaluation.output.toFixed(4)}
            </div>
            <span className="text-[10px] text-slate-500 font-mono">
              Target y = {architecture === 'single-neuron' ? singleParams.yTarget.toFixed(2) : mlpParams.yTarget.toFixed(2)}
            </span>
          </div>

          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3 flex flex-col">
            <span className="text-[11px] font-medium text-slate-400">Prediction Error (|a - y|)</span>
            <div className="text-lg sm:text-xl font-bold font-mono text-amber-300 tabular-nums">
              {Math.abs(
                currentEvaluation.output -
                  (architecture === 'single-neuron'
                    ? singleParams.yTarget
                    : mlpParams.yTarget)
              ).toFixed(4)}
            </div>
            <span className="text-[10px] text-slate-500 font-mono">
              residual gap
            </span>
          </div>

          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3 flex flex-col">
            <span className="text-[11px] font-medium text-slate-400">Local Slope ∂a/∂z</span>
            <div
              className={`text-lg sm:text-xl font-bold font-mono tabular-nums ${
                (currentEvaluation as any).da_dz < 0.05
                  ? 'text-rose-400'
                  : 'text-emerald-300'
              }`}
            >
              {architecture === 'single-neuron'
                ? ((currentEvaluation as any).da_dz ?? 0).toFixed(4)
                : 'Layered'}
            </div>
            <span className="text-[10px] text-slate-500 font-mono">
              activation sensitivity
            </span>
          </div>
        </section>

        {/* Core Layout: 2-Zone Sandbox */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
          {/* Left Zone: Computation Graph (7 or 8 columns on large screens) */}
          <div className="lg:col-span-8 flex flex-col gap-5">
            {/* SVG Interactive Computation Graph */}
            <div className="h-[460px] sm:h-[520px]">
              <ComputationGraph
                nodes={currentEvaluation.nodes}
                edges={currentEvaluation.edges}
                isForwardAnimating={isForwardAnimating}
                isBackwardAnimating={isBackwardAnimating}
                activeBackwardNodeId={activeBackwardNodeId}
                selectedNodeId={selectedNode?.id}
                onSelectNode={(n) => setSelectedNode(n)}
              />
            </div>

            {/* Chain Rule Running Log / Table */}
            <ChainRuleTable
              steps={currentEvaluation.derivSteps}
              activeNodeId={activeBackwardNodeId || selectedNode?.id}
              onHoverNode={(nodeId) => {
                if (!nodeId) {
                  setSelectedNode(null);
                } else {
                  const found = currentEvaluation.nodes.find((n) => n.id === nodeId);
                  if (found) setSelectedNode(found);
                }
              }}
            />
          </div>

          {/* Right Zone: Control & Concept Deck (4 or 5 columns on large screens) */}
          <div className="lg:col-span-4 flex flex-col gap-5">
            {/* Controls Panel */}
            <ControlsPanel
              architecture={architecture}
              singleParams={singleParams}
              mlpParams={mlpParams}
              onUpdateSingleParam={handleUpdateSingleParam}
              onUpdateMLPParam={handleUpdateMLPParam}
              onComputeGradients={handleComputeGradients}
              onStep={handleStep}
              onAutoPlayToggle={() => setIsAutoPlaying((prev) => !prev)}
              isAutoPlaying={isAutoPlaying}
              onReset={handleReset}
              onRandomize={handleRandomize}
              isBackwardAnimating={isBackwardAnimating}
              stepCount={stepCount}
            />

            {/* Loss Chart */}
            <LossChart
              history={lossHistory}
              currentLoss={currentEvaluation.loss}
              onClearHistory={() =>
                setLossHistory([{ step: stepCount, loss: currentEvaluation.loss }])
              }
              learningRate={
                architecture === 'single-neuron'
                  ? singleParams.learningRate
                  : mlpParams.learningRate
              }
            />

            {/* Selected Node Inspector Details */}
            {selectedNode && (
              <div className="bg-slate-900/60 rounded-xl border border-cyan-500/40 p-3.5 text-xs flex flex-col gap-2 animate-in fade-in">
                <div className="flex items-center justify-between pb-1.5 border-b border-slate-800">
                  <div className="flex items-center gap-1.5 font-bold text-white">
                    <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Node: {selectedNode.name}</span>
                  </div>
                  <span className="font-mono text-[10px] text-slate-400">
                    ID: {selectedNode.id}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
                  <div className="bg-slate-950/60 p-2 rounded border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">Data Value:</span>
                    <span className="text-slate-100 font-semibold">
                      {selectedNode.data.toFixed(4)}
                    </span>
                  </div>
                  <div className="bg-slate-950/60 p-2 rounded border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">Gradient (∂L/∂node):</span>
                    <span
                      className={`font-semibold ${
                        selectedNode.grad > 0
                          ? 'text-emerald-400'
                          : selectedNode.grad < 0
                          ? 'text-rose-400'
                          : 'text-slate-400'
                      }`}
                    >
                      {selectedNode.grad >= 0 ? '+' : ''}
                      {selectedNode.grad.toFixed(4)}
                    </span>
                  </div>
                </div>

                <div className="text-[11px] text-slate-400 leading-normal">
                  {selectedNode.isTrainable
                    ? 'This is an optimizable parameter. During gradient descent, it updates according to w ← w - η · ∂L/∂w.'
                    : selectedNode.isInput
                    ? 'This is a fixed input feature. It passes data forward and receives error sensitivity backward.'
                    : selectedNode.isLoss
                    ? 'This is the objective loss function. Reverse-mode backprop starts here with seed gradient 1.0.'
                    : 'This is an intermediate computation node produced by an elementary operation.'}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-4 px-6 text-center text-xs text-slate-500 font-mono">
        Micrograd Autodiff Engine · Educational Reverse-Mode Differentiation Sandbox
      </footer>

      {/* Numerical Check Modal */}
      <NumericalCheckModal
        isOpen={isNumericalCheckOpen}
        onClose={() => setIsNumericalCheckOpen(false)}
        results={numericalCheckResults}
      />

      {/* Autograd Educational Guide Modal */}
      <AutogradGuideModal
        isOpen={isAutogradInfoOpen}
        onClose={() => setIsAutogradInfoOpen(false)}
      />
    </div>
  );
}
