export type ActivationType = 'tanh' | 'sigmoid' | 'relu';
export type ArchitectureType = 'single-neuron' | 'mlp-2-2-1';

export interface GraphNode {
  id: string;
  name: string;
  op: string; // '', '*', '+', 'tanh', 'sigmoid', 'relu', 'sub', 'sqr'
  data: number;
  grad: number;
  prevIds: string[];
  layer: number; // For layout positioning
  x: number;
  y: number;
  isTrainable?: boolean;
  isInput?: boolean;
  isTarget?: boolean;
  isLoss?: boolean;
}

export interface GraphEdge {
  id: string;
  from: string;
  to: string;
  label?: string;
}

export interface DerivativeStep {
  target: string; // e.g. "dL/da"
  nodeId: string;
  formula: string; // e.g. "2 * (a - y)"
  substitution: string; // e.g. "2 * (0.7616 - 1.0000)"
  localDerivative: number;
  accumulatedGrad: number;
  description: string;
}

export interface NumericalCheckResult {
  paramName: string;
  analyticGrad: number;
  numericGrad: number;
  diff: number;
  matches: boolean;
}

export interface SingleNeuronParams {
  x1: number;
  x2: number;
  w1: number;
  w2: number;
  b: number;
  yTarget: number;
  activation: ActivationType;
  learningRate: number;
}

export interface MLPParams {
  x1: number;
  x2: number;
  // Layer 1 (Neuron 1)
  w11: number;
  w12: number;
  b1: number;
  // Layer 1 (Neuron 2)
  w21: number;
  w22: number;
  b2: number;
  // Layer 2 (Output Neuron)
  v1: number;
  v2: number;
  bOut: number;
  yTarget: number;
  activation: ActivationType;
  learningRate: number;
}
