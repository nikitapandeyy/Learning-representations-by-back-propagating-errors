import {
  ActivationType,
  DerivativeStep,
  GraphEdge,
  GraphNode,
  MLPParams,
  NumericalCheckResult,
  SingleNeuronParams,
} from './types';

export function activate(z: number, type: ActivationType): number {
  if (type === 'tanh') {
    return Math.tanh(z);
  }
  if (type === 'sigmoid') {
    // Avoid overflow/underflow in exp
    if (z >= 40) return 1;
    if (z <= -40) return 0;
    return 1 / (1 + Math.exp(-z));
  }
  // relu
  return Math.max(0, z);
}

export function activateDerivative(z: number, a: number, type: ActivationType): number {
  if (type === 'tanh') {
    return 1 - a * a;
  }
  if (type === 'sigmoid') {
    return a * (1 - a);
  }
  // relu: standard subgradient is 1 for z > 0, 0 otherwise
  return z > 0 ? 1 : 0;
}

export function activateFormula(type: ActivationType): { act: string; deriv: string } {
  if (type === 'tanh') {
    return { act: 'tanh(z)', deriv: '1 - a²' };
  }
  if (type === 'sigmoid') {
    return { act: 'σ(z)', deriv: 'a · (1 - a)' };
  }
  return { act: 'ReLU(z)', deriv: 'z > 0 ? 1 : 0' };
}

// Evaluate Single Neuron Computation Graph
export function evaluateSingleNeuron(
  params: SingleNeuronParams,
  backwardProgress: number = 1 // 0 (none), 0 to 1 (partial step), 1 (all gradients calculated)
) {
  const { x1, x2, w1, w2, b, yTarget, activation } = params;

  // Forward pass
  const w1x1 = w1 * x1;
  const w2x2 = w2 * x2;
  const sumWx = w1x1 + w2x2;
  const z = sumWx + b;
  const a = activate(z, activation);
  const diff = a - yTarget;
  const L = diff * diff;

  // Full backward calculations (analytic)
  const dL_dL = 1.0;
  const dL_ddiff = 2 * diff;
  const dL_da = dL_ddiff * 1.0; // 2 * (a - yTarget)
  const dL_dyTarget = dL_ddiff * -1.0;
  const da_dz = activateDerivative(z, a, activation);
  const dL_dz = dL_da * da_dz;
  const dL_dsumWx = dL_dz * 1.0;
  const dL_db = dL_dz * 1.0;
  const dL_dw1x1 = dL_dsumWx * 1.0;
  const dL_dw2x2 = dL_dsumWx * 1.0;
  const dL_dw1 = dL_dw1x1 * x1;
  const dL_dx1 = dL_dw1x1 * w1;
  const dL_dw2 = dL_dw2x2 * x2;
  const dL_dx2 = dL_dw2x2 * w2;

  // Ordered list of nodes during backpropagation (from output to inputs)
  // Step 0: L
  // Step 1: diff
  // Step 2: a & yTarget
  // Step 3: z
  // Step 4: sumWx & b
  // Step 5: w1x1 & w2x2
  // Step 6: w1, x1, w2, x2
  const maxStep = 6;
  const currentStep = Math.floor(backwardProgress * maxStep);

  const getGrad = (thresholdStep: number, actualGrad: number) => {
    if (backwardProgress === 0) return 0;
    return currentStep >= thresholdStep ? actualGrad : 0;
  };

  // Node definitions with 2D graph layout coordinates
  // Layers from left (layer 0) to right (layer 6)
  // X range: 80 to 920, Y range: 50 to 520
  const nodes: GraphNode[] = [
    // Layer 0: Inputs & Weights
    {
      id: 'w1',
      name: 'w₁',
      op: '',
      data: w1,
      grad: getGrad(6, dL_dw1),
      prevIds: [],
      layer: 0,
      x: 70,
      y: 70,
      isTrainable: true,
    },
    {
      id: 'x1',
      name: 'x₁',
      op: '',
      data: x1,
      grad: getGrad(6, dL_dx1),
      prevIds: [],
      layer: 0,
      x: 70,
      y: 160,
      isInput: true,
    },
    {
      id: 'w2',
      name: 'w₂',
      op: '',
      data: w2,
      grad: getGrad(6, dL_dw2),
      prevIds: [],
      layer: 0,
      x: 70,
      y: 260,
      isTrainable: true,
    },
    {
      id: 'x2',
      name: 'x₂',
      op: '',
      data: x2,
      grad: getGrad(6, dL_dx2),
      prevIds: [],
      layer: 0,
      x: 70,
      y: 350,
      isInput: true,
    },
    {
      id: 'b',
      name: 'b',
      op: '',
      data: b,
      grad: getGrad(4, dL_db),
      prevIds: [],
      layer: 0,
      x: 70,
      y: 450,
      isTrainable: true,
    },

    // Layer 1: Multiplications (w1*x1, w2*x2)
    {
      id: 'w1x1',
      name: 'w₁·x₁',
      op: '*',
      data: w1x1,
      grad: getGrad(5, dL_dw1x1),
      prevIds: ['w1', 'x1'],
      layer: 1,
      x: 230,
      y: 115,
    },
    {
      id: 'w2x2',
      name: 'w₂·x₂',
      op: '*',
      data: w2x2,
      grad: getGrad(5, dL_dw2x2),
      prevIds: ['w2', 'x2'],
      layer: 1,
      x: 230,
      y: 305,
    },

    // Layer 2: Sum of weighted inputs
    {
      id: 'sumWx',
      name: 'w₁x₁+w₂x₂',
      op: '+',
      data: sumWx,
      grad: getGrad(4, dL_dsumWx),
      prevIds: ['w1x1', 'w2x2'],
      layer: 2,
      x: 390,
      y: 210,
    },

    // Layer 3: Affine combination z = sumWx + b
    {
      id: 'z',
      name: 'z (affine)',
      op: '+',
      data: z,
      grad: getGrad(3, dL_dz),
      prevIds: ['sumWx', 'b'],
      layer: 3,
      x: 540,
      y: 240,
    },

    // Layer 4: Activation a = act(z)
    {
      id: 'a',
      name: `a = ${activation}`,
      op: activation,
      data: a,
      grad: getGrad(2, dL_da),
      prevIds: ['z'],
      layer: 4,
      x: 690,
      y: 240,
    },

    // Layer 4 extra: yTarget
    {
      id: 'yTarget',
      name: 'y_target',
      op: '',
      data: yTarget,
      grad: getGrad(2, dL_dyTarget),
      prevIds: [],
      layer: 4,
      x: 690,
      y: 390,
      isTarget: true,
    },

    // Layer 5: Difference (a - yTarget)
    {
      id: 'diff',
      name: 'a - y',
      op: '-',
      data: diff,
      grad: getGrad(1, dL_ddiff),
      prevIds: ['a', 'yTarget'],
      layer: 5,
      x: 840,
      y: 300,
    },

    // Layer 6: Squared Loss L = diff²
    {
      id: 'L',
      name: 'Loss (L)',
      op: 'sqr',
      data: L,
      grad: getGrad(0, dL_dL),
      prevIds: ['diff'],
      layer: 6,
      x: 980,
      y: 300,
      isLoss: true,
    },
  ];

  // Directed edges from source to target
  const edges: GraphEdge[] = [
    { id: 'e-w1-w1x1', from: 'w1', to: 'w1x1' },
    { id: 'e-x1-w1x1', from: 'x1', to: 'w1x1' },
    { id: 'e-w2-w2x2', from: 'w2', to: 'w2x2' },
    { id: 'e-x2-w2x2', from: 'x2', to: 'w2x2' },
    { id: 'e-w1x1-sumWx', from: 'w1x1', to: 'sumWx' },
    { id: 'e-w2x2-sumWx', from: 'w2x2', to: 'sumWx' },
    { id: 'e-sumWx-z', from: 'sumWx', to: 'z' },
    { id: 'e-b-z', from: 'b', to: 'z' },
    { id: 'e-z-a', from: 'z', to: 'a' },
    { id: 'e-a-diff', from: 'a', to: 'diff' },
    { id: 'e-yTarget-diff', from: 'yTarget', to: 'diff' },
    { id: 'e-diff-L', from: 'diff', to: 'L' },
  ];

  // Chain rule breakdown table
  const { deriv: actDerivFormula } = activateFormula(activation);
  const derivSteps: DerivativeStep[] = [
    {
      target: '∂L/∂L',
      nodeId: 'L',
      formula: '1.0 (seed gradient at terminal node)',
      substitution: '1.0000',
      localDerivative: 1.0,
      accumulatedGrad: dL_dL,
      description: 'Backpropagation begins at Loss with initial unit sensitivity.',
    },
    {
      target: '∂L/∂(diff)',
      nodeId: 'diff',
      formula: '2 · (a - y_target)',
      substitution: `2 · (${a.toFixed(4)} - ${yTarget.toFixed(4)}) = ${(2 * diff).toFixed(4)}`,
      localDerivative: 2 * diff,
      accumulatedGrad: dL_ddiff,
      description: 'Derivative of squared error (diff)² with respect to difference.',
    },
    {
      target: '∂L/∂a',
      nodeId: 'a',
      formula: '∂L/∂(diff) · ∂(diff)/∂a = ∂L/∂(diff) · 1',
      substitution: `${dL_ddiff.toFixed(4)} · 1.0000 = ${dL_da.toFixed(4)}`,
      localDerivative: 1.0,
      accumulatedGrad: dL_da,
      description: 'Chain rule from loss difference through to neuron output a.',
    },
    {
      target: '∂a/∂z (local)',
      nodeId: 'z',
      formula: actDerivFormula,
      substitution:
        activation === 'tanh'
          ? `1 - (${a.toFixed(4)})² = ${(1 - a * a).toFixed(4)}`
          : activation === 'sigmoid'
          ? `${a.toFixed(4)} · (1 - ${a.toFixed(4)}) = ${(a * (1 - a)).toFixed(4)}`
          : `${z.toFixed(4)} > 0 ? 1 : 0 = ${da_dz.toFixed(4)}`,
      localDerivative: da_dz,
      accumulatedGrad: da_dz,
      description: `Local activation derivative. If z is saturated (|z|>>0), this collapses toward zero.`,
    },
    {
      target: '∂L/∂z',
      nodeId: 'z',
      formula: '∂L/∂a · ∂a/∂z',
      substitution: `${dL_da.toFixed(4)} · ${da_dz.toFixed(4)} = ${dL_dz.toFixed(4)}`,
      localDerivative: da_dz,
      accumulatedGrad: dL_dz,
      description: 'Chain rule multiplying incoming gradient by local activation slope.',
    },
    {
      target: '∂L/∂b',
      nodeId: 'b',
      formula: '∂L/∂z · ∂z/∂b = ∂L/∂z · 1',
      substitution: `${dL_dz.toFixed(4)} · 1.0000 = ${dL_db.toFixed(4)}`,
      localDerivative: 1.0,
      accumulatedGrad: dL_db,
      description: 'Bias gradient directly matches pre-activation error gradient.',
    },
    {
      target: '∂L/∂w₁',
      nodeId: 'w1',
      formula: '∂L/∂z · ∂z/∂w₁ = ∂L/∂z · x₁',
      substitution: `${dL_dz.toFixed(4)} · ${x1.toFixed(4)} = ${dL_dw1.toFixed(4)}`,
      localDerivative: x1,
      accumulatedGrad: dL_dw1,
      description: 'Weight gradient proportional to the input feature x₁ activation.',
    },
    {
      target: '∂L/∂w₂',
      nodeId: 'w2',
      formula: '∂L/∂z · ∂z/∂w₂ = ∂L/∂z · x₂',
      substitution: `${dL_dz.toFixed(4)} · ${x2.toFixed(4)} = ${dL_dw2.toFixed(4)}`,
      localDerivative: x2,
      accumulatedGrad: dL_dw2,
      description: 'Weight gradient proportional to the input feature x₂ activation.',
    },
    {
      target: '∂L/∂x₁',
      nodeId: 'x1',
      formula: '∂L/∂z · ∂z/∂x₁ = ∂L/∂z · w₁',
      substitution: `${dL_dz.toFixed(4)} · ${w1.toFixed(4)} = ${dL_dx1.toFixed(4)}`,
      localDerivative: w1,
      accumulatedGrad: dL_dx1,
      description: 'Input sensitivity (gradient flowing backwards to inputs).',
    },
    {
      target: '∂L/∂x₂',
      nodeId: 'x2',
      formula: '∂L/∂z · ∂z/∂x₂ = ∂L/∂z · w₂',
      substitution: `${dL_dz.toFixed(4)} · ${w2.toFixed(4)} = ${dL_dx2.toFixed(4)}`,
      localDerivative: w2,
      accumulatedGrad: dL_dx2,
      description: 'Input sensitivity (gradient flowing backwards to inputs).',
    },
  ];

  // Vanishing gradient detection
  const isSaturated = Math.abs(z) > 2.5 && da_dz < 0.05 && activation === 'tanh';
  const isSigmoidSaturated = Math.abs(z) > 4.0 && da_dz < 0.03 && activation === 'sigmoid';
  const isReluDead = z <= 0 && activation === 'relu';

  return {
    nodes,
    edges,
    loss: L,
    output: a,
    z,
    da_dz,
    dL_dw1,
    dL_dw2,
    dL_db,
    derivSteps,
    isVanishing: isSaturated || isSigmoidSaturated,
    isReluDead,
    currentStep,
  };
}

// Numerical Gradient Checker via Finite Differences
export function computeNumericalGradientSingleNeuron(
  params: SingleNeuronParams,
  eps = 1e-4
): NumericalCheckResult[] {
  const forwardWith = (override: Partial<SingleNeuronParams>) => {
    const p = { ...params, ...override };
    const z = p.w1 * p.x1 + p.w2 * p.x2 + p.b;
    const a = activate(z, p.activation);
    const diff = a - p.yTarget;
    return diff * diff;
  };

  const analytic = evaluateSingleNeuron(params, 1);

  // w1
  const lossW1Plus = forwardWith({ w1: params.w1 + eps });
  const lossW1Minus = forwardWith({ w1: params.w1 - eps });
  const numW1 = (lossW1Plus - lossW1Minus) / (2 * eps);

  // w2
  const lossW2Plus = forwardWith({ w2: params.w2 + eps });
  const lossW2Minus = forwardWith({ w2: params.w2 - eps });
  const numW2 = (lossW2Plus - lossW2Minus) / (2 * eps);

  // b
  const lossBPlus = forwardWith({ b: params.b + eps });
  const lossBMinus = forwardWith({ b: params.b - eps });
  const numB = (lossBPlus - lossBMinus) / (2 * eps);

  const diffW1 = Math.abs(analytic.dL_dw1 - numW1);
  const diffW2 = Math.abs(analytic.dL_dw2 - numW2);
  const diffB = Math.abs(analytic.dL_db - numB);

  return [
    {
      paramName: 'w₁',
      analyticGrad: analytic.dL_dw1,
      numericGrad: numW1,
      diff: diffW1,
      matches: diffW1 < 1e-3,
    },
    {
      paramName: 'w₂',
      analyticGrad: analytic.dL_dw2,
      numericGrad: numW2,
      diff: diffW2,
      matches: diffW2 < 1e-3,
    },
    {
      paramName: 'b',
      analyticGrad: analytic.dL_db,
      numericGrad: numB,
      diff: diffB,
      matches: diffB < 1e-3,
    },
  ];
}

// Evaluate 2-Layer MLP (2 inputs -> 2 hidden neurons -> 1 output)
export function evaluateMLP(
  params: MLPParams,
  backwardProgress: number = 1
) {
  const { x1, x2, w11, w12, b1, w21, w22, b2, v1, v2, bOut, yTarget, activation } = params;

  // Layer 1 Forward
  const z1 = w11 * x1 + w12 * x2 + b1;
  const a1 = activate(z1, activation);

  const z2 = w21 * x1 + w22 * x2 + b2;
  const a2 = activate(z2, activation);

  // Layer 2 Forward
  const zOut = v1 * a1 + v2 * a2 + bOut;
  const aOut = activate(zOut, activation);

  // Loss
  const diff = aOut - yTarget;
  const L = diff * diff;

  // Backward Pass
  const dL_ddiff = 2 * diff;
  const dL_daOut = dL_ddiff;
  const daOut_dzOut = activateDerivative(zOut, aOut, activation);
  const dL_dzOut = dL_daOut * daOut_dzOut;

  // Output weights & bias
  const dL_dv1 = dL_dzOut * a1;
  const dL_dv2 = dL_dzOut * a2;
  const dL_dbOut = dL_dzOut * 1.0;

  // Gradients propagating into hidden layer activations
  const dL_da1 = dL_dzOut * v1;
  const dL_da2 = dL_dzOut * v2;

  // Gradients into hidden layer pre-activations
  const da1_dz1 = activateDerivative(z1, a1, activation);
  const dL_dz1 = dL_da1 * da1_dz1;

  const da2_dz2 = activateDerivative(z2, a2, activation);
  const dL_dz2 = dL_da2 * da2_dz2;

  // Hidden layer weights & bias
  const dL_dw11 = dL_dz1 * x1;
  const dL_dw12 = dL_dz1 * x2;
  const dL_db1 = dL_dz1 * 1.0;

  const dL_dw21 = dL_dz2 * x1;
  const dL_dw22 = dL_dz2 * x2;
  const dL_db2 = dL_dz2 * 1.0;

  // Input sensitivities
  const dL_dx1 = dL_dz1 * w11 + dL_dz2 * w21;
  const dL_dx2 = dL_dz1 * w12 + dL_dz2 * w22;

  const maxStep = 5;
  const currentStep = Math.floor(backwardProgress * maxStep);
  const getGrad = (threshold: number, actual: number) =>
    backwardProgress === 0 ? 0 : currentStep >= threshold ? actual : 0;

  // Layout for MLP nodes:
  // Layer 0: x1, x2 (X: 80)
  // Layer 1: H1 (z1, a1) & H2 (z2, a2) (X: 300, 480)
  // Layer 2: Output neuron (zOut, aOut) (X: 700, 840)
  // Layer 3: Loss (X: 1000)
  const nodes: GraphNode[] = [
    // Inputs
    { id: 'x1', name: 'x₁', op: '', data: x1, grad: getGrad(5, dL_dx1), prevIds: [], layer: 0, x: 70, y: 150, isInput: true },
    { id: 'x2', name: 'x₂', op: '', data: x2, grad: getGrad(5, dL_dx2), prevIds: [], layer: 0, x: 70, y: 350, isInput: true },

    // Hidden Neuron 1
    { id: 'z1', name: 'z₁ (hidden 1)', op: 'Σ', data: z1, grad: getGrad(4, dL_dz1), prevIds: ['x1', 'x2'], layer: 1, x: 280, y: 150 },
    { id: 'a1', name: 'a₁', op: activation, data: a1, grad: getGrad(3, dL_da1), prevIds: ['z1'], layer: 2, x: 440, y: 150 },

    // Hidden Neuron 2
    { id: 'z2', name: 'z₂ (hidden 2)', op: 'Σ', data: z2, grad: getGrad(4, dL_dz2), prevIds: ['x1', 'x2'], layer: 1, x: 280, y: 350 },
    { id: 'a2', name: 'a₂', op: activation, data: a2, grad: getGrad(3, dL_da2), prevIds: ['z2'], layer: 2, x: 440, y: 350 },

    // Output Neuron
    { id: 'zOut', name: 'z_out', op: 'Σ', data: zOut, grad: getGrad(2, dL_dzOut), prevIds: ['a1', 'a2'], layer: 3, x: 640, y: 250 },
    { id: 'aOut', name: 'a_out', op: activation, data: aOut, grad: getGrad(1, dL_daOut), prevIds: ['zOut'], layer: 4, x: 800, y: 250 },
    { id: 'yTarget', name: 'y_target', op: '', data: yTarget, grad: getGrad(1, -dL_ddiff), prevIds: [], layer: 4, x: 800, y: 400, isTarget: true },

    // Loss
    { id: 'L', name: 'Loss (L)', op: 'sqr', data: L, grad: getGrad(0, 1.0), prevIds: ['aOut', 'yTarget'], layer: 5, x: 970, y: 320, isLoss: true },
  ];

  const edges: GraphEdge[] = [
    { id: 'e-x1-z1', from: 'x1', to: 'z1', label: `w₁₁=${w11.toFixed(2)}` },
    { id: 'e-x2-z1', from: 'x2', to: 'z1', label: `w₁₂=${w12.toFixed(2)}` },
    { id: 'e-z1-a1', from: 'z1', to: 'a1' },
    { id: 'e-x1-z2', from: 'x1', to: 'z2', label: `w₂₁=${w21.toFixed(2)}` },
    { id: 'e-x2-z2', from: 'x2', to: 'z2', label: `w₂₂=${w22.toFixed(2)}` },
    { id: 'e-z2-a2', from: 'z2', to: 'a2' },
    { id: 'e-a1-zOut', from: 'a1', to: 'zOut', label: `v₁=${v1.toFixed(2)}` },
    { id: 'e-a2-zOut', from: 'a2', to: 'zOut', label: `v₂=${v2.toFixed(2)}` },
    { id: 'e-zOut-aOut', from: 'zOut', to: 'aOut' },
    { id: 'e-aOut-L', from: 'aOut', to: 'L' },
    { id: 'e-yTarget-L', from: 'yTarget', to: 'L' },
  ];

  const derivSteps: DerivativeStep[] = [
    {
      target: '∂L/∂(a_out)',
      nodeId: 'aOut',
      formula: '2 · (a_out - y_target)',
      substitution: `2 · (${aOut.toFixed(4)} - ${yTarget.toFixed(4)}) = ${dL_daOut.toFixed(4)}`,
      localDerivative: 2 * diff,
      accumulatedGrad: dL_daOut,
      description: 'Output error gradient flowing backwards from squared loss.',
    },
    {
      target: '∂L/∂(z_out)',
      nodeId: 'zOut',
      formula: '∂L/∂(a_out) · σ\'(z_out)',
      substitution: `${dL_daOut.toFixed(4)} · ${daOut_dzOut.toFixed(4)} = ${dL_dzOut.toFixed(4)}`,
      localDerivative: daOut_dzOut,
      accumulatedGrad: dL_dzOut,
      description: 'Chain rule at final layer pre-activation.',
    },
    {
      target: '∂L/∂v₁ (layer 2 weight)',
      nodeId: 'zOut',
      formula: '∂L/∂(z_out) · a₁',
      substitution: `${dL_dzOut.toFixed(4)} · ${a1.toFixed(4)} = ${dL_dv1.toFixed(4)}`,
      localDerivative: a1,
      accumulatedGrad: dL_dv1,
      description: 'Output layer weight connecting hidden neuron 1 to output.',
    },
    {
      target: '∂L/∂a₁ (hidden backprop)',
      nodeId: 'a1',
      formula: '∂L/∂(z_out) · v₁',
      substitution: `${dL_dzOut.toFixed(4)} · ${v1.toFixed(4)} = ${dL_da1.toFixed(4)}`,
      localDerivative: v1,
      accumulatedGrad: dL_da1,
      description: 'Error gradient distributed backward across Layer 2 weights into hidden activations.',
    },
    {
      target: '∂L/∂z₁ (hidden pre-act)',
      nodeId: 'z1',
      formula: '∂L/∂a₁ · σ\'(z₁)',
      substitution: `${dL_da1.toFixed(4)} · ${da1_dz1.toFixed(4)} = ${dL_dz1.toFixed(4)}`,
      localDerivative: da1_dz1,
      accumulatedGrad: dL_dz1,
      description: 'Chain rule passes through hidden layer non-linearity.',
    },
    {
      target: '∂L/∂w₁₁ (layer 1 weight)',
      nodeId: 'z1',
      formula: '∂L/∂z₁ · x₁',
      substitution: `${dL_dz1.toFixed(4)} · ${x1.toFixed(4)} = ${dL_dw11.toFixed(4)}`,
      localDerivative: x1,
      accumulatedGrad: dL_dw11,
      description: 'Hidden weight w₁₁ gradient formed from compound chain rule across 2 full layers.',
    },
  ];

  return {
    nodes,
    edges,
    loss: L,
    output: aOut,
    dL_dw11,
    dL_dw12,
    dL_db1,
    dL_dw21,
    dL_dw22,
    dL_db2,
    dL_dv1,
    dL_dv2,
    dL_dbOut,
    derivSteps,
    currentStep,
  };
}
