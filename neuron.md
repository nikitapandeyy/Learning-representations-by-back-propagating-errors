z = w1*x1 + w2*x2 + ... + b     (weighted sum + bias)
a = activation(z)               (squash it)
Inputs (x): the numbers coming in.
Weights (w): how much the neuron cares about each input. These are what training changes.
Bias (b): a shift, so the neuron can fire even when all inputs are 0.
Activation: a non-linear function such as tanh, sigmoid, or relu.