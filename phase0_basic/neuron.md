z = w1*x1 + w2*x2 + ... + b     (weighted sum + bias)
a = activation(z)               (squash it)
Inputs (x): the numbers coming in.
Weights (w): how much the neuron cares about each input. These are what training changes.
Bias (b): a shift, so the neuron can fire even when all inputs are 0.
Activation: a non-linear function such as tanh, sigmoid, or relu.

Numeric example: x = [1, 2], w = [0.5, -1], b = 0.5

z = 0.5*1 + (-1)*2 + 0.5 = -1.0
a = tanh(-1.0) = -0.7616

Your Step 1 model yc = w*x was already a neuron with one input, no bias, and no activation.

Why the activation matters

Without it, stacking layers gives you nothing. Two linear layers collapse into one linear layer, because a linear function of a linear function is still linear. A single neuron can only draw a straight line to separate data, and XOR can't be separated by a line. Minsky and Papert pointed this out in 1969, which is the problem the 1986 paper solved. Non-linear activations plus hidden layers plus backprop can learn XOR.

The derivative you'll need

For tanh: d tanh(z)/dz = 1 - tanh(z)². Here that's 1 - 0.7616² ≈ 0.42. This is the "local derivative" that your future Value.tanh() will use in its _backward.