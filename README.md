# Backpropagation & Autograd Visualizer

> **Learn backpropagation by seeing every calculation happen step by step.**

This project is a learning-focused implementation of **backpropagation, automatic differentiation, computational graphs, and neural-network training**.

The project is inspired by the research paper:

> **Learning representations by back-propagating errors**
> David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
> *Nature, 1986*

The goal is not to hide the mathematics behind a deep-learning framework.

Instead, this project builds the concepts **from the ground up**:

```text
Derivatives
     ↓
Chain Rule
     ↓
Forward Propagation
     ↓
Backpropagation
     ↓
Gradient Descent
     ↓
Manual Backpropagation
     ↓
Computational Graph
     ↓
Automatic Differentiation
     ↓
Custom Autograd Engine
     ↓
Neural Network
     ↓
Interactive Visualization
```

---

# Table of Contents

* [About the Research Paper](#about-the-research-paper)
* [Why Was This Research Important?](#why-was-this-research-important)
* [The Problem: Learning Hidden Representations](#the-problem-learning-hidden-representations)
* [What Is a Neural Network?](#what-is-a-neural-network)
* [A Tiny Neural Network](#a-tiny-neural-network)
* [Forward Propagation](#forward-propagation)
* [Activation Function](#activation-function)
* [Calculating the Error](#calculating-the-error)
* [Why Do We Need Backpropagation?](#why-do-we-need-backpropagation)
* [The Chain Rule](#the-chain-rule)
* [Backpropagation Mathematics](#backpropagation-mathematics)
* [Backward Pass](#backward-pass)
* [Gradient Descent](#gradient-descent)
* [How Hidden Units Learn](#how-hidden-units-learn)
* [Example Calculation](#example-calculation)
* [From Mathematics to Code](#from-mathematics-to-code)
* [Manual Backpropagation](#manual-backpropagation)
* [Computational Graph](#computational-graph)
* [Automatic Differentiation](#automatic-differentiation)
* [Our Custom Autograd Engine](#our-custom-autograd-engine)
* [Matrix Version](#matrix-version)
* [3 Inputs → 2 Hidden Neurons → 1 Output](#3-inputs--2-hidden-neurons--1-output)
* [Project Architecture](#project-architecture)
* [Visualization](#visualization)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Running the Project](#running-the-project)
* [Learning Roadmap](#learning-roadmap)
* [What This Project Teaches](#what-this-project-teaches)
* [Limitations](#limitations)
* [References](#references)

---

# About the Research Paper

The foundation of this project is the 1986 paper:

**"Learning representations by back-propagating errors"**

by:

* David E. Rumelhart
* Geoffrey E. Hinton
* Ronald J. Williams

The paper describes a method for training multi-layer neural networks using **backpropagation**.

The central idea is simple:

> A neural network makes a prediction, measures how wrong that prediction is, and then propagates information about the error backward through the network so that the weights can be adjusted.

The paper shows how this allows neural networks to learn useful internal representations using hidden units.

The paper also explains that the derivatives needed for learning can be calculated efficiently using a combination of a **forward pass** and a **backward pass**.

---

# Why Was This Research Important?

A neural network can contain multiple layers:

```text
Input Layer
     ↓
Hidden Layer
     ↓
Hidden Layer
     ↓
Output Layer
```

The difficulty is that the hidden layers do not directly receive the desired output.

For example:

```text
Input → Hidden → Output
                  ↓
              Target
```

We know whether the final output is correct.

But how do we know how much each hidden neuron contributed to the error?

And how much should each weight change?

Backpropagation solves this problem using the **chain rule of calculus**.

Instead of trying to calculate everything at once, we break the derivative into smaller pieces.

---

# The Problem: Learning Hidden Representations

One of the important ideas in the paper is that hidden units can learn useful internal representations.

Consider:

```text
Input
  ↓
Hidden Units
  ↓
Output
```

The hidden units are not explicitly told:

```text
"Neuron 1 should learn this feature."
"Neuron 2 should learn that feature."
```

Instead, the network receives an error signal from the output.

Backpropagation propagates this information backward.

Over repeated training steps, the weights change so that the hidden units develop representations that are useful for the task.

This is one of the important ideas behind multi-layer neural networks.

---

# What Is a Neural Network?

At its simplest, a neuron performs three operations:

```text
Inputs
  ↓
Weighted Sum
  ↓
Activation Function
  ↓
Output
```

Suppose we have:

```text
x₁, x₂, x₃
```

and weights:

```text
w₁, w₂, w₃
```

The neuron calculates:

```text
z = w₁x₁ + w₂x₂ + w₃x₃
```

Then an activation function is applied:

```text
y = activation(z)
```

For a sigmoid neuron:

```text
y = sigmoid(z)
```

---

# A Tiny Neural Network

Before building a large neural network, we start with the smallest useful example.

Consider:

```text
x
│
│ w₁
▼
Hidden Neuron
│
│ w₂
▼
Output Neuron
│
▼
y
```

There is:

* 1 input
* 1 hidden neuron
* 1 output neuron

We also have a target:

```text
d
```

Our goal is to make:

```text
y ≈ d
```

The complete computation is:

```text
x
 ↓
zₕ
 ↓
h
 ↓
zᵧ
 ↓
y
 ↓
E
```

Where:

* `x` = input
* `w₁` = input → hidden weight
* `zₕ` = hidden pre-activation
* `h` = hidden activation
* `w₂` = hidden → output weight
* `zᵧ` = output pre-activation
* `y` = prediction
* `d` = desired output / target
* `E` = error

---

# Forward Propagation

The first stage is the **forward pass**.

Information flows from the input toward the output.

## Step 1: Hidden Pre-Activation

The hidden neuron receives:

```text
zₕ = w₁ × x
```

## Step 2: Hidden Activation

Apply sigmoid:

```text
h = sigmoid(zₕ)
```

## Step 3: Output Pre-Activation

The hidden neuron sends its activation to the output:

```text
zᵧ = w₂ × h
```

## Step 4: Output Activation

The final prediction is:

```text
y = sigmoid(zᵧ)
```

Therefore:

```text
x
 ↓
zₕ = w₁x
 ↓
h = sigmoid(zₕ)
 ↓
zᵧ = w₂h
 ↓
y = sigmoid(zᵧ)
```

This entire process is called **forward propagation**.

---

# Activation Function

We use the sigmoid activation function:

```text
sigmoid(z) = 1 / (1 + e⁻ᶻ)
```

Its derivative has a particularly useful form:

```text
sigmoid'(z) = sigmoid(z)(1 - sigmoid(z))
```

If:

```text
h = sigmoid(z)
```

then:

```text
dh/dz = h(1 - h)
```

Similarly, if:

```text
y = sigmoid(zᵧ)
```

then:

```text
dy/dzᵧ = y(1 - y)
```

This derivative is extremely important during backpropagation.

---

# Calculating the Error

After the forward pass, we compare the prediction with the target.

We use the squared error:

```text
E = ½(d - y)²
```

Where:

```text
d = desired output
y = predicted output
```

If the prediction is close to the target:

```text
E → small
```

If the prediction is far from the target:

```text
E → large
```

Our goal is:

```text
minimize E
```

---

# Why Do We Need Backpropagation?

Suppose our network produced:

```text
y = 0.62
```

but the target is:

```text
d = 1.0
```

The prediction is wrong.

We want to change:

```text
w₁
w₂
```

so that the next prediction becomes closer to:

```text
1.0
```

But we need to answer:

```text
How much should w₁ change?
How much should w₂ change?
```

In mathematical form, we need:

```text
∂E/∂w₁
```

and:

```text
∂E/∂w₂
```

These derivatives tell us how the error changes when each weight changes.

---

# The Chain Rule

This is the heart of backpropagation.

Suppose:

```text
a → b → c
```

and:

```text
c = f(b)
b = g(a)
```

Then:

```text
dc/da = dc/db × db/da
```

This is the **chain rule**.

Neural networks contain many connected operations.

For example:

```text
w₁
 ↓
zₕ
 ↓
h
 ↓
zᵧ
 ↓
y
 ↓
E
```

To calculate:

```text
∂E/∂w₁
```

we follow this path backward:

```text
∂E/∂y
×
∂y/∂zᵧ
×
∂zᵧ/∂h
×
∂h/∂zₕ
×
∂zₕ/∂w₁
```

Therefore:

```text
∂E/∂w₁
=
∂E/∂y
×
∂y/∂zᵧ
×
∂zᵧ/∂h
×
∂h/∂zₕ
×
∂zₕ/∂w₁
```

This is backpropagation.

---

# Backpropagation Mathematics

Let's derive everything step by step.

Our forward equations are:

```text
zₕ = w₁x
```

```text
h = sigmoid(zₕ)
```

```text
zᵧ = w₂h
```

```text
y = sigmoid(zᵧ)
```

```text
E = ½(d - y)²
```

---

## Step 1: Derivative of Error with Respect to Output

Start with:

```text
E = ½(d - y)²
```

Differentiate with respect to `y`:

```text
∂E/∂y = y - d
```

This tells us how the error changes when the prediction changes.

---

## Step 2: Derivative of Output Activation

We have:

```text
y = sigmoid(zᵧ)
```

Therefore:

```text
∂y/∂zᵧ = y(1 - y)
```

---

## Step 3: Derivative of Output Pre-Activation

We have:

```text
zᵧ = w₂h
```

Therefore:

```text
∂zᵧ/∂w₂ = h
```

Using the chain rule:

```text
∂E/∂w₂
=
∂E/∂y
×
∂y/∂zᵧ
×
∂zᵧ/∂w₂
```

Therefore:

```text
∂E/∂w₂
=
(y - d)
×
y(1 - y)
×
h
```

---

# Gradient for the First Weight

Now we move further backward.

We want:

```text
∂E/∂w₁
```

The dependency path is:

```text
w₁
 ↓
zₕ
 ↓
h
 ↓
zᵧ
 ↓
y
 ↓
E
```

Therefore:

```text
∂E/∂w₁
=
∂E/∂y
×
∂y/∂zᵧ
×
∂zᵧ/∂h
×
∂h/∂zₕ
×
∂zₕ/∂w₁
```

We already know:

```text
∂E/∂y = y - d
```

```text
∂y/∂zᵧ = y(1 - y)
```

From:

```text
zᵧ = w₂h
```

we get:

```text
∂zᵧ/∂h = w₂
```

From:

```text
h = sigmoid(zₕ)
```

we get:

```text
∂h/∂zₕ = h(1 - h)
```

And from:

```text
zₕ = w₁x
```

we get:

```text
∂zₕ/∂w₁ = x
```

Therefore:

```text
∂E/∂w₁
=
(y - d)
×
y(1 - y)
×
w₂
×
h(1 - h)
×
x
```

This is the gradient for the first weight.

---

# Backward Pass

The forward pass calculates:

```text
x → h → y → E
```

The backward pass calculates derivatives:

```text
E → y → h → w₂ → w₁
```

So the complete process is:

```text
FORWARD

x
 ↓
zₕ
 ↓
h
 ↓
zᵧ
 ↓
y
 ↓
E


BACKWARD

E
 ↓
∂E/∂y
 ↓
∂E/∂zᵧ
 ↓
∂E/∂h
 ↓
∂E/∂w₂
 ↓
∂E/∂zₕ
 ↓
∂E/∂w₁
```

This is why it is called **backpropagation**.

The error information propagates backward through the network.

---

# Gradient Descent

Once we calculate the gradients, we need to update the weights.

The basic gradient descent rule is:

```text
new weight = old weight - learning rate × gradient
```

For `w₁`:

```text
w₁ ← w₁ - η(∂E/∂w₁)
```

For `w₂`:

```text
w₂ ← w₂ - η(∂E/∂w₂)
```

Where:

```text
η = learning rate
```

The learning rate controls how large each update is.

---

# Understanding the Gradient

Suppose:

```text
∂E/∂w = -0.05
```

and:

```text
η = 0.1
```

Then:

```text
w_new = w_old - 0.1(-0.05)
```

So:

```text
w_new = w_old + 0.005
```

The weight increases.

If the gradient is positive:

```text
∂E/∂w = +0.05
```

then:

```text
w_new = w_old - 0.005
```

The weight decreases.

The gradient tells us the direction in which the error increases.

Gradient descent moves in the opposite direction.

---

# Example Calculation

Let's use:

```text
x = 1
w₁ = 0.5
w₂ = 0.8
d = 1
```

and:

```text
η = 0.1
```

---

## Forward Pass

First:

```text
zₕ = w₁x
```

Therefore:

```text
zₕ = 0.5 × 1
```

```text
zₕ = 0.5
```

Now:

```text
h = sigmoid(0.5)
```

Approximately:

```text
h ≈ 0.6225
```

Next:

```text
zᵧ = w₂h
```

```text
zᵧ = 0.8 × 0.6225
```

```text
zᵧ ≈ 0.498
```

Then:

```text
y = sigmoid(0.498)
```

Approximately:

```text
y ≈ 0.622
```

The target is:

```text
d = 1
```

So:

```text
E = ½(1 - 0.622)²
```

Approximately:

```text
E ≈ 0.0714
```

---

# Calculate the Gradient for w₂

We have:

```text
∂E/∂w₂
=
(y - d)
×
y(1 - y)
×
h
```

Using:

```text
y ≈ 0.622
d = 1
h ≈ 0.6225
```

we get approximately:

```text
∂E/∂w₂ ≈ -0.0552
```

---

# Calculate the Gradient for w₁

We have:

```text
∂E/∂w₁
=
(y - d)
×
y(1 - y)
×
w₂
×
h(1 - h)
×
x
```

Substituting the values gives approximately:

```text
∂E/∂w₁ ≈ -0.0167
```

---

# Update the Weights

Using:

```text
w_new = w_old - η × gradient
```

For `w₂`:

```text
w₂_new
=
0.8 - 0.1(-0.0552)
```

Therefore:

```text
w₂_new ≈ 0.80552
```

For `w₁`:

```text
w₁_new
=
0.5 - 0.1(-0.0167)
```

Therefore:

```text
w₁_new ≈ 0.50167
```

The network has now completed one training step.

---

# How Hidden Units Learn

This is one of the most important ideas to understand.

The hidden neuron does not directly know:

```text
"What should my output be?"
```

Instead, it receives information through the chain rule.

The error starts at the output:

```text
E
```

and propagates backward:

```text
E
 ↓
Output
 ↓
Hidden
 ↓
Weights
```

This allows the hidden layer's weights to change even though there is no direct target for the hidden neuron.

After many training iterations:

```text
forward pass
      ↓
calculate error
      ↓
backward pass
      ↓
calculate gradients
      ↓
update weights
      ↓
repeat
```

The hidden representation gradually changes to help reduce the final task error.

---

# From Mathematics to Code

Now we translate the equations directly into Python.

A minimal forward pass looks like:

```python
import math


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


x = 1.0
w1 = 0.5
w2 = 0.8

zh = w1 * x
h = sigmoid(zh)

zy = w2 * h
y = sigmoid(zy)

print("Hidden:", h)
print("Output:", y)
```

The important thing is that every line corresponds directly to a mathematical equation.

---

# Manual Backpropagation

Now we implement the backward pass ourselves.

```python
d = 1.0

# Error derivative
dE_dy = y - d

# Output activation derivative
dy_dzy = y * (1 - y)

# Gradient for w2
dzy_dw2 = h

dw2 = dE_dy * dy_dzy * dzy_dw2

# Continue backward through hidden neuron
dzy_dh = w2
dh_dzh = h * (1 - h)
dzh_dw1 = x

dw1 = (
    dE_dy
    * dy_dzy
    * dzy_dh
    * dh_dzh
    * dzh_dw1
)

print("dE/dw2:", dw2)
print("dE/dw1:", dw1)
```

This is intentionally written in a verbose way.

We are not trying to write the shortest code.

We are trying to make the computation visible.

---

# Why Build It Manually?

Modern frameworks can calculate gradients automatically.

For example:

```python
loss.backward()
```

can calculate gradients for us.

But if we only use:

```python
loss.backward()
```

we may not understand what is actually happening.

This project therefore follows this progression:

```text
1. Calculate derivatives manually
             ↓
2. Implement backpropagation manually
             ↓
3. Represent operations as a graph
             ↓
4. Store local derivatives
             ↓
5. Propagate gradients automatically
             ↓
6. Build a custom autograd engine
```

The goal is to understand what automatic differentiation is doing underneath the framework.

---

# Computational Graph

A neural network can be represented as a computational graph.

For our example:

```text
x
│
▼
zₕ = w₁ × x
│
▼
h = sigmoid(zₕ)
│
▼
zᵧ = w₂ × h
│
▼
y = sigmoid(zᵧ)
│
▼
E = ½(d - y)²
```

Each operation becomes a node in the graph.

For example:

```text
w₁ ──┐
     × ──> zₕ
x  ──┘
```

and:

```text
zₕ
 │
 ▼
sigmoid
 │
 ▼
h
```

The computational graph gives us a structure that can be traversed backward.

---

# Automatic Differentiation

Automatic differentiation, or **autodiff**, is the process of automatically calculating derivatives by decomposing a computation into elementary operations.

For example:

```text
a = x × y
```

The local derivatives are:

```text
∂a/∂x = y
```

and:

```text
∂a/∂y = x
```

If another operation depends on `a`, the gradient can be propagated backward using the chain rule.

This means we don't need to manually derive an enormous equation for every network.

Instead, we can:

```text
Build graph
    ↓
Store operations
    ↓
Store local derivatives
    ↓
Run forward
    ↓
Run backward
    ↓
Accumulate gradients
```

---

# Our Custom Autograd Engine

The next stage of this project is a tiny autograd engine.

The basic object can be thought of as:

```python
Value
```

Each `Value` stores information such as:

```text
value
gradient
operation
parents
backward function
```

Conceptually:

```python
class Value:

    def __init__(self, data):
        self.data = data
        self.grad = 0.0
```

For an operation such as addition:

```python
c = a + b
```

we want the graph to remember:

```text
a ──┐
    + ──> c
b ──┘
```

During backward propagation:

```text
dc/da = 1
dc/db = 1
```

For multiplication:

```python
c = a * b
```

we have:

```text
dc/da = b
```

and:

```text
dc/db = a
```

This is the foundation of the autograd engine.

---

# Local Gradients

The key idea is:

> Every operation knows how its output changes with respect to its inputs.

For multiplication:

```text
c = a × b
```

we store:

```text
dc/da = b
dc/db = a
```

For addition:

```text
c = a + b
```

we store:

```text
dc/da = 1
dc/db = 1
```

For sigmoid:

```text
c = sigmoid(a)
```

we store:

```text
dc/da = c(1 - c)
```

Then the autograd engine combines these local derivatives using the chain rule.

---

# Backward Propagation in the Autograd Engine

Suppose:

```text
L = f(g(x))
```

The graph looks like:

```text
x
 ↓
g
 ↓
f
 ↓
L
```

The backward pass starts with:

```text
dL/dL = 1
```

Then:

```text
dL/dg
```

and finally:

```text
dL/dx
```

using:

```text
dL/dx
=
dL/dg
×
dg/dx
```

This is exactly the chain rule.

The autograd engine simply performs this process automatically over the computational graph.

---

# Matrix Version

After understanding the one-neuron example, we move to a real multi-neuron network.

Consider:

```text
3 inputs
   ↓
2 hidden neurons
   ↓
1 output neuron
```

Our input vector is:

```text
x =
[x₁
 x₂
 x₃]
```

Its shape is:

```text
3 × 1
```

The hidden-layer weight matrix is:

```text
W₁ =
[w₁₁  w₁₂  w₁₃
 w₂₁  w₂₂  w₂₃]
```

Its shape is:

```text
2 × 3
```

Therefore:

```text
W₁ × x
```

has shape:

```text
(2 × 3) × (3 × 1)
```

which produces:

```text
2 × 1
```

So:

```text
zₕ = W₁x
```

produces two hidden pre-activations.

---

# 3 Inputs → 2 Hidden Neurons → 1 Output

The network looks like:

```text
          ┌───────────────┐
x₁ ──────►│               │
x₂ ──────►│ Hidden Layer  │──────► Output
x₃ ──────►│               │
          └───────────────┘
```

More explicitly:

```text
             h₁
           ↗
x₁ ───────┤
           ↘
             h₂ ───────► y
           ↗
x₂ ───────┤
           ↘
x₃ ───────┘
```

The forward pass becomes:

```text
zₕ = W₁x
```

```text
h = sigmoid(zₕ)
```

Then:

```text
zᵧ = W₂h
```

and:

```text
y = sigmoid(zᵧ)
```

Finally:

```text
E = ½(d - y)²
```

---

# Matrix Dimensions

Understanding dimensions is extremely important.

For:

```text
3 inputs → 2 hidden → 1 output
```

we use:

```text
x  = 3 × 1
W₁ = 2 × 3
b₁ = 2 × 1

h  = 2 × 1

W₂ = 1 × 2
b₂ = 1 × 1

y  = 1 × 1
```

The forward pass is:

```text
z₁ = W₁x + b₁
```

giving:

```text
2 × 1
```

Then:

```text
h = sigmoid(z₁)
```

Then:

```text
z₂ = W₂h + b₂
```

giving:

```text
1 × 1
```

Finally:

```text
y = sigmoid(z₂)
```

---

# Why Matrix Multiplication?

Instead of writing every neuron separately:

```text
z₁ = w₁₁x₁ + w₁₂x₂ + w₁₃x₃

z₂ = w₂₁x₁ + w₂₂x₂ + w₂₃x₃
```

we can write:

```text
z = Wx
```

This makes neural networks much easier to implement efficiently.

It also makes the dimensions explicit.

---

# Visualization

One of the main goals of this project is to make backpropagation **visible**.

The visualizer should show:

```text
Input
  ↓
Weighted Sum
  ↓
Activation
  ↓
Output
  ↓
Loss
  ↓
Backward Pass
  ↓
Gradients
  ↓
Weight Update
```

Instead of simply showing:

```text
loss = 0.04
```

the visualizer should explain:

```text
Why is the loss 0.04?

Which operation produced it?

Which gradient caused the weight to change?

How did the error travel backward?
```

---

# Visualizer Concept

The basic network:

```text
┌─────────┐
│ Input x │
└────┬────┘
     │
     │ w₁
     ▼
┌──────────┐
│ Hidden h │
└────┬─────┘
     │
     │ w₂
     ▼
┌──────────┐
│ Output y │
└────┬─────┘
     │
     ▼
┌──────────┐
│  Loss E  │
└──────────┘
```

During the forward pass:

```text
x → h → y → E
```

is highlighted.

During the backward pass:

```text
E → y → h → weights
```

is highlighted.

The visualizer can display:

```text
Current value
Local derivative
Gradient
Updated value
```

for every operation.

---

# Example Visualizer State

For example:

```text
Input x
1.0000

Hidden pre-activation zₕ
0.5000

Hidden activation h
0.6225

Output pre-activation zᵧ
0.4980

Prediction y
0.6220

Target d
1.0000

Loss E
0.0714
```

Then:

```text
∂E/∂w₂
-0.0552

∂E/∂w₁
-0.0167
```

Then:

```text
Old w₁ = 0.5000
New w₁ = 0.5017

Old w₂ = 0.8000
New w₂ = 0.8055
```

This makes the complete training step observable.

---

# Project Architecture

The project is designed in layers.

```text
                    ┌─────────────────────┐
                    │     Visualizer      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │     Neural Network  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    Custom Autograd  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Computational Graph │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Manual Backprop     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Calculus + Chain    │
                    │ Rule                │
                    └─────────────────────┘
```

The project therefore goes from **theory → implementation → automation → visualization**.

---

# Technology Stack

## Python

Main programming language.

Used for:

* mathematical implementation
* neural network implementation
* autograd engine
* experiments

---

## NumPy

Used for:

* vectors
* matrices
* matrix multiplication
* numerical calculations

Example:

```python
import numpy as np

x = np.array([[1.0],
              [2.0],
              [3.0]])

W = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6]])

z = W @ x
```

---

## Streamlit

Used to create the interactive learning interface.

The user can change:

```text
Input
Weights
Target
Learning rate
```

and observe how the network changes.

---

## Plotly

Used for interactive visualizations such as:

* loss curves
* computational graphs
* gradient visualizations
* training progress
* matrix visualizations

---

# Project Structure

A possible project structure is:

```text
backprop-autograd-visualizer/
│
├── README.md
├── requirements.txt
├── app.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── manual_backprop.py
│   ├── autograd.py
│   ├── neural_network.py
│   ├── computational_graph.py
│   └── utils.py
│
├── visualization/
│   ├── __init__.py
│   ├── graph_visualizer.py
│   ├── gradient_visualizer.py
│   └── training_visualizer.py
│
├── tests/
│   ├── test_derivatives.py
│   ├── test_autograd.py
│   └── test_network.py
│
└── notebooks/
    ├── 01_derivatives.ipynb
    ├── 02_chain_rule.ipynb
    ├── 03_manual_backprop.ipynb
    ├── 04_autograd.ipynb
    └── 05_matrix_network.ipynb
```

---

# Running the Project

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project:

```bash
cd backprop-autograd-visualizer
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application should open in your browser.

---

# Requirements

A basic `requirements.txt` can contain:

```text
numpy
streamlit
plotly
networkx
```

Additional packages can be added as the project grows.

---

# Learning Roadmap

The project follows a deliberate learning sequence.

## Phase 1 — Calculus

Learn:

```text
Functions
   ↓
Derivatives
   ↓
Partial Derivatives
   ↓
Chain Rule
```

---

## Phase 2 — Single Neuron

Implement:

```text
x
 ↓
z
 ↓
sigmoid
 ↓
y
 ↓
loss
```

Then calculate:

```text
dL/dw
```

manually.

---

## Phase 3 — One Hidden Neuron

Implement:

```text
Input
 ↓
Hidden
 ↓
Output
 ↓
Loss
```

Calculate:

```text
∂E/∂w₁
∂E/∂w₂
```

manually.

---

## Phase 4 — Manual Backpropagation

Implement the complete process:

```text
Forward pass
      ↓
Loss
      ↓
Backward pass
      ↓
Gradients
      ↓
Weight update
```

without using an automatic differentiation library.

---

## Phase 5 — Computational Graph

Represent every operation as a node.

For example:

```text
x ──┐
    × ──> z
w ──┘
      │
      ▼
   sigmoid
      │
      ▼
      y
```

---

## Phase 6 — Custom Autograd

Build a `Value` object capable of:

```text
+
-
*
/
power
exp
log
sigmoid
```

Each operation should record its local derivative.

---

## Phase 7 — Backward Engine

Implement:

```python
.backward()
```

The method should:

1. Build a topological ordering of the graph.
2. Start the output gradient at `1`.
3. Traverse the graph backward.
4. Apply each local derivative.
5. Accumulate gradients.

---

## Phase 8 — Matrix Neural Network

Move from:

```text
1 → 1 → 1
```

to:

```text
3 → 2 → 1
```

Then eventually:

```text
n → hidden layers → m
```

---

## Phase 9 — Visualization

Visualize:

```text
Computational graph
Forward pass
Backward pass
Gradients
Weight updates
Loss
Training progress
```

---

## Phase 10 — Deployment

The final application can be deployed as an interactive educational tool.

Possible deployment options include:

```text
Streamlit-based hosting
Cloud deployment
Docker
```

---

# Gradient Checking

An important part of the project is verifying that our autograd implementation is correct.

We can compare:

```text
Analytical gradient
```

against:

```text
Numerical gradient
```

The numerical derivative can be approximated using:

```text
f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)
```

For a weight `w`:

```text
numerical_gradient
=
[E(w + ε) - E(w - ε)]
/
(2ε)
```

Then compare it with:

```text
autograd_gradient
```

If the values are very close, our implementation is probably correct.

---

# Why Gradient Checking Matters

A tiny mistake in backpropagation can produce completely incorrect training.

For example:

```text
Correct gradient:
-0.0167

Incorrect gradient:
+0.167
```

The network may still run.

Python may not produce an error.

But training can fail.

Gradient checking helps catch these silent mathematical errors.

---

# What This Project Is Trying to Teach

This project is not just about making a neural network.

It is about understanding what happens underneath neural-network libraries.

By the end, the goal is to understand:

```text
What is a derivative?
```

```text
What is a partial derivative?
```

```text
What is the chain rule?
```

```text
What is a gradient?
```

```text
What is forward propagation?
```

```text
What is backpropagation?
```

```text
How does a hidden neuron receive an error signal?
```

```text
How does gradient descent update a weight?
```

```text
What is a computational graph?
```

```text
What is automatic differentiation?
```

```text
How does an autograd engine work?
```

```text
How does matrix multiplication implement neural-network layers?
```

---

# Backpropagation in One Picture

The entire idea can be summarized as:

```text
                 FORWARD PASS
                     
Input
  │
  ▼
Weighted Sum
  │
  ▼
Activation
  │
  ▼
Prediction
  │
  ▼
Loss
  │
  │
  │
  ▼
BACKWARD PASS

Loss
  │
  ▼
Gradients
  │
  ▼
Weight Updates
  │
  ▼
Better Prediction
```

Then repeat:

```text
Forward
   ↓
Loss
   ↓
Backward
   ↓
Update
   ↓
Forward
   ↓
...
```

---

# Backpropagation vs Autograd

These concepts are related but should not be treated as exactly the same thing.

## Backpropagation

Backpropagation is an algorithm for efficiently calculating gradients through a layered computational structure.

It uses the chain rule.

---

## Automatic Differentiation

Automatic differentiation is a broader technique for automatically computing derivatives by decomposing computations into elementary operations.

Our custom autograd engine uses reverse-mode automatic differentiation.

For a neural-network loss with many parameters and a small number of outputs, reverse-mode autodiff is particularly useful.

The relationship can be viewed as:

```text
Calculus
   ↓
Chain Rule
   ↓
Computational Graph
   ↓
Reverse-Mode Automatic Differentiation
   ↓
Backpropagation
```

---

# Limitations

The original paper discusses important limitations of backpropagation.

One important issue is that gradient-based optimization can encounter **local minima**.

This means the optimization process does not necessarily guarantee that the network will find the global minimum of the error function.

The paper also discusses the biological plausibility of backpropagation and notes that the learning procedure described is not intended as a biologically realistic model of learning.

Our implementation has additional educational limitations:

* It starts with very small networks.
* It focuses on understanding rather than performance.
* It is not intended to compete with production deep-learning frameworks.
* The visualizer prioritizes transparency over computational efficiency.
* The custom autograd engine is intentionally small.

---

# What Makes This Project Different?

Instead of starting with:

```python
import torch

loss.backward()
```

we start with:

```text
Why does backward() work?
```

Then we build it ourselves.

The learning sequence is:

```text
Math
 ↓
Manual calculation
 ↓
Python implementation
 ↓
Computational graph
 ↓
Autograd
 ↓
Neural network
 ↓
Visualization
```

The objective is to turn something that often feels like a black box into something that can be inspected step by step.

---

# Future Improvements

Possible future extensions include:

## More Layers

```text
Input
 ↓
Hidden 1
 ↓
Hidden 2
 ↓
Hidden 3
 ↓
Output
```

---

## More Activation Functions

Implement:

```text
Sigmoid
ReLU
Tanh
Softmax
```

---

## More Loss Functions

Implement:

```text
Mean Squared Error
Binary Cross Entropy
Cross Entropy
```

---

## Bias Parameters

Extend the network from:

```text
z = Wx
```

to:

```text
z = Wx + b
```

---

## Mini-Batch Training

Instead of training one example at a time:

```text
sample 1
sample 2
sample 3
...
```

train using batches.

---

## Optimizers

Implement:

```text
Gradient Descent
Momentum
Adam
```

---

## Better Computational Graph

Allow users to inspect:

```text
Node
Value
Operation
Local derivative
Gradient
Parents
```

for every operation.

---

# The Ultimate Goal

The final version of this project should allow someone to enter:

```text
x = 1
w₁ = 0.5
w₂ = 0.8
target = 1
learning rate = 0.1
```

and visually watch:

```text
1. Forward pass starts
        ↓
2. zₕ is calculated
        ↓
3. sigmoid produces h
        ↓
4. zᵧ is calculated
        ↓
5. sigmoid produces y
        ↓
6. Loss is calculated
        ↓
7. Backward pass starts
        ↓
8. Gradients are calculated
        ↓
9. Weights are updated
        ↓
10. Loss changes
```

Then the same idea should scale to:

```text
3 inputs
    ↓
2 hidden neurons
    ↓
1 output
```

and eventually to larger neural networks.

---

# References

## Research Paper

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).

**Learning representations by back-propagating errors.**

Nature, 323, 533–536.

DOI:

```text
https://doi.org/10.1038/323533a0
```

---

# Project Philosophy

> **Don't just use backpropagation. Understand it.**

Modern machine-learning frameworks make training neural networks extremely easy.

But the mathematics underneath is still:

```text
Derivatives
+
Chain Rule
+
Computational Graph
+
Gradient Descent
```

This project is an attempt to make those ideas visible.

Instead of treating:

```python
loss.backward()
```

as magic, we build the mechanism ourselves.

---

# Final Learning Map

```text
                 MACHINE LEARNING
                        │
                        ▼
                   NEURAL NETWORK
                        │
                        ▼
                  FORWARD PASS
                        │
                        ▼
                      LOSS
                        │
                        ▼
                  CHAIN RULE
                        │
                        ▼
                  BACKPROPAGATION
                        │
                        ▼
                    GRADIENTS
                        │
                        ▼
                 GRADIENT DESCENT
                        │
                        ▼
                COMPUTATIONAL GRAPH
                        │
                        ▼
                 AUTOMATIC DIFFERENTIATION
                        │
                        ▼
                  CUSTOM AUTOGRAD
                        │
                        ▼
                    VISUALIZER
```

**The goal is not just to train a neural network.**

**The goal is to understand every number that makes the network learn.**
