# 🧠 Backpropagation & Autograd Visualizer

### Understanding Neural Networks by Rebuilding Backpropagation from Scratch

An interactive, educational project inspired by the research paper **“Learning representations by back-propagating errors” (1986)** by David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams.

This project explores how neural networks learn, how hidden neurons develop useful representations, how errors travel backward through a network, and how gradients help update weights.

Rather than hiding the mathematics behind a deep-learning framework, this project aims to make each calculation visible and understandable. We begin with a tiny neural network, implement backpropagation manually, and gradually build our own miniature autograd engine.

> **The central idea:** A neural network can learn useful internal representations by adjusting its connection weights to reduce the difference between its predictions and the desired outputs.

---

## 📚 Table of Contents

* [1. About the Research Paper](#1-about-the-research-paper)
* [2. Why Was This Research Important?](#2-why-was-this-research-important)
* [3. The Problem: Learning Hidden Representations](#3-the-problem-learning-hidden-representations)
* [4. How a Neural Network Works](#4-how-a-neural-network-works)
* [5. Forward Propagation](#5-forward-propagation)
* [6. The Error Function](#6-the-error-function)
* [7. Why Do We Need Backpropagation?](#7-why-do-we-need-backpropagation)
* [8. The Chain Rule](#8-the-chain-rule)
* [9. Backpropagation: The Mathematics](#9-backpropagation-the-mathematics)
* [10. Gradient Descent and Weight Updates](#10-gradient-descent-and-weight-updates)
* [11. How Hidden Units Learn Representations](#11-how-hidden-units-learn-representations)
* [12. Experiments in the Paper](#12-experiments-in-the-paper)
* [13. Limitations Discussed in the Paper](#13-limitations-discussed-in-the-paper)
* [14. From the Paper to Our Implementation](#14-from-the-paper-to-our-implementation)
* [15. Our Autograd Engine](#15-our-autograd-engine)
* [16. Project Architecture](#16-project-architecture)
* [17. Technology Stack](#17-technology-stack)
* [18. Running the Project](#18-running-the-project)
* [19. Learning Roadmap](#19-learning-roadmap)
* [20. What This Project Teaches](#20-what-this-project-teaches)
* [21. References](#21-references)

---

## 1. About the Research Paper

**Paper:** Learning representations by back-propagating errors
**Authors:** David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
**Published:** 1986
**Journal:** Nature, Volume 323, pages 533–536
**DOI:** [10.1038/323533a0](https://doi.org/10.1038/323533a0)

The paper describes a learning procedure called **back-propagation**, which repeatedly adjusts the weights of connections in a neural network to minimize the difference between actual outputs and desired outputs.

The authors demonstrate that this process can do more than simply map inputs to outputs. It can also allow intermediate, or *hidden*, units to develop internal representations that are useful for solving a task.

This is a particularly important idea because, in many learning problems, we know what answer a network should produce, but we do not know what every intermediate neuron should represent.

The paper studies how a network can learn those internal representations through the process of adjusting its weights.

### The paper's central contributions

* Describes a general learning procedure for layered networks of neuron-like units.
* Shows how derivatives of an error function can be propagated backward through a network.
* Explains how weights can be adjusted using gradient descent.
* Demonstrates that hidden units can learn useful internal representations.
* Presents experiments involving symmetry detection and family-tree relationships.
* Discusses the limitations of gradient descent and the biological plausibility of the proposed procedure.

This project uses these ideas as the foundation for an interactive implementation.

---

## 2. Why Was This Research Important?

Consider a simple task: a neural network receives an input and must produce a correct answer.

If the input units connect directly to the output units, it is relatively straightforward to adjust the connection weights based on the output error.

However, many tasks are more complex. The network may need to recognize patterns, combine information, or discover relationships that are not explicitly provided as inputs.

This is where hidden units become useful.

Imagine asking a network to recognize whether a visual pattern is symmetrical. The input describes the pattern, and the output indicates whether it is symmetrical. But what should the hidden neurons detect to solve the problem?

We might manually design features for the network, but that requires us to know in advance which features are useful.

The paper explores a different approach: let the network learn suitable internal representations by adjusting its weights according to the error at its output.

### The key challenge

A network receives a desired answer at its output, but the task does not directly specify the correct activation of each hidden neuron.

Backpropagation provides a way to calculate how changing those hidden activations and their incoming weights would affect the final error.

This gives the network a learning signal for its intermediate layers.

---

## 3. The Problem: Learning Hidden Representations

A neural network can contain three broad types of layers:

| Layer         | Purpose                                                    |
| ------------- | ---------------------------------------------------------- |
| Input layer   | Receives the information supplied to the network           |
| Hidden layers | Transform information and develop internal representations |
| Output layer  | Produces the network's prediction                          |

A simple network can be represented as:

```text
Input Layer       Hidden Layer       Output Layer

   x₁ ────────────► h₁ ──────────────►
                    ▲                  │
   x₂ ─────────────┤                  ▼
                    │                  y
   x₃ ────────────► h₂ ──────────────►
```

The hidden units are not simply fixed feature detectors. Their incoming weights can change during learning, allowing their responses to adapt to the task.

For example, a hidden neuron may become responsive to a particular combination of input patterns. Another hidden neuron may learn a different combination.

The network's useful internal representation emerges through the interactions of these learned units.

### What does “representation” mean?

A representation is the way information is encoded inside a network.

For example, the original input might describe an object using many individual values. A hidden layer transforms those values into a new pattern of activations.

That new pattern may make it easier for the output layer to produce the desired answer.

The important point is that a hidden representation does not necessarily correspond to a human-readable concept. It is an internal pattern of activity that helps the network perform its task.

---

## 4. How a Neural Network Works

The paper describes units that receive values from lower layers, combine them using connection weights, and apply a nonlinear function to produce their outputs.

A typical unit calculates a weighted sum:

$$
z_j = \sum_i w_{ij}y_i + b_j
$$

Where:

* \(y_i\) is the output from a connected unit in the previous layer.
* \(w_{ij}\) is the weight of the connection from unit \(i\) to unit \(j\).
* \(b_j\) is the bias.
* \(z_j\) is the total input to the unit.

The unit then applies an activation function:

$$
y_j = f(z_j)
$$

For our initial implementation, we use the sigmoid activation function:

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

The sigmoid function maps any real-valued input to a value between 0 and 1.

### Why use an activation function?

Without a nonlinear activation function, stacking layers of weighted sums would still produce an overall linear transformation.

Nonlinear activation functions allow a network to represent more complex input-output relationships.

### What is a bias?

A bias is an additional adjustable parameter that shifts a unit's weighted input before the activation function is applied.

It can be understood as an additional weight connected to an input that is always equal to 1.

The paper treats biases as weights in this way.

---

## 5. Forward Propagation

Forward propagation is the process of calculating the network's outputs from its inputs.

We begin with a very small network:

```text
       Input           Hidden           Output

         x ── w₁ ──►     h ── w₂ ──►      y
```

For simplicity, this initial example has:

* One input value
* One hidden neuron
* One output neuron
* Two trainable weights
* No explicit bias terms

Let:

$$
x = 1,\quad w_1 = 0.5,\quad w_2 = 0.8
$$

### Step 1: Calculate the hidden neuron's weighted input

$$
z_h = w_1x
$$

$$
z_h = 0.5 \times 1 = 0.5
$$

### Step 2: Calculate the hidden neuron's activation

$$
h = \sigma(z_h)
$$

$$
h = \sigma(0.5) \approx 0.6225
$$

The hidden neuron now has an activation of approximately 0.6225.

### Step 3: Calculate the output neuron's weighted input

$$
z_y = w_2h
$$

$$
z_y = 0.8 \times 0.6225 \approx 0.4980
$$

### Step 4: Calculate the output

$$
y = \sigma(z_y)
$$

$$
y = \sigma(0.4980) \approx 0.6220
$$

The network's prediction is approximately 0.6220.

### Forward pass summary

| Quantity              | Calculation     | Approximate value |
| --------------------- | --------------- | ----------------: |
| Input                 | \(x\)           |            1.0000 |
| Hidden weighted input | \(w_1x\)        |            0.5000 |
| Hidden activation     | \(\sigma(z_h)\) |            0.6225 |
| Output weighted input | \(w_2h\)        |            0.4980 |
| Prediction            | \(\sigma(z_y)\) |            0.6220 |

Forward propagation calculates the prediction. It does not, by itself, tell us how to improve the weights.

For that, we need an error function and a learning procedure.

---

## 6. The Error Function

A learning system needs a way to measure how different its prediction is from the desired output.

Let:

* \(y\) be the actual output produced by the network.
* \(d\) be the desired output.
* \(E\) be the error.

For our simple example, we use squared error:

$$
E = \frac{1}{2}(d-y)^2
$$

The factor \(\frac{1}{2}\) is included because it simplifies the derivative.

Suppose the desired output is:

$$
d = 1
$$

And the network prediction is:

$$
y \approx 0.6220
$$

Then:

$$
E = \frac{1}{2}(1-0.6220)^2
$$

$$
E \approx 0.0714
$$

The objective is to reduce this error by adjusting the weights.

### Error across multiple cases

The paper also describes the total error over a finite set of input-output cases.

In simplified notation:

$$
E = \frac{1}{2}\sum_c\sum_j (d_{cj}-y_{cj})^2
$$

Where:

* \(c\) identifies an input-output case.
* \(j\) identifies an output unit.
* \(d_{cj}\) is the desired output.
* \(y_{cj}\) is the actual output.

The total error combines the output differences across the cases and output units being considered.

Our first visualizer uses a single case and a single output so the mathematics remains easy to follow.

---

## 7. Why Do We Need Backpropagation?

After the forward pass, we know the prediction and the error.

But we still need to answer:

* Which weights contributed to the error?
* How sensitive is the error to each weight?
* Should each weight increase or decrease?
* By how much should we change it?

A simple network has only a few weights, so we might calculate the effect of each weight individually.

But a larger neural network can contain many layers and thousands or millions of weights. Recalculating the entire network's error for every possible weight change would be inefficient.

Backpropagation addresses this by using derivatives and the chain rule to calculate how the error depends on the network's weights.

It begins at the output, where the error is measured, and propagates derivative information backward through the network.

```text
Forward pass

Input → Hidden → Output → Error
  ────────────────────────────►

Backward pass

Input ← Hidden ← Output ← Error
  ◄────────────────────────────
```

The forward pass calculates values.

The backward pass calculates gradients.

Together, they provide the information needed to update the network's weights.

---

## 8. The Chain Rule

The chain rule is the mathematical foundation of backpropagation.

It tells us how to calculate the derivative of a quantity that depends on another quantity through an intermediate variable.

Suppose:

$$
z = f(x)
$$

and:

$$
E = g(z)
$$

Then:

$$
\frac{\partial E}{\partial x}
=
\frac{\partial E}{\partial z}
\frac{\partial z}{\partial x}
$$

In simple words, if changing \(x\) changes \(z\), and changing \(z\) changes the error, the chain rule combines those two effects.

### A simple example

Suppose:

$$
z = 2x
$$

$$
E = z^2
$$

Then:

$$
\frac{dE}{dz}=2z
$$

And:

$$
\frac{dz}{dx}=2
$$

Using the chain rule:

$$
\frac{dE}{dx}
=
\frac{dE}{dz}\frac{dz}{dx}
$$

$$
= 2z \times 2
$$

Since \(z=2x\):

$$
\frac{dE}{dx}=8x
$$

We calculated how the final error changes with \(x\), even though the error depends on \(x\) through an intermediate variable.

### Why is this important for neural networks?

A neural network contains many connected operations:

```text
weight → multiplication → sum → activation → next layer → loss
```

The output depends on earlier weights through a chain of intermediate values.

The chain rule lets us calculate each weight's contribution to the final error by multiplying the relevant local derivatives along the path.

This is the core idea that our visualizer will make visible.

---

## 9. Backpropagation: The Mathematics

We will derive backpropagation for our one-hidden-neuron network.

The network is:

$$
z_h=w_1x
$$

$$
h=\sigma(z_h)
$$

$$
z_y=w_2h
$$

$$
y=\sigma(z_y)
$$

$$
E=\frac{1}{2}(d-y)^2
$$

We want to calculate:

$$
\frac{\partial E}{\partial w_1}
\quad\text{and}\quad
\frac{\partial E}{\partial w_2}
$$

### Step 1: Derivative of error with respect to output

$$
E=\frac{1}{2}(d-y)^2
$$

Differentiating with respect to \(y\):

$$
\frac{\partial E}{\partial y}=y-d
$$

This derivative tells us how the error changes when the prediction changes.

### Step 2: Derivative of sigmoid

For sigmoid:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Its derivative can be written in terms of its output:

$$
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

Therefore:

$$
\frac{\partial y}{\partial z_y}=y(1-y)
$$

### Step 3: Gradient of the output weight

The output weighted input is:

$$
z_y=w_2h
$$

So:

$$
\frac{\partial z_y}{\partial w_2}=h
$$

Using the chain rule:

$$
\frac{\partial E}{\partial w_2}
=
\frac{\partial E}{\partial y}
\frac{\partial y}{\partial z_y}
\frac{\partial z_y}{\partial w_2}
$$

Substituting:

$$
\boxed{
\frac{\partial E}{\partial w_2}
=
(y-d)y(1-y)h
}
$$

This is the gradient for the weight connecting the hidden neuron to the output neuron.

### Step 4: Gradient of the hidden weight

The hidden weight affects the error through several intermediate values:

$$
w_1 \rightarrow z_h \rightarrow h \rightarrow z_y \rightarrow y \rightarrow E
$$

Therefore:

$$
\frac{\partial E}{\partial w_1}
=
\frac{\partial E}{\partial y}
\frac{\partial y}{\partial z_y}
\frac{\partial z_y}{\partial h}
\frac{\partial h}{\partial z_h}
\frac{\partial z_h}{\partial w_1}
$$

We already know:

$$
\frac{\partial E}{\partial y}=y-d
$$

$$
\frac{\partial y}{\partial z_y}=y(1-y)
$$

Also:

$$
\frac{\partial z_y}{\partial h}=w_2
$$

$$
\frac{\partial h}{\partial z_h}=h(1-h)
$$

And:

$$
\frac{\partial z_h}{\partial w_1}=x
$$

Combining these:

$$
\boxed{
\frac{\partial E}{\partial w_1}
=
(y-d)y(1-y)w_2h(1-h)x
}
$$

Notice that the gradient for \(w_1\) includes more factors because its effect travels through the hidden neuron before reaching the output.

### Numerical example

Using:

$$
x=1,\quad w_1=0.5,\quad w_2=0.8,\quad d=1
$$

The forward pass gives approximately:

$$
h=0.6225,\quad y=0.6220
$$

The gradients are approximately:

$$
\frac{\partial E}{\partial w_2}\approx -0.0552
$$

$$
\frac{\partial E}{\partial w_1}\approx -0.0167
$$

These values describe how the error changes when each weight changes slightly, while the other values are held fixed.

The negative signs indicate that increasing these weights locally reduces the error for this example.

---

## 10. Gradient Descent and Weight Updates

Calculating gradients is not the final step. We use them to adjust the weights.

The paper describes minimizing error using gradient descent.

The basic update rule is:

$$
\boxed{
w_{\text{new}}=w_{\text{old}}-\eta\frac{\partial E}{\partial w}
}
$$

Where:

* \(w_{\text{old}}\) is the current weight.
* \(\eta\) is the learning rate.
* \(\frac{\partial E}{\partial w}\) is the gradient.
* \(w_{\text{new}}\) is the updated weight.

The learning rate controls the size of the update.

### Example

Suppose:

$$
w_2=0.8
$$

$$
\frac{\partial E}{\partial w_2}=-0.0552
$$

$$
\eta=0.1
$$

Then:

$$
w_{2,\text{new}}
=
0.8-0.1(-0.0552)
$$

$$
w_{2,\text{new}}\approx 0.80552
$$

The weight increases because its gradient is negative.

The same procedure is applied to \(w_1\).

### What happens after updating?

The network's prediction and error may change after the weights are updated.

So training repeats the process:

1. Perform a forward pass.
2. Calculate the error.
3. Perform a backward pass.
4. Calculate gradients.
5. Update the weights.
6. Repeat.

The goal is to reduce the error over the training cases.

### Momentum

The paper also discusses an extension involving momentum.

Momentum incorporates a contribution from previous weight changes into the current update. This can affect the path taken during optimization.

Our first implementation focuses on the basic gradient-descent rule. Momentum can be added as a later extension.

---

## 11. How Hidden Units Learn Representations

One of the paper's important contributions is its demonstration that hidden units can learn useful internal representations without being given explicit target activations.

Consider a task with inputs and desired outputs. The network is trained to produce the correct output, but the desired activity of its hidden units is not directly specified.

Backpropagation calculates how the output error depends on the hidden units. That information can be used to adjust the weights leading into those units.

Over repeated learning steps, hidden units can develop activation patterns that help the network solve the task.

### Distributed representations

Information does not have to be represented by a single neuron dedicated to one concept.

Instead, a pattern across multiple hidden units can represent useful information. Different patterns of activity can encode different information, and the same unit can contribute to more than one representation.

The paper's family-tree experiment provides an example of learned distributed representations.

### Why does this matter?

The network is not simply memorizing a fixed collection of manually designed features. Its internal connection weights can change to construct representations that support the required input-output behavior.

This is the sense in which backpropagation can help a network *learn representations*.

---

## 12. Experiments in the Paper

The paper illustrates the learning procedure through tasks designed to demonstrate what hidden units can learn.

### Experiment 1: Symmetry detection

The paper studies a task in which the network must determine whether an input pattern is symmetrical.

The network receives a pattern and must produce an output corresponding to the symmetry decision.

The authors show that the learning procedure can discover a solution using two intermediate units.

The hidden units develop complementary responses that allow the network to distinguish symmetrical from non-symmetrical patterns.

This example demonstrates that a network can learn a useful internal solution rather than relying entirely on features manually specified in advance.

### Experiment 2: Family-tree relationships

The second experiment involves two isomorphic family trees.

The information is expressed as triples of the form:

```text
(person 1, relationship, person 2)
```

For example, the network may receive a person and a relationship and be trained to produce the person who completes the relationship.

The task includes relationships such as:

* Father
* Mother
* Husband
* Wife
* Son
* Daughter
* Uncle
* Aunt
* Brother
* Sister
* Nephew
* Niece

The paper describes a layered network that learns distributed representations of people and relationships.

The network was trained on 100 of the 104 possible triples. The paper uses its learned internal activity patterns to illustrate how representations develop within the hidden layers.

The task is important because the network must learn relationships between entities rather than simply return an unrelated output for each input.

### What do the experiments demonstrate?

Together, the experiments illustrate that backpropagation can adjust weights so that hidden units develop internal activity patterns useful for a task.

They support the paper's central argument that learning can construct internal representations through gradient descent.

They are demonstrations of the learning procedure on specific tasks, not a claim that every neural-network problem will be solved easily or that every learned representation will be human-interpretable.

---

## 13. Limitations Discussed in the Paper

The paper identifies important limitations.

### Local minima

The error surface of a network can contain local minima.

Gradient descent follows local gradient information. It is therefore not guaranteed to find a global minimum.

The authors note that, in their experience with many tasks, networks did not often become stuck in significantly poor local minima. They also discuss how adding some additional connections can create extra dimensions in weight space and may provide paths around certain barriers.

This is an observation and explanation from the paper, not a guarantee for every architecture or dataset.

### Biological plausibility

The authors explicitly state that the learning procedure, in its current form, is not a plausible model of learning in brains.

The paper presents backpropagation as a powerful computational learning procedure and suggests looking for more biologically plausible ways to perform gradient descent in neural networks.

### Learning useful representations is task-dependent

The hidden representations are developed through the learning objective and network structure.

The procedure does not mean that every hidden unit will learn a simple, meaningful concept or that the learned features will always be easy for a person to interpret.

### What we take from these limitations

Backpropagation provides a way to calculate gradients and improve a network's performance, but it does not remove all challenges of learning.

Understanding the method includes understanding both what it enables and what it does not guarantee.

---

## 14. From the Paper to Our Implementation

This project translates the paper's mathematical ideas into a small, interactive program.

The paper explains a general learning procedure for layered networks. Our first implementation deliberately uses a much smaller network so that every calculation can be inspected.

### Our first network

```text
Input
  │
  ▼
Hidden neuron
  │
  ▼
Output neuron
  │
  ▼
Loss
```

The implementation calculates:

* Weighted inputs
* Sigmoid activations
* Prediction
* Error
* Local derivatives
* Gradients for weights
* Gradient-descent updates

### What is simplified?

Our first example has one input, one hidden neuron, one output neuron, and two weights. It omits explicit bias parameters and uses a single input-output case.

The paper's procedure is more general. It covers layered networks with multiple units and cases, and it explains how derivatives from multiple connected units contribute to the gradients.

The small example is a teaching model that makes the central chain-rule process easy to follow.

### How the project will grow

After the manual version works, we will implement a `Value` object that stores values, gradients, parent relationships, and local backward functions.

This will let us build a computational graph and calculate gradients by traversing it backward, rather than writing a separate full gradient formula for each network weight.

Later, the project will expand to multiple neurons and matrix operations.

---

## 15. Our Autograd Engine

Automatic differentiation calculates derivatives by tracking how values are produced through a sequence of operations.

Our goal is to implement a small version of this idea ourselves.

### The `Value` object

Each value in the computational graph will store:

| Property    | Meaning                                  |
| ----------- | ---------------------------------------- |
| `data`      | The numerical value                      |
| `grad`      | The accumulated gradient                 |
| `parents`   | The values used to produce it            |
| `operation` | The operation that created it            |
| `backward`  | The local rule for propagating gradients |

A simplified example:

```python
class Value:
    def __init__(self, data, parents=(), operation=""):
        self.data = data
        self.grad = 0.0
        self.parents = set(parents)
        self.operation = operation
        self._backward = lambda: None
```

This is only the starting structure. Operations such as addition, multiplication, powers, and sigmoid will each need to define how gradients are propagated to their inputs.

### Example computational graph

Suppose:

```python
x = Value(1.0)
w = Value(0.5)

z = x * w
h = z.sigmoid()
```

The graph is:

```text
 x ─────┐
        ▼
      Multiply ───► z ───► Sigmoid ───► h
        ▲
 w ─────┘
```

Each operation creates a new value and records which earlier values it depends on.

During the backward pass, the engine calculates local derivatives and combines them with the gradients arriving from later operations.

### Why build our own?

A ready-made autograd library can calculate gradients without exposing every internal step.

Our educational engine is intended to show how the process works:

1. Values are created.
2. Operations connect those values.
3. A computational graph is formed.
4. A starting gradient is assigned at the output.
5. Gradients are propagated backward.
6. The gradients can be used to update parameters.

This recreates the central mechanics of reverse-mode automatic differentiation on a small scale. It is not intended to reproduce every feature of a mature deep-learning framework.

---

## 16. Project Architecture

The project separates the mathematics, network structure, and visualization.

```text
backprop-visualizer/
│
├── app.py
│
├── autograd/
│   ├── __init__.py
│   ├── value.py
│   └── operations.py
│
├── neural_network/
│   ├── __init__.py
│   ├── neuron.py
│   ├── layer.py
│   └── network.py
│
├── visualization/
│   ├── network_graph.py
│   ├── computation_graph.py
│   └── charts.py
│
├── utils/
│   └── math_utils.py
│
├── tests/
│   ├── test_operations.py
│   ├── test_gradients.py
│   └── test_network.py
│
├── requirements.txt
│
└── README.md
```

### `app.py`

The Streamlit application. It brings together the controls, mathematical results, visualizations, and explanations.

### `autograd/`

Contains the miniature automatic differentiation engine.

### `neural_network/`

Contains the structures needed to build neurons, layers, and complete networks.

### `visualization/`

Contains the network view, computational graph, and charts.

### `utils/`

Contains reusable mathematical helper functions.

### `tests/`

Contains checks for operations, gradients, and network behavior.

This separation helps keep the mathematical implementation independent of the user interface.

---

## 17. Technology Stack

| Technology                | Role in the project                          |
| ------------------------- | -------------------------------------------- |
| Python                    | Main programming language                    |
| NumPy                     | Numerical calculations and matrix operations |
| Streamlit                 | Interactive web application                  |
| Plotly                    | Interactive network views and charts         |
| NetworkX                  | Computational graph structure, if needed     |
| Git and GitHub            | Version control and public source repository |
| Streamlit Community Cloud | Planned deployment platform                  |

### Why Streamlit?

Streamlit allows us to build an interactive Python application without first creating a separate frontend in JavaScript.

It provides controls such as sliders, buttons, columns, and expandable sections. These are useful for changing network parameters and inspecting calculations.

### Why Plotly?

Plotly can display interactive charts and diagrams. It will be used to visualize the network, computational graph, gradients, and loss history.

### Why NumPy?

NumPy provides arrays and matrix operations that will become especially useful when the project expands from a single neuron to a network with multiple inputs and hidden neurons.

### Why not use PyTorch autograd?

The initial learning objective is to understand and implement the calculations, not simply call an existing gradient engine.

We can explore PyTorch later for comparison, but our core educational implementation will calculate gradients ourselves.

---

## 18. Running the Project

### Requirements

* Python 3.10 or a compatible newer version
* pip
* A web browser

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/backprop-visualizer.git
cd backprop-visualizer
```

Replace `YOUR_USERNAME` with your GitHub username and use the actual repository name if it differs.

### 2. Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The initial `requirements.txt` will include:

```text
streamlit
numpy
plotly
```

Additional packages, such as NetworkX, can be added when the computational-graph visualization requires them.

### 4. Run the application

```bash
streamlit run app.py
```

Streamlit will start the application and provide a local address to open in your browser.

### 5. Explore the network

Change the input, weights, target, and learning rate. Run the forward and backward steps and inspect the calculations.

The goal is to understand not just the final prediction, but how the network arrived at it and how the gradients influence the next update.

---

## 19. Learning Roadmap

The project will be developed in stages so that every concept is understood before adding more complexity.

### Phase 1 — Manual forward and backward propagation

Build a one-input, one-hidden-neuron, one-output network.

Implement the sigmoid function, loss, chain-rule derivatives, gradients, and weight updates manually.

### Phase 2 — Interactive visualization

Display the network, intermediate values, error, gradients, and updated weights.

Add controls to change the input and network parameters and inspect their effects.

### Phase 3 — Build our own autograd engine

Implement a `Value` class with arithmetic operations and backward functions.

Create a computational graph automatically as values are combined.

### Phase 4 — Visualize the computational graph

Display operations as nodes and dependencies as edges.

Show each node's value and gradient.

### Phase 5 — Step-by-step execution

Allow users to execute one operation at a time during the forward and backward passes.

Highlight the active node and explain the calculation taking place.

### Phase 6 — Multiple neurons

Expand the network to:

```text
3 inputs → 2 hidden neurons → 1 output
```

Introduce weight matrices, bias vectors, and vectorized operations.

### Phase 7 — Matrix visualization

Show the dimensions of input vectors, weight matrices, and output vectors.

Visualize how matrix multiplication combines inputs and weights to produce each layer's values.

### Phase 8 — Training visualization

Repeat forward and backward passes over training steps.

Display changes in weights, predictions, gradients, and loss over time.

### Phase 9 — Gradient checking

Compare gradients from our backward implementation with numerical approximations using finite differences.

This helps identify mistakes in the derivative calculations.

### Phase 10 — Deployment

Prepare the application for deployment using Streamlit Community Cloud, with the source code maintained in GitHub.

---

## 20. What This Project Teaches

By following this project, a learner can explore several connected ideas:

* How a neuron calculates a weighted sum and applies an activation function.
* How neural networks transform input values into predictions.
* How a loss function measures prediction error.
* Why derivatives describe sensitivity to changes in values.
* How the chain rule connects derivatives through multiple operations.
* How backpropagation calculates gradients from output to earlier layers.
* How gradient descent uses those gradients to update weights.
* How hidden units can develop useful internal representations.
* How a computational graph records dependencies between operations.
* How a small autograd engine can propagate gradients automatically.
* How matrix multiplication supports networks with multiple neurons.
* How gradient checking can help validate an implementation.

### The main learning objective

The objective is not just to run a neural network.

It is to understand the process that allows the network to learn, from the first multiplication in the forward pass to the final weight update.

The project connects the mathematical ideas in the 1986 paper to a working implementation that can be inspected and extended.

---

## 21. References

### Primary research paper

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).

**Learning representations by back-propagating errors.**

*Nature, 323*, 533–536.

https://doi.org/10.1038/323533a0

This paper is the primary source for the learning procedure, mathematical framing, hidden representations, experiments, and limitations discussed in this README.

### Related foundational work

The paper situates its method in the broader history of neural-network learning, including earlier perceptron learning procedures and work on distributed representations. Its own reference list provides further sources for readers interested in the historical background.

---

## ✨ Final Note

This project is a learning implementation inspired by the paper, not a reproduction of every experiment or every architectural detail in the original research.

The initial network is intentionally small, and its equations are written explicitly so that the learning process is easy to inspect. As the project grows, the same underlying ideas will be extended to multiple neurons, computational graphs, and an autograd engine built from scratch.

**The goal is to make backpropagation something you can see, calculate, implement, debug, and truly understand.**
