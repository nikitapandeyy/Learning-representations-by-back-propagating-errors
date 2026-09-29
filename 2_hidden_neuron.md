Yes. Let's build the **2-hidden-neuron version** now, but keep it connected to the scalar version we just learned.

Our network will be:

```text
                 ┌── h1 ──┐
x ──►           │         │
                 └── h2 ──┼──► y
```

So:

```text
1 input
   ↓
2 hidden neurons
   ↓
1 output
```

## 1. The architecture

We have one input:

```python
x = 1.0
```

Two hidden neurons:

```text
        ┌── h1
x ──────┤
        └── h2
```

Each hidden neuron gets its **own weight and bias**:

```text
h1 = sigmoid(x*w1_1 + b1_1)

h2 = sigmoid(x*w1_2 + b1_2)
```

Then the output neuron receives **both** hidden neurons:

```text
y = sigmoid(h1*w2_1 + h2*w2_2 + b2)
```

That's the important new idea.

---

# 2. Draw the computation graph

```text
                         ┌── w2_1 ──┐
                         │          │
x ──► w1_1 ──► z1 ──► h1 ──────────┤
│                                    ├──► z_output ──► y
│                                    │
└──► w1_2 ──► z2 ──► h2 ──────────┤
                         │          │
                         └── w2_2 ──┘

                         + b2
```

More explicitly:

```text
z1 = x*w1_1 + b1_1
h1 = sigmoid(z1)

z2 = x*w1_2 + b1_2
h2 = sigmoid(z2)

z_output = h1*w2_1 + h2*w2_2 + b2

y = sigmoid(z_output)
```

---

# 3. Let's code ONLY the forward pass first

Don't worry about backprop yet.

```python
import math


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# -------------------------
# INPUT
# -------------------------

x = 1.0


# -------------------------
# PARAMETERS
# -------------------------

# Hidden neuron 1
w1_1 = 0.5
b1_1 = 0.0

# Hidden neuron 2
w1_2 = -0.5
b1_2 = 0.0


# Output neuron
w2_1 = 0.5
w2_2 = 0.5
b2 = 0.0


# -------------------------
# FORWARD PASS
# -------------------------

# Hidden neuron 1
z1 = x * w1_1 + b1_1
h1 = sigmoid(z1)


# Hidden neuron 2
z2 = x * w1_2 + b1_2
h2 = sigmoid(z2)


# Output neuron
z_output = (
    h1 * w2_1
    + h2 * w2_2
    + b2
)

y = sigmoid(z_output)


# -------------------------
# PRINT
# -------------------------

print("FORWARD PASS")
print("--------------------")

print("x =", x)

print("\nHidden neuron 1")
print("z1 =", z1)
print("h1 =", h1)

print("\nHidden neuron 2")
print("z2 =", z2)
print("h2 =", h2)

print("\nOutput neuron")
print("z_output =", z_output)
print("y =", y)
```

---

# 4. Let's manually follow the numbers

We started with:

```text
x = 1
```

### Hidden neuron 1

```text
w1_1 = 0.5
b1_1 = 0
```

Therefore:

$$
z_1 = xw_{1,1}+b_{1,1}
$$

```text
z1 = 1 × 0.5 + 0
   = 0.5
```

Then:

```text
h1 = sigmoid(0.5)
```

approximately:

```text
h1 = 0.6225
```

---

### Hidden neuron 2

We deliberately gave it a different weight:

```text
w1_2 = -0.5
```

Therefore:

```text
z2 = 1 × (-0.5) + 0
   = -0.5
```

Then:

```text
h2 = sigmoid(-0.5)
```

approximately:

```text
h2 = 0.3775
```

So our hidden layer is:

```text
h = [0.6225, 0.3775]
```

**This is our first tiny vector!**

---

# 5. Now the output neuron

The output receives **both** values:

```text
h1 = 0.6225
h2 = 0.3775
```

and:

```text
w2_1 = 0.5
w2_2 = 0.5
```

Therefore:

$$
z_{output}
=
h_1w_{2,1}
+
h_2w_{2,2}
+
b_2
$$

Substitute:

```text
z_output
= (0.6225)(0.5)
+ (0.3775)(0.5)
+ 0
```

```text
= 0.31125 + 0.18875
```

```text
= 0.5
```

Then:

```text
y = sigmoid(0.5)
```

so:

```text
y ≈ 0.6225
```

---

# 6. Notice what changed

Our previous network was:

```text
x
│
▼
h
│
▼
y
```

There was only **one path**.

Now we have:

```text
              ┌── h1 ──┐
              │        │
x ────────────┤        ├──► y
              │        │
              └── h2 ──┘
```

There are now **two paths from x to y**.

And this is exactly why the matrix version becomes useful.

---

# 7. The same thing as matrices

We currently have:

```text
x = 1 × 1
```

and two hidden neurons.

The weights going from input → hidden layer can be represented as:

$$
W_1 =
\begin{bmatrix}
0.5\\
-0.5
\end{bmatrix}
$$

and:

$$
x =
\begin{bmatrix}
1
\end{bmatrix}
$$

Then:

$$
W_1x
=
\begin{bmatrix}
0.5\\
-0.5
\end{bmatrix}
\begin{bmatrix}
1
\end{bmatrix}
$$

giving:

$$
\begin{bmatrix}
0.5\\
-0.5
\end{bmatrix}
$$

Then add biases:

$$
z_h =
W_1x+b_1
$$

and apply sigmoid element-by-element:

$$
h =
\sigma(z_h)
$$

giving:

$$
h =
\begin{bmatrix}
0.6225\\
0.3775
\end{bmatrix}
$$

---

# 8. Why I want to stop here for a moment

You originally asked about:

> `x = 1×3` and `W1 = 2×3`

This tiny network is the **perfect bridge** to that.

Right now:

```text
1 input
↓
2 neurons
```

So:

```text
x  = 1 input
W1 = 2 weights
```

When we move to:

```text
3 inputs
↓
2 hidden neurons
```

we'll have:

```text
x = [x1, x2, x3]
```

and each of the two neurons needs **3 weights**:

```text
             x1
             x2
             x3
              │
       ┌──────┴──────┐
       ▼             ▼
      h1             h2
```

Therefore:

```text
W1 =
[
    [w11, w12, w13],
    [w21, w22, w23]
]
```

which is:

```text
2 × 3
```

And **now the matrix multiplication question you had earlier will make intuitive sense instead of being a shape rule you just memorize.**

---

## Next step: backprop through BOTH hidden neurons

The interesting part is that we'll discover something new:

```text
dL/dh1
```

and

```text
dL/dh2
```

are different.

Then we'll calculate:

```text
dL/dw2_1
dL/dw2_2
dL/dw1_1
dL/dw1_2
```

and finally put the whole thing into a training loop.

That will be our last scalar-style implementation before we turn the exact same network into **NumPy matrix operations**.
