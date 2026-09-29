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