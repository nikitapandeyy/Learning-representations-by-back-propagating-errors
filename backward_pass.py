import numpy as np

##initial values
x=1.0
w1=0.5
w2=0.8
target=1.0
##activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

## forward pass 
z_hidden=w1*x
h=sigmoid(z_hidden)

##output neuron
z_oup=w2*h
y=sigmoid(z_oup)

##loss
loss=0.5*(target-y)**2
print("FORWARD PASS")
print("--------------------")

print("x =", x)

print("z_hidden =", z_hidden)
print("h =", h)

print("z_output =", z_oup)
print("y =", y)

print("target =", target)
print("loss =", loss)
# -------------------------
# BACKWARD PASS
# -------------------------

# 1. How does loss change with output?
dL_dy = y - target

# 2. How does output change with z_output?
dy_dz_output = y * (1 - y)

# 3. How does z_output change with w2?
dz_output_dw2 = h

# Gradient for w2
dL_dw2 = dL_dy * dy_dz_output * dz_output_dw2


# -------------------------
# Gradient for w1
# -------------------------

# How does z_output change with h?
dz_output_dh = w2

# How does h change with z_hidden?
dh_dz_hidden = h * (1 - h)

# How does z_hidden change with w1?
dz_hidden_dw1 = x

# Gradient for w1
dL_dw1 = (
    dL_dy
    * dy_dz_output
    * dz_output_dh
    * dh_dz_hidden
    * dz_hidden_dw1
)


print("\nBACKWARD PASS")
print("--------------------")

print("dL/dy =", dL_dy)
print("dy/dz_output =", dy_dz_output)

print("dz_output/dw2 =", dz_output_dw2)
print("dL/dw2 =", dL_dw2)

print("\ndz_output/dh =", dz_output_dh)
print("dh/dz_hidden =", dh_dz_hidden)
print("dz_hidden/dw1 =", dz_hidden_dw1)

print("dL/dw1 =", dL_dw1)

# -------------------------
# UPDATE WEIGHTS
# -------------------------

learning_rate = 0.1

old_w1 = w1
old_w2 = w2

w1 = w1 - learning_rate * dL_dw1
w2 = w2 - learning_rate * dL_dw2


print("\nWEIGHT UPDATE")
print("--------------------")

print("w1:", old_w1, "->", w1)
print("w2:", old_w2, "->", w2)