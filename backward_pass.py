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

