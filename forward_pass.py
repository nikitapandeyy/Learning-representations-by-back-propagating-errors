import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
x=np.array([1.0,0.0,1.0])
W1=np.array([
    [0.5,-1.0,0.8],
    [-0.2,0.3,0.4]
])
b1=np.array([0.1,-0.1])

z1=W1 @ x+b1
h=sigmoid(z1)
print("Hidden pre-activation:", z1)
print("Hidden activation:", h)
# Then the output layer uses the hidden representation
w2=np.array([
    [1.0,-1.0]
])
b2=np.array([0.2])

z2=w2 @ h + b2 
y=sigmoid(z2)
print("Output pre-activation:", z2)
print("Output:", y)













