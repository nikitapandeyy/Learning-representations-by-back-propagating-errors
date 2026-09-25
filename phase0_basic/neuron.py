import math
x=[1.0,2.0]
w=[0.5,-1.0]
b=10
z=sum(wi*xi for wi, xi in zip(w,x))+b
a=math.tanh(z)
print("z =", z, " a =", a)


import math

x = [1.0, 2.0]
w = [0.5, -1.0]
b = 0.5
y = 1.0

def loss_fn(w1):
    z = w1 * x[0] + w[1] * x[1] + b
    a = math.tanh(z)
    return (a - y) ** 2

h = 1e-5
numeric = (loss_fn(w[0] + h) - loss_fn(w[0] - h)) / (2 * h)
print("Numeric dL/dw1:", numeric)