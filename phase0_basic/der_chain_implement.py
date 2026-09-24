def f(x):
    return x**2
def derivative(x):
    return 2*x
x=3
print("f(x)=",f(x))
print("derivative=",derivative(x))


#implemnt the derivate and the chain rule 
#model 
x=2
w=4
ya=10. #actual y value
#yc=w*x  #calculate the y value

yc=w*x
#loss func = dl/dw
#loss calculation
loss=(yc-ya)**2

#dl/dw=dl/dy * dy/dw
gradient=2*(yc-ya)*x
print("Prediction:", yc)
print("Loss:", loss)
print("Gradient:", gradient)
def loss_fn(w):
    return (w * x - ya) ** 2

h = 1e-5
numeric = (loss_fn(w + h) - loss_fn(w - h)) / (2 * h)
print("Analytic:", gradient)
print("Numeric: ", numeric)   # both should be ≈ -8


x=2
ya=10.0
w=4.0
lr=0.3
for step in range(10):
    yc=w*x
    loss=(yc-ya)**2
    grad=2*(yc-ya)*x
    w=w-lr*grad
    print(f"step {step}: w={w:.4f}  loss={loss:.6f}  grad={grad:.4f}")






