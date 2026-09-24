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









