model 
        y=x**2
derivative= if i change x a tiny amount how much does y changes.
dy/dx=2x

x=3
dy/dx=6(If x increases by 1 tiny unit around 3, y increases by about 6 units.)
But how does this become ML?

Imagine we're trying to predict house prices.

Our model is ridiculously simple:

$$ \hat y = wx $$

where:

x = house size
w = parameter the model learns
ŷ = prediction

Suppose:

x = 2
w = 3

Then:

$$ \hat y = 3(2)=6 $$

But the actual answer is:

y = 10

So our model is wrong.

We need a way to measure how wrong it is.

That's the loss function.
L=(y(cal)​−y(actual))**2

So:

$$ L=(6-10)^2=16 $$

Our goal in machine learning is essentially:

Change w so that the loss gets smaller.

he derivative tells us which direction to move

Our equations are:

$$ w \rightarrow \hat y \rightarrow L $$

More explicitly:

$$ w \rightarrow wx \rightarrow (wx-y)^2 $$

We want:

$$ \frac{dL}{dw} $$

But L doesn't directly look like a function of w.

This is exactly why we need the chain rule.

Now let's use the chain rule for our ML example

y^​=wx
L=(y^​−y)2

We want:

$$ \frac{dL}{dw} $$

There are three steps:

$$ w \rightarrow \hat y \rightarrow L $$

Therefore:

dl/dw=dL/dy(cal) * dy(cal)/dw

L=(y^​−y)2
dL/dy(cal)=2(y(cal)−y)
y^​=wx
dy^/dw=x

dL/dw=2(y^-y)x