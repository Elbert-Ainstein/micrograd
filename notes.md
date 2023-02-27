# Notes made for the [video](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ), hopefully it helps you understand

Credit: Andrej Karpathy
His github [here](https://github.com/karpathy)

### Autograd engine:
- -> Automatic Gradient
- Implements backpropagation:
    - iteratively tune weights of neural network to minimize loss function
    - improves accuracy of network
- Neural networks are just mathematical functions & calculations

### About Micrograd
- micrograd is technically all you need to train networks
- However, it isn't really much of complex coding ( basic python + a bit of calculus kekw )
- All you need to understand neural networks; Everything else is efficiency
- Fundamentals
- In this article, all data inputs will be numbers.

### Other
- No actual math in Neural Network codes
- Just to know what things are measuring, aka. derivative for instantaneous slope and impact on 
- Jupyter notebook in github to let you check out the product.
- Andrej's jupyter notebook is [here](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/). It's more completed and detailed.

![derivative measure](/images/derivative.png)

### the code section + Processes and more notes

This is the skeleton of the Value class; Keeps track of a single data value.

```python
class Value:
    
    def __init__(self, data):
        self.data = data
    
    def __repr__(self):
        return f"Value(data={self.data})"
a = Value(2.0)
print(a)
```

Therefore running it with 

```sh
python engine.py
```

gets us the answer of Value(data=2.0).
The __repr__ method is making python printing the results in a nicer way, unless you want torture responses like this:

![norepr](/images/noRepr.png)

### However, doing operations on Value objects is not doable yet, like adding Value(a) with Value(b)

So we need to add a way to do operations

```python
class Value:
    
    def __init__(self, data):
        self.data = data
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        output = Value(self.data + other.data)
        return output

a = Value(2.0)
b = Value(-3.0)
print(a+b)

```
In here, python will perform this:
```python
a.__add__(b)
```

this giving us a new Value object with the new number: Value(data=-1.0)

Same thing for other operations, in this case implementing multiplication:

```python
class Value:
    
    def __init__(self, data):
        self.data = data
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        output = Value(self.data + other.data)
        return output

    def __mul__(self, other):
        output = Value(self.data * other.data)
        return output

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
# (a__mul__(b)).__add__(c)
print(d)
```

Now the only thing that is missing is the connective tissue of the expression
as we need to keep the expression graphs. What we need is pointers that shows what Values makes what other Values. In this case implementing a new variable called _children:

```python
class Value:
    # by default __children will be an empty tuple
    def __init__(self, data, _children=()):
        self.data = data
        # for efficiency, it'll be in a set
        self._prev = set(_children)
    
    def __repr__(self):
        return f"Value(data={self.data})"
    # when we do the operations, we are feeding in the children of the value
    def __add__(self, other):
        output = Value(self.data + other.data, (self, other))
        return output

    def __mul__(self, other):
        output = Value(self.data * other.data, (self, other))
        return output

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
# _children will be empty, and _prev will be an empty set
print(d._prev)
# {Value(data=-6.0), Value(data=10.0)}
# (a*b) + c
```

Now we know the children of every single value, but don't know what operation created it. Therefore needed another element _op:

```python
class Value:

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self):
        return f"Value(data={self.data})"

    #Telling us which operation made the children
    def __add__(self, other):
        output = Value(self.data + other.data, (self, other), '+')
        return output

    def __mul__(self, other):
        output = Value(self.data * other.data, (self, other), '*')
        return output

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
print(d._op)
#'+', caused by addition of (a*b) + c
```

***Have full mathematical expression & building data structure, and now we know how each value is made, by what expression and from what other Values***

We want a nice way to visualize the stuff we are building out now, just to visually see things, when it outputs graphs I recommend using [jupyter notebook](https://jupyter.org/) to see the visualizations:

```python
from graphviz import Digraph

def trace(root):
    #builds a set of node & edges in a graph
    nodes, edges = set(), set()
    def build(v):
        nodes.add(v)
        for child in v._prev:
            edges.add((child, v))
            build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr = {'rankdir': 'LR'}) #LR = Left -> Right

    nodes, edges = trace(root) # building a graph of nodes and edges
    for n in nodes:
        uid = str(id(n))
        # make a rectangular "record" node for any value in the graph
        dot.node(name = uid, label = "{ %s | data %.4f }" % (n.label, n.data), shape='record')
        if n._op:
            # if this value is a result of an operation make an _op node
            dot.node(name = uid + n._op, label = n._op)
            # connect the op node
            dot.edge(uid + n._op, uid)
    
    for x, y in edges:
        # connect x to the op node of y
        dot.edge(str(id(x)), str(id(y)) + y._op)
    
    return dot
```

Of course, labels are needed:

```python
class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        output = Value(self.data + other.data, (self, other), '+')
        return output

    def __mul__(self, other):
        output = Value(self.data * other.data, (self, other), '*')
        return output

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label='e'
d = e + c; d.label='d'
f = Value(-2.0, label='f')
L = d * f; L.label='L'
# L = -8.0
```

# Time for some backpropagation action. Example #1: Simple expressions
We run back from L and calculate the gradient along those intermediate values.
- Compute the derivative of the node *dL* with respect to *L*
- *dL/dL* = 1
- Derivative of L with respect to *f*, respect to *d*, etc.
- Derivative of output with respect to leaf nodes, which will eventually be weights of a neural network, whilst other leaf nodes will be data, which is not oftenly used

In neural networks, we are interested for the derivative of the loss function with respect to the weights of a NN. (*I'll be calling neural networks NN from now on*)

- Cannot really go for derivative of L with respect to data cause data is fixed
- However weights will be iterated using the gradient info
### Now it is time to make a variable that maintains the derivative of L with respect to the weights, right now with respect to the values, and the name of the variable in this case is going to be called grad 

```python
class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 # 0 means no effect, more explanation in the slanted text below.
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        output = Value(self.data + other.data, (self, other), '+')
        return output

    def __mul__(self, other):
        output = Value(self.data * other.data, (self, other), '*')
        return output

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label='e'
d = e + c; d.label='d'
f = Value(-2.0, label='f')
L = d * f; L.label='L'
```

*The grad is initially 0, which means in initialization we are assuming values does not affect the output. If grad = 0, the change of this variable does not change the loss function.*
Now we can change the graphing code a bit:

```python
from graphviz import Digraph

def trace(root):
    #builds a set of node & edges in a graph
    nodes, edges = set(), set()
    def build(v):
        nodes.add(v)
        for child in v._prev:
            edges.add((child, v))
            build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr = {'rankdir': 'LR'}) 

    nodes, edges = trace(root) 
    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record') # changing it here to show grad
        if n._op:
            dot.node(name = uid + n._op, label = n._op)
            dot.edge(uid + n._op, uid)
    
    for x, y in edges:
        dot.edge(str(id(x)), str(id(y)) + y._op)
    
    return dot
```

What the engine.py should look like right now:

```python
from graphviz import Digraph

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 # 0 means no effect, more explanation in the slanted text below.
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        output = Value(self.data + other.data, (self, other), '+')
        return output

    def __mul__(self, other):
        output = Value(self.data * other.data, (self, other), '*')
        return output

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label='e'
d = e + c; d.label='d'
f = Value(-2.0, label='f')
L = d * f; L.label='L'

def trace(root):
    #builds a set of node & edges in a graph
    nodes, edges = set(), set()
    def build(v):
        nodes.add(v)
        for child in v._prev:
            edges.add((child, v))
            build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr = {'rankdir': 'LR'}) 

    nodes, edges = trace(root) 
    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record') # changing it here to show grad
        if n._op:
            dot.node(name = uid + n._op, label = n._op)
            dot.edge(uid + n._op, uid)
    
    for x, y in edges:
        dot.edge(str(id(x)), str(id(y)) + y._op)
    
    return dot
```

### Time to work with the gradient.
- For example what is the derivative of *L* with respect to L?
- Well derivative of *L* with respect to *L* is 1.
- so what if wecalculate things like *dL/dd* and *dL/df*

if 

```python
L = d * f
```

,

```python
dL/dd = ?
```

Well the answer is *f*, if you don't understand why you can do some derivative revision.

The value of *d* is obvious, which is -2, 
this also makes the grad for variable *f* to be equal to the value of *d*, which is 4.
```diff
+ f.grad = 4.0
+ d.grad = -2.0
```
Here is the code for the stuff above:

```python
def weirdCalcs():
    h = 0.001

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label='e'
    d = e + c; d.label='d'
    f = Value(-2.0, label='f')
    L = d * f; L.label='L'
    L1 = L.data

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label='e'
    d = e + c; d.label='d'
    d.data += h
    f = Value(-2.0, label='f')
    L = d * f; L.label='L'
    L2 = L.data

    print((L2 - L1)/h)

weirdCalcs()
# the result should be -2.000000000000668
```
What we are doing here is more like a inline gradient check, which is getting the derivative with respect to all intermediate results, in numerical gradients here it is just estimating with small step sizes.

## **THE CRUX OF BACKPROPAGATION, READ THIS**
- now we are doing *dL/dc* & *dL/de*; derivative of *L* with respect to *c* and derivative of *L* with respect to *e*.
- reasons being we have already calculated the other gradients already with values *d* & *f*

### Now the question is how is L sensitive to *c*, in other words, how does the change of *c* impact L throught variable *d*
> *d* = *e* + *c*, and *L* = *d* * *f*

- First thing to consider: what is *dd/dc* ?
- Since we know *d* = *e* + *c*, this makes the differentiation of *c* + *e* with respect to *c* gives 1.0
- By symmetry, we now also know *dd/de* = 1.0 as well

### So now we understand how *c* and *e* affect *d*, and how *d* impact *L*, how do we write *dL/dc*? The answer is ***Chain rule*** 

![chain-rule](/images/chainRule_1.png)

In this case, we are using this notation instead:

![chain-rule2](/images/chainRule_2.png)

- For example, if *dz/dy* = 2, and *dy/dx* = 4, then *dz/dx* = 8

- This makes *dL/dc* = *dL/dd* * *dd/dc*
- *dL/dc* = -2.0
- *c*.grad = -2.0
- *e*.grad = -2.0

**So now we can recurse back further, applying Chain rule**

- *dL/de* = -2
- *e* = *a* * *b*
- *de/da* = *b* = -3.0
- *dL/da* = *dL/de* * *de/da* = 6
- *a*.grad = *e*.grad * *b* = -2.0 * -3.0 = 6
- *b*.grad = *e*.grad * *a* = -2.0 * 2.0 = -4

### Essentially, we iterated throught the nodes and locally applied the chain rule. We know what the effect *L* has on the output, and we look at how the output was produced. We go on and multiply the local derivatives.

## **That is basically backpropagation, a recursive application of chain rule backwards in a computational graph**

# Backpropagation example #2: Neurons
Typical Neuron that we will be following basically:
![neuron](/images/neuron.jpeg)

```python
# inputs x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1, w2, which are synaptic strength 
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.7, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
# n is the cell body without the activation for now, it'll be added in the future
n = x1w1x2w2 + b; n.label = n

draw_dot(n)
```

This will generate an output that looks like this:
![x1w1](/images/x1w1.png)

Now I will add the activation function in:

```python
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.7, label='b')
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
# activation function
o = n.tanh(); o.label = 'o'

draw_dot(n)
```

! Our current value class does not support the creation of a hyperbolic tangent function, because we only implemented add and multiply. We need to add exponents, subtraction, and division. However, we don't neccessarily need to do that, as the only thing we need to take care of is to know the local derivative. Therefore, although breaking it down into pieces is doable, I will implement tanh straightforwardly.

```python
class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        output = Value(self.data + other.data, (self, other), '+')
        return output

    def __mul__(self, other):
        output = Value(self.data * other.data, (self, other), '*')
        return output

    #the tanh function
    def tanh(self):
        x = self.data
        tanh = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        # One child, which is wrapped in a tuple
        out = Value(tanh, (self, ), 'tanh')
        return out
    
    #other codes

    draw_dot(o)
```

This should produce the following result of your data is the exactly the same as mine. The tanh function is now pretty much a micrograd supported node as an operation:

![tanh](/images/tanh.png)
Now, as long as we know the derivative of tanh, we can backpropagate through it.

Since the tanh function graph looks like this

![tanhfunc](/images/tanhfunc.png)
We can see that the bigger the input the more the function squashes it. For example if *b* = 8, the input into the tanh function would be 2, which will output 0.96 in the function.

### More backpropagation

What we are trying to do now is to determine *do/dx1*, *do/dx2*, *do/dw1*, *do/dw2*.

Of course, in a NN setting, we care more about the weights *w1* and *w2*, and *do/dw1* and *do/dw2* because they are the things we are changing in optimization.
**Just for reminder, this is a single neuron, so eventually in the big puzzle there is going to be a loss function determining the accuracy of the NN, and we backpropagate with respect to that accuracy, while trying to increase it.**


### Math time
- I will set *b*.data = 6.8813735870195432 for good local derivative outputs. You can use any number.
- *do/do* = 1; o.grad = 1

Right now, the entire graph looks like this:

![graph#1](/images/graph_1.png)

To backpropagate through tanh, we need to know the local derivative of tanh.
- *o* = tanh(*n*)
- *do/dn* = ?
- Since in calculus d/dx tanh(x) = 1 - tanh(*x*)**2,
- *do/dn* = 1 - tanh(*n*)**2 (the ** notation means ^, so 2^3 = 8, and 2**3 = 8, same thing.)
- So what this means is *do/dn* = 1 - *o***2 = 0.5
- *n*.grad = 0.5
- As we can see, the nodes next are *x1w1+x2w2* & *b*
- Since *n* = *x1w1+x2w2* + *b*, the differentiation of *x1w1+x2w2* + *b* with respect to *b* is 1.0, so *x1w1+x2w2*'s local derivative = 1, and by symmetry we know *b*'s local derivative is also 1
- Applying Chain rule, *x1w1+x2w2*.grad = 0.5, *b*.grad = 0.5
- Tracing back, the node is a plus node, so *x1w1*.grad = 0.5, and *x2w2*.grad = 0.5
- Going back even more, we can see that:
    - x1.grad = w1.data * x1w1.grad = -1.5
    - w1.grad = x1.data * x1w1.grad = 1.0
    - x2.grad = w2.data * x2w2.grad = 0.5
    - w2.grad = x2.data * x2w2.grad = 0.0

Let's think about why w2.grad = 0 in a more intuitive way. The derivative always tells us the children's influence (in this case w2) on the final result. And so if we change w2.data, it will not have an impact on the output, as we are multiplying by 0. No changing means no derivative, which means that w2.grad = 0

The graph now updates to this:

![graph#2](/images/graph_2.png)


# Automatically doing backpropagation
Ok so let's end the suffering and implement this so that it runs more automatically. Now we understand how addition and multiplication determines the gradient of the variables, and implement these concepts. 

The first step we need is to implement a _backward function to the init function that operates the chain rule at each node and chain up the outputs to the next nodes' inputs

```python
class Value:
  
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
        
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad = (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
```

Now we can just call _backward() in order. But since grad is initialized to 0, make o.grad 1 first
- o._backward() # n.grad becomes 0.5
- n._backward() # b.grad = 0.5, x1w1x2w2.grad = 0.5
- b._backward() is the empty function because it doesn't have leaf nodes
- x1w1x2w2._backward() # x1w1 = 0.5, x2w2 = 0.5
- x1w1._backward() # x1.grad = -1.5, w1.grad = 1.0
- x2w2._backward() # x2.grad = 0.5, w2.grad = 0.0

Let's recap a briefly to clear up our minds: We have written a math expression, and we are trying to go backwards through the expression.

**Now let's get rid of calling _backward() manually.**

We have to start from the last node, as you cannot perform backpropagation on a node before the nodes after it. It is like how all the leaves on the tree depends on the trunk, and their own respective branches.

This ordering of graphs could be achieved by something called topological sorting. What topological sort does is that node *v* never gets visited until everything that it depends on gets visited. All the edges goes from left to right.

![topograph](/images/topograph.png)

example code for topological sorting in the context of our NN setting:

```python
o.grad = 1 # Base case

topo = []
visited = set()
def build_topo(v):
    # If V is not visited, we mark it as visited
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            # Iterates through its children nodes and builds topological sort in them
            build_topo(child)
        # after all its children has been processed it will append to the node list
        # guarantee that it will be in the list after all its children have been processed
        topo.append(v)

# Topological sort starts at o.
build_topo(o)

# Builds the gradients
for node in reversed(topo):
    node._backward()
```

Now let's implement this into the Value class.

Updated Value class:

```python
class Value:
  
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
        
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad = (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
        build_topo(self)

        self.grad = 1
        for node in reversed(topo):
            node._backward()

    #other codes

o.backward()
```

# Debug

Ok you might see a bug right here:
For example if we have this piece of code here

```python
a = Value(3.0, label='a')
b = a + a; b.label='b'
b.backward()
draw_dot(b)
```

This will output this:

![grad](/images/grad.png)

There is two a nodes on top of each other, and the gradient is wrong here. Doing calculus, we know that *db/da* = 2

Intuitively, *b* = *a* + *a* and we called backward(). In the addition function in the Value class, we can see that

```python
self.grad = 1.0 * out.grad
other.grad = 1.0 * out.grad
```

But since self & other are the same object, we can see that the grad gets reset in self.grad **and** other.grad. Meaning that grad got reset twice.

grad = 1.0 * out.grad => 1.0 => 1.0 * out.grad
so a.grad got reset to 1.0 twice.

Let's do a second example:

```python
a = Value(-2.0, label='a')
b = Value(3.0, label='b')
d = a * b    ; d.label = 'd'
e = a + b    ; e.label = 'e'
f = d * e    ; f.label = 'f'

f.backward()

draw_dot(f)
```

![grad_2](/images/grad_2.png)

What happens is that if a variable is used more than once, there will be an error with the gradients. The calculated gradients for nodes leading to *e* & *d* will overwrite each other. The solution to this is to use the multivariable variation of [chain rule](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/differentiating-vector-valued-functions/a/multivariable-chain-rule-simple-version). Essentially, what happens is that we need to accumulate the gradients. 

Updated Value class:

```python
class Value:
  
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # The += operation is OK because we initialize grad with 0.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
        build_topo(self)

        self.grad = 1
        for node in reversed(topo):
            node._backward()
```

Now the gradient is correct:

![grad_correct](/images/grad_correct.png)

![grad_correct_2](/images/grad_correct_2.png)

*e*.backward() and *d*.backward() will deposit their individual gradients, and will add on top of each other.

## If you have been questioning this article why am I not breaking down tanh, I am going to do it now to do some lightweighted excercising on operations.

![tanhfunction](/images/tanhfunction.png)

```python
tanh = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
```

What we did with tanh above was making it a single function as we know its derivative and can backpropagate through it. However, we can also break down tanh as it also forces us to implement more operations, and to prove that it gives the same gradients :D

Some things we need to clean up first though. Remember that when we do addition,  we did 

```python
self.data + other.data
```

This makes it so that if we want to do something like *a* = Value(2.0) and *a* + 1,
python is just going to slap an error in your face saying that 1 has no attribute ***"data"***. This makes it that adding only allows to add Value objects with each other. For convenience of breaking up tanh(), we can add this:

```python
def __add__(self, other):
    # If other is a Value object, then just use it as it is, but if other is not a Value object we will just assume it as a integer or a float value, and put it in a new Value instance with other as its data.
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
        
    def _backward():
        self.grad += 1.0 * out.grad
        other.grad += 1.0 * out.grad
    out._backward = _backward
        
    return out
```

Do the same thing for the other operations we have right now, aka the multiplication and division and subtraction operation. However, just putting in the if statement does not work by itself, because python cannot do something like 2 * *a* as it will just call 2 * *a* and not **a.__mul__(2)**, as there is not a thing like **2.__mul__(a)**. Therefore we can make a fallback function, to check if *a* can multiply 2 instead of trying to do the original 2 * *a*. Let's do the same thing for addition as well.

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
        
    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward
    
    return out
def __rmul__(self, other): # other * self
    return self * other

def __radd__(self, other): # other + self
    return self + other
```

Looking at the tanh function, we should still put in the operations for exp and division. 

Let's do exponents first.

```python
def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')

    def _backward():
        # you can kinda think about it here
        self.grad += #??? 
    out.grad = _backward

    return out
```

Answer:

Basically, we need to know what ther local derivative is. 

***d/dx  e^x = e^x***

```python
def exp(self):
    x = self.data
    # e^x
    out = Value(math.exp(x), (self, ), 'exp')

    def _backward():
        self.grad += out.data * out.grad
    out.grad = _backward
 
    return out
```


Divide function. Technically, if we do *a*/*b*, it is the same as *a* * (*b*^-1), and this makes the operation more general and easy to work with. We can redefine division as 

```python
def __truediv__(self, other): # self/other
    return self * other**-1
```

Well we need a function to also be called when a value is raised to a power too right? In it *other* will be the power.

```python
def __pow__(self, other):
    # making sure other = int or a float
    assert isinstance(other, (int, float)), "only supports int/float powers for now"
    out = Value(self.data**other, (self, ), f'**{other}')

    def _backward():
        self.grad += other * (self.data**(other - 1)) * out.grad
    out._backward = _backward

    return out
```

The last thing to do is to subtract. The way we do it is to implement a negation, and then subtracting the negation.

```python
# the negation
def __neg__(self, other):
    return self * -1

# Subtraction
def __sub__(self, other):
    return self + (-other)
```

Ok let's change how we define *o* below to match the broken up tanh function.

```python
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.8813735870195432, label='b')
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'

# --------------------------------------
# e^2x-
e = ( 2 * n ).exp(); e.label = 'e'
o = ( e - 1 ) / ( e + 1 )
# ---------------------------------------
o.label='o'
```

Updated code:

```python
class Value:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

def __repr__(self):
    return f"Value(data={self.data})"
  
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    
    def _backward():
        self.grad += 1.0 * out.grad
        other.grad += 1.0 * out.grad
    out._backward = _backward
    
    return out

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    
    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward
      
    return out
  
  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward

    return out
    
def __rmul__(self, other): # other * self
    return self * other

def __truediv__(self, other): # self / other
    return self * other**-1

def __neg__(self): # -self
    return self * -1

def __sub__(self, other): # self - other
    return self + (-other)

def __radd__(self, other): # other + self
    return self + other

def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')
    
    def _backward():
        self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    
    return out
  
def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')
    
    def _backward():
        self.grad += out.data * out.grad 
    out._backward = _backward
    
    return out
  
  
def backward(self):
    
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
        node._backward()
```

Rerunning should provide these results since the expressions have been elongated.

![elong_1](/images/elong_1.png)
![elong_2](/images/elong_2.png)

If we compare the results, the forward & backward passes should be working just fine. In some way, as long as the forward & backward passes are accurate, it doesn't really matter what operation it is.

# Comparison to [Pytorch](https://pytorch.org/get-started)
```python
# pip install pytorch
import torch

x1 = torch.Tensor([2.0]).double()                ; x1.requires_grad = True
x2 = torch.Tensor([0.0]).double()                ; x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double()               ; w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()                ; w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double()  ; b.requires_grad = True
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward() # pytorch has backward() like how we implemented

print('---')
print('x2', x2.grad.item()) # gradients
print('w2', w2.grad.item())
print('x1', x1.grad.item())
print('w1', w1.grad.item())
```

In micrograd, the engine only has scalar values because it is a scalar valued engine. However, in PyTorch everything is based around tensors, which are ***n-dimensional*** arrays of ***scalars***. In here we only implemented one element into the tensor, but in normal situations we would be working with tensors that look a bit more closer to this:

```python
torch.Tensor([[1, 2, 3], [4, 5, 6]])
```

That tensor above is a 2x3 array or scalars in a single representation. Running that returns its shape, which is ***torch.Size([2, 3])***, which is its dimensions. The reason of using .double is to keep the calculations the same as the Python lang, as it uses double precision, whilst PyTorch uses single precision. Also, since those are leaf nodes, by default PyTorch assumes they don't need gradients, so I setted it to true as well to explicitly say that we need gradients for those leaf node inputs.

After defining values, we can perform arithmetic, and in PyTorch they have .data and .grad attributes like micrograd. The difference is .item() which is to strip the element from its Tensor like picking out a soldier from his battalion. 

(o.item & o.data.item() produces the same result)

The results of running is this:

![pytorchResult](/images/pytorchResult.png)

Essentially, PyTorch can do just what we did, but on a special case where tensors are just single-element tensors. However, PyTorch is just way more efficient as they work with tensors that can run parallel with each other.

# Building a neural net library (aka multi-layer perceptron) in Micrograd

First we are going to build a neural net, and then build out two layers of multi-layer perceptron. 

We are going to make a neuron, but it will subscribe to PyTorch's API and how they made their models. Just like how we can match Pytorch to the autograd part, we can also do that on neural networks.

There will be a lots of notes in the code below. I apologize but I cannot think of a way to put them outside of the code while explaining.
```python
class Neuron:
    # How many inputs come to a neuron
    def __init__(self, numOfInputs):
        # Create a weight that is between -1 & 1
        self.w = [Value(random.uniform(-1, 1)) for _ in range(numOfInputs)]
        # Create a bias that is between -1 & 1 that controls the trigger happiness of the neuron
        self.b = Value(random.uniform(-1, 1))

    # Different value each time a neuron is called
    def __call__(self, x):
        # w * x + b
        # w * x is a dot product
        
        # What we can do with __call__ is 
        # if x = [2.0, 3.0]
        # n = Neuron(2)
        # we can call n(x) for an output
        # Using the n(x) notation will call n.__call__()

        # Now we can do the forward pass of the neuron
        
        # We pair up elements of w and x and multiply them

        # Creates an iterator over two iterators to read tuples of corresponding values of w and x 

        # w: [Value(data=1), Value(data=0.5), Value(data=0.3)]
        # x: [1, 2, 3]
        # the zip function: [(Value(data=1), 1), (Value(data=0.5), 2), (Value(data=0.3), 3)]

        # zip(self.w, x)
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)

        # Pass that through a linearity
        out = act.tanh()
        return out
```

Now let's make a layer of neurons. The neurons are not connected to each other if they are in the same layer, but they are fully connected to the layer prior to it, and evaluates independently.
It looks like this:

![layer](/images/layer.jpeg)

```python
class Layer:
    def __init__(self, numOfInputs, numOfOutputs):
        # initialize neurons with the amount of outputs we want.
        self.neurons = [Neuron(numOfInputs) for _ in range(numOfOutputs)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs
```

Lets complete the picture and the MLP (multi-layer perceptron).

As we can see in the pic, the outputs of the layer gets fully fed into the second layer.

```python
class MLP:
    # instead of saying how many neurons in a single layer ( numOfOutputs ), we are taking a list of outputs and the list defines the sizes of layers in the MLP.
    def __init__(self, numOfInputs, numOfOutputs):
        size = [numOfInputs] + numOfOutputs
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(numOfOutputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

Well, just to follow the picture, we can put in three input neurons, and two four-neuron hidden layers and a output unit.

```python
x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
n(x)
```

Well to make it a bit nicer and aesthetic, lets just return the outs[0] in the Layers class if it is only a single thing to output.

```python
class Layer:
    def __init__(self, numOfInputs, numOfOutputs):
        self.neurons = [Neuron(numOfInputs) for _ in range(numOfOutputs)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
```

You can run draw_dot(*n*(*x*)) if you want to too. You can see that the network gets upscaled a good amount haha :D

## Updated code:
```python
class Neuron:
    def __init__(self, numOfInputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(numOfInputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = activation.tanh()
        return out

class Layer:
    def __init__(self, numOfInputs, numOfOutputs):
        self.neurons = [Neuron(numOfInputs) for _ in range(numOfOutputs)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class MLP:
    def __init__(self, numOfInputs, numOfOutputs):
        size = [numOfInputs] + numOfOutputs
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(numOfOutputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

# Loss function, Desired outputs, Neural Nets, and Collection of Parameters

Let's make a small dataset first.
```python
# 4 possible inputs
xs = [
  [2.0, 3.0, -1.0], # desire 1.0
  [3.0, -1.0, 0.5], # desire -1.0
  [0.5, 1.0, 1.0],  # desire -1.0
  [1.0, 1.0, -1.0], # desire 1.0
]
ys = [1.0, -1.0, -1.0, 1.0] # 4 desired targets

# get the predictions
ypred = [n(x) for x in xs]
```

Here is my output; Everyone should have different outputs because we haven't written any improvement functions aka loss functions

![dataset_inacc_1](/images/dataset_1_inacc.png)

As you can see, the first output I want to push it up, the second one I want it to push it down, third to go down, and fourth to go up.

We need to find a number to measure the total performance of the NN and how well it is performing, which is the loss. As we can see, we are not performing well, being very off to the target number, making the loss high.

What we are going to do is to implement a loss function called the [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) loss. 

```python
[(yOutput - yGroundTruth)**2 for yGroundTruth, yOutput in zip(ys, ypred)]

loss = sum([(yOutput - yGroundTruth)**2 for yGroundTruth, yOutput in zip(ys, ypred)])
```

We pair up the ground truths with the predictions and the zip iterates through the tuples. For each one of the four outputs we are taking the prediction and the ground truth (in the above is the array ys) and we are subtracting them and squaring them. The square is to ensure that we always get a positive number. Loss is the sum of all the examples of yOutput - yGroundTruth. Also, only if the output and ground truth are equal, aka your prediction hits the jackpot, loss will be zero. The more off target we are, the more loss there will be. 

### Now let's minimize the loss

Ok. This is the fun part. Try running 
```python
loss.backward()
# first layer's first neuron's weights' grad
n.layers[0].neurons[0].w[0].grad
```

This gave the following result for me:

![loss_backward](/images/loss_backward.png)

the loss.backward() gave the particular neuron's weight's grad a value. Since for this one the value is positive, that means the influence that it will give to the weight is positive, pushing the output up, and decreasing the loss. This info will also go towards every neuron and their parameters. 

You can now run 
```python
draw_dot(loss)
``` 
and see for yourselves.

## Now we need some code to collect the parameters of the neural network

The purpose of this is to allow us to operate on all of them simultaneously. Every one of them we will nudge them a bit based on the gradient.

***Note: PyTorch also has parameters on every nn Module. It does what we are doing right now, it just returns the parameter scalars. [Link](https://pytorch.org/docs/stable/nn.html)***

Code here:

```python
class Neuron:
    def __init__(self, numOfInputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(numOfInputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = activation.tanh()
        return out
    
    def parameters(self):
        # list + list => list
        return self.w + [self.b]
    
class Layer:
    def __init__(self, numOfInputs, numOfOutputs):
        self.neurons = [Neuron(numOfInputs) for _ in range(numOfOutputs)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        # The straightforward way 

        # params = []
        # for neuron in self.neurons:
        #     # params of neuron
        #     ps = neuron.parameters()
        #     # put them on top of the params list
        #     params.extend(ps)
        # return params

        # The more complex way

        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, numOfInputs, numOfOutputs):
        size = [numOfInputs] + numOfOutputs
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(numOfOutputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```
 
Running 

```python
n.parameters()
```

should give you the result of a list of Value(data), which are all the weights and biases in the NN.

Now, what we want to do is to change the data according to the gradient info. It basically is a tiny update in the gradient descent scheme. In gradient descent, we can imagine the gradient as a vector pointing towards the direction of increased loss. We are modifying p.data by a small step in the direction of the gradient. If the gradient is negative, we increase the data and vice versa, and it will decrease the loss. 

```python
for p in n.parameters():
    # -0.01 as step size, it is also called the learning rate
    p.data += -0.01 * p.grad
```

Before and after comparison:

![before](/images/before.png)
![after](/images/after.png)

Essentially:

Forward pass

![forward](/images/forward.png)

Backward pass

![backward](/images/backward.png)

Update

![update](/images/update.png)

And the NN will improve. Yeah. That's how to train the Neural Net. It isn't that intelligent, is it? >:)


### Well let's make a function to automate it

```python
for k in range(20):

    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum([(yOutput - yGroundTruth)**2 for yGroundTruth, yOutput in zip(ys, ypred)])

    # backward pass

    # We have to zero the grad like how it is in the constructor, otherwise things mess up, as derivatives don't accumulate correctly.
    for p in n.parameters():
        p = 0.0
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(k, loss.data)
```


# SUMMARY

Neural networks are math expressions. Fairly simple ones, at least for multi-layer perceptrons, and they take input as data, and take in inputs, weights, and parameters of the NN. We use the forward pass followed by a loss function. The loss function measures the accuracy of the predictions, and the loss is low when predictions are matching targets/desired outputs. When the loss is low, the NN does what you want it to do in your problem. We use backpropragation to get the gradient, and we tune the params to decrease loss locally. This is repeated many times over gradient descent, which we follow the gradient info and it minimizes the loss. 

These neurons are fascinating. For example in GPT, we get loads of text from the internet, and we are trying to get a neural net to predict to take a few words and to predict the next word in the sequence. That kind of NN would have billions and trillions of params, not the 41 we have. However, fundamentally it runs on the same principles as micrograd.
