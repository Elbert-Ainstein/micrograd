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

We want a nice way to visualize the stuff we are building out now, just to visually see things, when it outputs things I recommend using jupyter notebook to see it:
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


**Math time**
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
