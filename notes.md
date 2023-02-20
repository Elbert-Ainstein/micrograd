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

### other notes 
- No actual math in Neural Network codes
- Just to know what derivative is measuring, aka. instantaneous slope

![derivative measure](derivative.png)

### the code section

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

![norepr](noRepr.png)

**However, doing operations on Value objects is not doable yet, like adding Value(a) with Value(b)**

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
```
Now we know the children of every single value, but don't know what operation created it. Therefore needed another element _op:
```python
class Value:

    def __init__(self, data, _children=(), _op="):
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
