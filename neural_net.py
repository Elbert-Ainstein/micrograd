import engine
from engine import Value
import random

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

