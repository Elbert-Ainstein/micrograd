# The micrograd

This is essentially the replica of [Andrej Karpathy](https://github.com/karpathy)'s Micrograd. However, I have added plenty of notes in [Notes.md](https://github.com/Elbert-Ainstein/micrograd/blob/main/notes.md) 

## The notes are for people who do not have enough time to complete the entire video but still want to know how everything works in detail. 

- Jupyter notebook in github to let you check out the product.
- Andrej's jupyter notebook is [here](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/). It's more completed and detailed.

Credits: **Andrej Karpathy**

Here are some small overviews:

The main classes in this code are Value, Neuron, Layer, and MLP.

The Value class represents a scalar value in the computation graph. It stores the value of the scalar, its gradient with respect to some output, its children nodes (i.e., the nodes that depend on it), the operation performed to produce the node, and a label for the node (which is only used for debugging).

The Neuron class represents a single neuron in a layer of the neural network. It takes a list of inputs and computes a weighted sum of those inputs, applies a bias term, and passes the result through a hyperbolic tangent activation function. The weights and bias of the neuron are represented as Value objects.

The Layer class represents a layer of neurons in the neural network. It takes a number of inputs and a number of outputs, and creates that many neurons with the same number of inputs. The output of the layer is a list of Value objects representing the output of each neuron.

The MLP class represents the entire neural network. It takes a number of inputs and a list of numbers representing the number of neurons in each layer. It creates a Layer object for each layer and chains them together. The output of the network is a Value object representing the output of the last layer.

The backward method of the Value class performs backpropagation on the computation graph starting at the current node. It uses a topological sort of the graph to ensure that nodes are visited in the correct order, and it computes the gradients of each node by propagating gradients backwards through the graph using the chain rule.

The parameters method of the Neuron and Layer classes returns a list of the Value objects representing the weights and biases of the neuron or layer. These can be used to update the weights and biases during training using techniques like stochastic gradient descent.

Overall, this code provides a basic implementation of a neural network using automatic differentiation, which can be used for both forward and backward propagation during training. However, the implementation is fairly limited in its capabilities and would need to be extended to support more complex models and optimization algorithms.
