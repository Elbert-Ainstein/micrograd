# Use this file for some playing around
from neural_net import Neuron, Layer, MLP

# 4 possible inputs
xs = [
  [2.0, 3.0, -1.0], # desire 1.0
  [3.0, -1.0, 0.5], # desire -1.0
  [0.5, 1.0, 1.0],  # desire -1.0
  [1.0, 1.0, -1.0], # desire 1.0
]
ys = [1.0, -1.0, -1.0, 1.0] # 4 desired targets

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
n(x)

for k in range(50):

    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum([(yOutput - yGroundTruth)**2 for yGroundTruth, yOutput in zip(ys, ypred)])

    # backward pass
    for p in n.parameters():
        p = 0.0
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.01 * p.grad

    print(k, loss.data)

print(ypred)