from enum import Enum
import numpy as np


class layerType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class Layer:
    def __init__(self, neurons, inputs, layertype):
        self.neurons = neurons
        self.inputs = inputs
        self.type = layertype

    def A(self):
        return np.array([neuron.weights for neuron in self.neurons])

    def x(self):
        return self.inputs

    def y(self):
        return self.A() @ self.x()

    # Pass dataVector if input layer
    # Pass y if hidden layer
    def layerOutput(self):
        if self.type == layerType.INPUT:
            return self.inputs
        else:
            return sigmoid(self.y())

def sigmoid(v):
    outputVector = list()
    for x in v:
        s = 1 / (1 + np.exp(-x))
        s = np.interp(s, [0, 1], [-1, 1])
        outputVector.append(s)
    return outputVector
