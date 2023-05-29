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
        self.outputs = []  # initialize as empty, will be set in layerOutput()
        self.type = layertype

    def A(self):
        return np.array([neuron.weights for neuron in self.neurons])

    def x(self):
        return self.inputs

    def y(self):
        return np.dot(self.A(), self.x())

    def layerOutput(self):
        if self.type == layerType.INPUT:
            self.outputs = self.inputs
        else:
            self.outputs = sigmoid(self.y())
        return self.outputs


def sigmoid(v):
    outputVector = list()
    for x in v:
        s = 1 / (1 + np.exp(-x))
        outputVector.append(s)
    return outputVector
