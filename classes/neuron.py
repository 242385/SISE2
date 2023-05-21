import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return np.interp(s, [0, 1], [-1, 1])

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights              # Weights on connection between this neuron and the neurons from PREVIOUS layer
        self.bias = bias

    def calculateNeuronOutput(self, inputs):
        result = np.dot(self.weights, inputs)
        return self.activationFunction(sigmoid, result)

    def activationFunction(self, function, value):
        x = function(value + self.bias)
        return x
