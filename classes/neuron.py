import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights              # Weights on connection between this neuron and the neurons from PREVIOUS layer
        self.bias = bias

    def calculateNeuronOutput(self, inputs):
        return np.dot(self.weights, inputs) + self.bias
