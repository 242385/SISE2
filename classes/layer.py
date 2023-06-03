from enum import Enum
import numpy as np
import random
from classes.neuron import *

class layerType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class Layer:
    def __init__(self, num_neurons, bias):

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def forwardPropagation(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

def sigmoid(v):
    outputVector = list()
    for x in v:
        s = 1 / (1 + np.exp(-x))
        outputVector.append(s)
    return outputVector
