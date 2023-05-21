from enum import Enum
import numpy as np

class layerType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class Layer:
    def __init__(self, neurons, layertype):
        self.neurons = neurons
        self.type = layertype
        self.A = np.array([neuron.weights for neuron in self.neurons])
        self.x = np.array([neuron.value for neuron in self.neurons])
        self.y = self.A @ self.x
