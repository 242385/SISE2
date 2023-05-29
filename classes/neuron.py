import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights              # Weights on connection between this neuron and the neurons from PREVIOUS layer
        self.bias = bias
        self.delta = float()
        if weights is not None:
            self.prev_weights = [0 for _ in range(len(weights))]
        self.prev_bias = 0

    def z(self, inputs):
        return np.dot(self.weights, inputs) + self.bias

    def a(self, inputs):
        #print(f"a {1 / (1 + np.exp(-self.z(inputs)))}")
        return 1 / (1 + np.exp(-self.z(inputs)))

    def C0(self, inputs, y):
        #print(f"C0 {0.5 * (self.a(inputs) - y)**2}")
        return 0.5 * (self.a(inputs) - y)**2

    def dC0da(self, inputs, y):
        #print(f"dC0da {self.a(inputs) - y}")
        return self.a(inputs) - y

    def dadz(self, inputs):
        s = 1 / (1 + np.exp(-self.z(inputs)))
        #print(f"dadz {s}")
        return s * (1 - s)

    def dzdw(self, inputsPreviousLayer):
        return self.a(inputsPreviousLayer)

