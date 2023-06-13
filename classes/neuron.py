import numpy as np


class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def a(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(self.z())
        return self.output

    def z(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    def sigmoid(self, total_net_input):
        return 1 / (1 + np.exp(-total_net_input))

    def dC0dz(self, target_output):
        return self.dC0da(target_output) * self.dSigmoid()

    def C0(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    def dC0da(self, target_output):
        return -(target_output - self.output)

    def dSigmoid(self):
        return self.output * (1 - self.output)

    def dzdw(self, index):
        return self.inputs[index]
