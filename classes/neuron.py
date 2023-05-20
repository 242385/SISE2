import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return np.interp(s, [0, 1], [-1, 1])

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

        # leave it like that?
        #weights.reverse()
        #weights.append(bias)
        #weights.reverse()


    def calculateNeuronOutput(self, inputs):
        result = np.dot(self.weights, inputs)
        print(f"result: {result}")
        print(f"weights: {self.weights}")
        print(f"inputs: {inputs}")
        return self.activationFunction(sigmoid, result)

    def activationFunction(self, function, value):
        x = function(value + self.bias)
        print(f"result after sigmoid: {x}")
        if x >= 0:
            return True
        else:
            return False
