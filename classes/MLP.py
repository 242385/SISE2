import numpy as np

class MLP:
    def __init__(self, inputlayer, hiddenlayers, outputlayer):
        self.inputlayer = inputlayer
        self.hiddenlayers = hiddenlayers
        self.outputlayer = outputlayer
        self.correspondingTargets = list()

    def backPropagation(self, target_outputs, learning_rate):
        # Step 1: Compute delta for output layer
        for i in range(len(self.outputlayer.neurons)):
            neuron = self.outputlayer.neurons[i]
            # calculate the error signal (delta)
            output = neuron.calculateNeuronOutput()
            neuron.delta = self.dEdy(output, target_outputs[i]) * self.dSigmdx(neuron)

        # Step 2: Compute delta for hidden layers
        for i in reversed(range(len(self.hiddenlayers))):
            layer = self.hiddenlayers[i]
            next_layer = self.hiddenlayers[i + 1] if i < len(self.hiddenlayers) - 1 else self.outputlayer
            for j in range(len(layer.neurons)):
                neuron = layer.neurons[j]
                # calculate the error signal (delta)
                neuron.delta = sum([next_neuron.weights[j] * next_neuron.delta for next_neuron in next_layer.neurons]) * self.dSigmdx(neuron)

        # Step 3: Update weights
        # For hidden layers
        for layer in self.hiddenlayers:
            for neuron in layer.neurons:
                for k in range(len(neuron.weights)):
                    neuron.weights[k] -= learning_rate * neuron.delta * self.dInpdW(layer)[k]

        # For output layer
        for neuron in self.outputlayer.neurons:
            for k in range(len(neuron.weights)):
                neuron.weights[k] -= learning_rate * neuron.delta * self.dInpdW(self.hiddenlayers[-1])[k]

    def forwardPropagation(self):
        self.hiddenlayers[0].inputs = self.inputlayer.layerOutput()

        for i in range(1, len(self.hiddenlayers)):
            self.hiddenlayers[i].inputs = self.hiddenlayers[i-1].layerOutput()

        self.outputlayer.inputs = self.hiddenlayers[len(self.hiddenlayers)-1].layerOutput()
        return 0

    def calculateResult(self):
        return self.outputlayer.layerOutput()

    def errorFunction(self, y, d):
        return 0.5 * (y-d)**2

    # Partial derivatives

    def dEdy(self, y, d):
        return y - d

    def dSigmdx(self, neuron):
        s = 1 / (1 + np.exp(-neuron.calculateNeuronOutput()))
        return s * (1 - s)

    def dInpdW(self, layer):
        return layer.inputs
