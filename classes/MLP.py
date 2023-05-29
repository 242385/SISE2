import numpy as np

class MLP:
    def __init__(self, inputlayer, hiddenlayers, outputlayer):
        self.inputlayer = inputlayer
        self.hiddenlayers = hiddenlayers
        self.outputlayer = outputlayer
        self.correspondingTargets = list()

    def setInput(self, input):
        self.inputlayer.inputs = input

    def backPropagation(self, targetOutputs, learningRate, momentumCoeff, applyBias, applyMomentum):
        # Calculate the delta (error term) for each neuron in the output layer
        for i in range(len(self.outputlayer.neurons)):
            neuron = self.outputlayer.neurons[i]
            inputs = self.hiddenlayers[-1].outputs
            delta = neuron.dC0da(inputs, targetOutputs[i]) * neuron.dadz(inputs)
            neuron.delta = delta

        # Propagate the error back through the hidden layers
        for i in reversed(range(len(self.hiddenlayers))):
            currentLayer = self.hiddenlayers[i]
            nextLayer = self.hiddenlayers[i + 1] if i < len(self.hiddenlayers) - 1 else self.outputlayer
            for j in range(len(currentLayer.neurons)):
                neuron = currentLayer.neurons[j]
                inputs = self.hiddenlayers[i - 1].outputs if i > 0 else self.inputlayer.outputs
                delta = sum([next_neuron.weights[j] * next_neuron.delta for next_neuron in nextLayer.neurons]) * neuron.dadz(inputs)
                neuron.delta = delta

        # Update the weights and biases of the neurons in the output layer and the hidden layers
        if not applyMomentum:
            momentumCoeff = 0
        for i, layer in enumerate(self.hiddenlayers + [self.outputlayer]):
            inputs = self.hiddenlayers[i - 1].outputs if i != 0 else self.inputlayer.outputs
            for neuron in layer.neurons:
                for k in range(len(neuron.weights)):
                    # Compute gradient (negative for descent)
                    gradient = - neuron.delta * inputs[k]
                    # Compute the difference between current and previous weight
                    weight_difference = neuron.weights[k] - neuron.prev_weights[k]
                    # Compute the weight update
                    delta_weight = learningRate * gradient + momentumCoeff * weight_difference
                    # Update the previous weights to current weights before weight update
                    neuron.prev_weights[k] = neuron.weights[k]
                    # Update the weights
                    neuron.weights[k] += delta_weight

                if applyBias:
                    # Similar steps for the bias
                    gradient_bias = - neuron.delta
                    bias_difference = neuron.bias - neuron.prev_bias
                    delta_bias = learningRate * gradient_bias + momentumCoeff * bias_difference
                    neuron.prev_bias = neuron.bias
                    neuron.bias += delta_bias

    def forwardPropagation(self):
        # The output of the input layer is its inputs
        self.inputlayer.outputs = self.inputlayer.inputs

        # The inputs of the first hidden layer are the outputs of the input layer
        self.hiddenlayers[0].inputs = self.inputlayer.outputs

        for i in range(1, len(self.hiddenlayers)):
            # The outputs of the current hidden layer
            self.hiddenlayers[i - 1].outputs = self.hiddenlayers[i - 1].layerOutput()

            # The inputs of the next hidden layer are the outputs of the current hidden layer
            self.hiddenlayers[i].inputs = self.hiddenlayers[i - 1].outputs

        # The outputs of the last hidden layer
        self.hiddenlayers[-1].outputs = self.hiddenlayers[-1].layerOutput()

        # The inputs of the output layer are the outputs of the last hidden layer
        self.outputlayer.inputs = self.hiddenlayers[-1].outputs

    def networkOutput(self):
        return self.outputlayer.layerOutput()

    def computeError(self, y):
        # Collect all output neurons
        output_neurons = self.outputlayer.neurons

        # Calculate errors for each neuron
        errors = list()
        for i in range(0, len(y)):
            errors.append(output_neurons[i].C0(self.outputlayer.inputs, y[i]))

        return sum(errors)

