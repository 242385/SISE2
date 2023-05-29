# Import, system
import random
import sys
import re
import queue
import time

import numpy as np

from classes.neuron import *
from classes.layer import *
from classes.MLP import *

### STATIC PARAMETERS ###

numbers = []  # number of neurons in each layer
considerBias = None  # 0/1
considerMomentum = None  # 0/1
programMode = None  # 0 - learning/1 - testing
networkFile = None  # path
patternFile = None  # path

dataVector = [list()]
target = list()

testedData = [list()]
outputs = list()

### LOADING SETTINGS ###

with open("settings.txt", "r") as file:
    lines = file.readlines()
    reading_numbers = True

    for line in lines:
        line = line.strip()
        if not line:
            reading_numbers = False
            continue

        if reading_numbers:
            try:
                numbers.append(int(line))
            except ValueError:
                print("Unexpected value encountered while reading numbers:", line)
                continue

        if not reading_numbers:
            if considerBias is None:
                considerBias = int(line)
            elif considerMomentum is None:
                considerMomentum = int(line)
            elif programMode is None:
                programMode = int(line)
            elif networkFile is None:
                networkFile = line
            elif patternFile is None:
                patternFile = line
            else:
                print("Unexpected line in the file:", line)


### FUNCTIONS ###

# Flush keyboard input
def flush_input():
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        import sys, termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)


# Clear screen
def clear():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
    flush_input()


def randomWeights(count):
    weights = list()
    for i in range(0, count):
        r = random.Random()
        x = r.uniform(-0.5, 0.5)
        weights.append(x)
    return weights


def loadLearningData():
    global dataVector, target
    data = []
    target = []
    with open('patterns/learning.csv', 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                values = line.split(',')
                features = [float(value) for value in values[:4]]
                data.append(features)
                if values[4] == 'Iris-setosa':
                    target.append([1, 0, 0])
                if values[4] == 'Iris-versicolor':
                    target.append([0, 1, 0])
                if values[4] == 'Iris-virginica':
                    target.append([0, 0, 1])

    dataVector = np.array(data)
    target = np.array(target)


def loadTestingData():
    global testedData, outputs
    data = []
    output = []
    with open('patterns/learning.csv', 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                values = line.split(',')
                features = [float(value) for value in values[:4]]
                data.append(features)
                output.append([])

    testedData = np.array(data)
    outputs = np.array(output)


def setupMLP():
    inputNeuronsNumber = len(dataVector[0])
    print("Ile ma być warstw ukrytych?")
    hiddenLayersNumber = int(input())
    clear()
    hiddenLayersNeuronNumbers = list()
    for i in range(0, hiddenLayersNumber):
        print(f"Liczba neuronów w warstwie ukrytej #{i + 1}:")
        hiddenLayersNeuronNumbers.append(int(input()))
        clear()

    if considerBias:
        print("Proszę podać bias (ma znaczenie tylko, gdy ustawiono branie biasu pod uwagę):")
        bias = float(input())
        clear()
    else:
        bias = 0

    print("Liczba neuronów w warstwie wyjściowej:")
    outputNeuronsNumber = int(input())
    clear()

    neurons = list()
    for i in range(0, inputNeuronsNumber):
        n = Neuron(None, bias)
        neurons.append(n)
    inputLayer = Layer(neurons, None, layerType.INPUT)

    neurons = list()
    for j in range(0, hiddenLayersNeuronNumbers[0]):
        n = Neuron(randomWeights(len(inputLayer.neurons)), bias)
        neurons.append(n)
    hiddenLayers = list()
    hiddenLayers.append(Layer(neurons, None, layerType.HIDDEN))

    for i in range(1, hiddenLayersNumber):
        neurons = list()
        for j in range(0, hiddenLayersNeuronNumbers[i]):
            n = Neuron(randomWeights(hiddenLayersNeuronNumbers[i - 1]), bias)
            neurons.append(n)
        hiddenLayers.append(Layer(neurons, None, layerType.HIDDEN))

    neurons = list()
    for i in range(0, outputNeuronsNumber):
        n = Neuron(randomWeights(hiddenLayersNeuronNumbers[len(hiddenLayersNeuronNumbers) - 1]), bias)
        neurons.append(n)
    outputLayer = Layer(neurons, None, layerType.OUTPUT)
    return MLP(inputLayer, hiddenLayers, outputLayer)


### MAIN ###

loadLearningData()
loadTestingData()
mlp = setupMLP()


def train(mlp, inputpoints, targets, learning_rate, momentumCoeff, epochs):
    for epoch in range(epochs):
        # Reset previous biases and weights:
        #for h in mlp.hiddenlayers:
        #    for n in h.neurons:
        #        n.prev_weight_updates = [0 for _ in range(len(n.weights))]
        #        n.prev_bias_update = 0
        #for n in mlp.outputlayer.neurons:
        #    n.prev_weight_updates = [0 for _ in range(len(n.weights))]
        #    n.prev_bias_update = 0

        for i in range(len(inputpoints)):
            # Forward propagate
            mlp.setInput(inputpoints[i])
            mlp.forwardPropagation()

            # Compute and print error
            output = mlp.networkOutput()
            error = mlp.computeError(targets[i])

            # Back propagate
            mlp.backPropagation(targets[i], learning_rate, momentumCoeff, considerBias, considerMomentum)

        print(f'Epoch: {epoch}, Error: {error}')


def test(mlp, test_inputs, test_targets):
    # Store the number of correct predictions
    correct_predictions = 0

    # For each test input
    for i in range(len(test_inputs)):
        # Get the MLP's output for this input
        mlp.setInput(test_inputs[i])
        mlp.forwardPropagation()
        output = mlp.networkOutput()

        # If the output is close to the target, count it as a correct prediction
        if np.argmax(output) == np.argmax(test_targets[i]):
            correct_predictions += 1

    # Calculate the accuracy
    accuracy = correct_predictions / len(test_inputs)
    print(f"Properly classified: {accuracy}/1.0")


train(mlp, dataVector, target, 0.5, 0.1, 500)
test(mlp, testedData, target)

if programMode == 0:
    import learning
else:
    import testing
