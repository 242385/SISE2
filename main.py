# Import, system
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

from classes.neuron import *
from classes.layer import *
from classes.MLP import *

### STATIC PARAMETERS ###

numbers = list()  # number of neurons in each layer
considerBias = None  # 0/1
considerMomentum = None  # 0/1
programMode = None  # 0 - learning/1 - testing
biasValue = None  # 0-1
learningRate = None  # 0-1
momentumRate = None  # 0-1
epochsNumber = None  # int
networkFile = None  # path
learningFile = None  # path
testingFile = None  # path

dataVector = [list()]
target = list()

testedData = [list()]
targetsTesting = list()
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
            elif biasValue is None:
                biasValue = float(line)
            elif learningRate is None:
                learningRate = float(line)
            elif momentumRate is None:
                momentumRate = float(line)
            elif epochsNumber is None:
                epochsNumber = int(line)
            elif networkFile is None:
                networkFile = line
            elif learningFile is None:
                learningFile = line
            elif testingFile is None:
                testingFile = line
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
    with open(learningFile, 'r') as file:
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
    global testedData, outputs, targetsTesting
    data = []
    target = []
    output = []
    with open(testingFile, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                values = line.split(',')
                features = [float(value) for value in values[:4]]
                data.append(features)
                output.append([])
                if values[4] == 'Iris-setosa':
                    target.append([1, 0, 0])
                if values[4] == 'Iris-versicolor':
                    target.append([0, 1, 0])
                if values[4] == 'Iris-virginica':
                    target.append([0, 0, 1])

    testedData = np.array(data)
    outputs = np.array(output)
    targetsTesting = np.array(target)


def setupMLP():
    inputNeuronsNumber = len(dataVector[0])
    hiddenLayersNumber = len(numbers) - 1
    hiddenLayersNeuronNumbers = list()
    for i in range(0, hiddenLayersNumber):
        hiddenLayersNeuronNumbers.append(numbers[i])

    outputNeuronsNumber = numbers[len(numbers) - 1]

    if considerBias:
        bias = biasValue
    else:
        bias = 0

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
    errorsPlot = []
    epochsPlot = []
    error = 1
    for epoch in range(epochs):
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
            errorsPlot.append(float(error))
            epochsPlot.append(epoch)

    plotting(epochsPlot, errorsPlot)


def plotting(epochs, errors):
    plt.plot(epochs, errors)
    plt.xlabel("Epoka")
    plt.ylabel("Wartość Błędu")
    plt.title("Zależność Wartości Błędu od Epoki")
    plt.xticks(np.arange(start=0, stop=(len(epochs) + 1), step=(len(epochs) / 10)))
    plt.show()


def test(mlp, test_inputs, test_targets):
    # Store the number of correct predictions
    correct_predictions = 0
    targets = list()
    for t in test_targets:
        targets.append(np.argmax(t))

    # For each test input
    for i in range(len(test_inputs)):
        # Get the MLP's output for this input
        mlp.setInput(test_inputs[i])
        mlp.forwardPropagation()
        output = mlp.networkOutput()

        # If the output is close to the target, count it as a correct prediction
        outputMax = np.argmax(output)

        if abs(output[outputMax] - test_targets[i][targets[i]]) <= 0.2:
            correct_predictions += 1

    # Calculate the accuracy
    accuracy = correct_predictions / len(test_inputs)
    print(f"Properly classified: {accuracy}/1.0")


train(mlp, dataVector, target, learningRate, momentumRate, epochsNumber)
test(mlp, testedData, targetsTesting)


def save(path, mlpObject):
    file = open(path, 'wb')
    pickle.dump(mlpObject, file)
    file.close()


def load(path):
    file = open(path, 'rb')
    mlpObject = pickle.load(file)
    file.close()
    return mlpObject
