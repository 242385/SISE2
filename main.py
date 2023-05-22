# Import, system
import random
import sys
import re
import queue
import time
from classes.neuron import *
from classes.layer import *
from classes.MLP import *

### STATIC PARAMETERS ###

numbers = []  # number of neurons in each layer
considerBias = None  # 0/1
programMode = None  # 0 - learning/1 - testing
networkFile = None  # path
patternFile = None  # path

dataVector = [0.23, 2.3, 5.5, 1.12]

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
        x = r.uniform(-1.0, 1.0)
        weights.append(x)
    return weights


def setupMLP():
    inputNeuronsNumber = len(dataVector)
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
    inputLayer = Layer(neurons, dataVector, layerType.INPUT)

    neurons = list()
    for j in range(0, hiddenLayersNeuronNumbers[0]):
        n = Neuron(randomWeights(len(inputLayer.neurons)), bias)
        neurons.append(n)
    hiddenLayers = list()
    hiddenLayers.append(Layer(neurons, None, layerType.HIDDEN))

    for i in range(1, hiddenLayersNumber):
        neurons = list()
        for j in range(0, hiddenLayersNeuronNumbers[i]):
            n = Neuron(randomWeights(hiddenLayersNeuronNumbers[i-1]), bias)
            neurons.append(n)
        hiddenLayers.append(Layer(neurons, None, layerType.HIDDEN))

    neurons = list()
    for i in range(0, outputNeuronsNumber):
        n = Neuron(randomWeights(hiddenLayersNeuronNumbers[len(hiddenLayersNeuronNumbers)-1]), bias)
        neurons.append(n)
    outputLayer = Layer(neurons, None, layerType.OUTPUT)
    return MLP(inputLayer, hiddenLayers, outputLayer)


### MAIN ###

mlp = setupMLP()
mlp.forwardPropagation()

if programMode == 0:
    import learning
else:
    import testing
