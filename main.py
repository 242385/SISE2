# Import, system
import sys
import re
import queue
import time
from classes.neuron import *

numbers = []            # number of neurons in each layer
considerBias = None     # 0/1
programMode = None      # 0 - learning/1 - testing
networkFile = None      # path
patternFile = None      # path

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

if programMode == 0:
    import learning
else:
    import testing

n = Neuron([-0.28, 1.42, 0.99], 0)
inputs = [3.5, 1.2, -0.7]

print(f"inputs: {inputs}, output: {n.calculateNeuronOutput(inputs)}")
