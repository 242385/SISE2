# Import, system
import sys
import re
import queue
import time
from classes.neuron import *
from classes.layer import *

### STATIC PARAMETERS ###

numbers = []            # number of neurons in each layer
considerBias = None     # 0/1
programMode = None      # 0 - learning/1 - testing
networkFile = None      # path
patternFile = None      # path

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

def setupMLP():
    print("Ile neuronów umieścić w warstwie wejściowej?")
    inputNeuronsNumber = int(input())
    clear()
    print("Ile ma być warstw ukrytych?")
    hiddenLayersNumber = int(input())
    clear()
    hiddenLayersNeuronNumbers = list()
    for i in range(0, hiddenLayersNumber):
        print(f"Liczba neuronów w warstwie ukrytej #{i+1}:")
        hiddenLayersNeuronNumbers.append(int(input()))
        clear()

    


setupMLP()





if programMode == 0:
    import learning
else:
    import testing

