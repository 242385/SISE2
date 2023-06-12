# Import, system
import pickle
import matplotlib.pyplot as plt

from classes.layer import *
from classes.MLP import *

### STATIC PARAMETERS ###

numbers = list()  # number of neurons in each layer
considerBias = None  # 0/1
considerMomentum = None  # 0/1
programMode = None  # 0 - learning/1 - testing
exercise = None  # 0 - classification/1 - autoencoder
biasValue = None  # 0-1
learningRate = None  # 0-1
momentumRate = None  # 0-1
epochsNumber = None  # int
networkFile = None  # path
learningFile = None  # path
testingFile = None  # path
autolearningFile = None  # path
autotestingFile = None  # path

dataVector = [list()]
target = list()

testedData = list()
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
            elif exercise is None:
                exercise = int(line)
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
            elif autolearningFile is None:
                autolearningFile = line
            elif autotestingFile is None:
                autotestingFile = line
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


def loadLearningData():
    global dataVector, target
    data = []
    target = []
    with open(learningFile if exercise == 0 else autolearningFile, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                values = line.split(',')
                features = [float(value) for value in values[:4]]
                data.append(features)
                if exercise == 0:
                    if values[4] == 'Iris-setosa':
                        target.append([1, 0, 0])
                    if values[4] == 'Iris-versicolor':
                        target.append([0, 1, 0])
                    if values[4] == 'Iris-virginica':
                        target.append([0, 0, 1])
                else:
                    if values[4] == '0':
                        target.append([1.0, 0.0, 0.0, 0.0])
                    if values[4] == '1':
                        target.append([0.0, 1.0, 0.0, 0.0])
                    if values[4] == '2':
                        target.append([0.0, 0.0, 1.0, 0.0])
                    if values[4] == '3':
                        target.append([0.0, 0.0, 0.0, 1.0])

    dataVector = np.array(data)
    target = np.array(target)


def loadTestingData():
    global testedData, targetsTesting
    data = []
    target = []
    with open(testingFile if exercise == 0 else autolearningFile, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                values = line.split(',')
                features = [float(value) for value in values[:4]]
                data.append(features)
                if exercise == 0:
                    if values[4] == 'Iris-setosa':
                        target.append([1, 0, 0])
                    if values[4] == 'Iris-versicolor':
                        target.append([0, 1, 0])
                    if values[4] == 'Iris-virginica':
                        target.append([0, 0, 1])
                else:
                    if values[4] == '0':
                        target.append([1.0, 0.0, 0.0, 0.0])
                    if values[4] == '1':
                        target.append([0.0, 1.0, 0.0, 0.0])
                    if values[4] == '2':
                        target.append([0.0, 0.0, 1.0, 0.0])
                    if values[4] == '3':
                        target.append([0.0, 0.0, 0.0, 1.0])

    testedData = np.array(data)
    targetsTesting = np.array(target)

def save(path, mlpObject):
    file = open(path, 'wb')
    pickle.dump(mlpObject, file)
    file.close()


def load(path):
    file = open(path, 'rb')
    mlpObject = pickle.load(file)
    file.close()
    return mlpObject

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

    inputLayer = Layer(inputNeuronsNumber, bias)
    hiddenLayers = list()

    for i in range(hiddenLayersNumber):
        hiddenLayers.append(Layer(hiddenLayersNeuronNumbers[i], bias))

    outputLayer = Layer(outputNeuronsNumber, bias)

    return MLP(inputLayer, hiddenLayers, outputLayer)


### MAIN ###

loadLearningData()
loadTestingData()
mlp = setupMLP()


def train(mlp, inputpoints, targets, learning_rate, momentumCoeff, epochs):
    errorsPlot = []
    epochsPlot = []
    # Combine inputpoints and targets into pairs
    data = list(zip(inputpoints, targets))

    for epoch in range(epochs):
        # Shuffle the data
        random.shuffle(data)

        # Unpack shuffled data
        inputpoints_shuffled, targets_shuffled = zip(*data)

        errors = list()
        for i in range(len(inputpoints_shuffled)):
            mlp.backPropagation(inputpoints_shuffled[i], targets_shuffled[i], learning_rate, momentumCoeff, considerBias, considerMomentum)

            # Compute and print error
            output = mlp.output_layer.get_outputs()
            err_param = [(inputpoints_shuffled[i], targets_shuffled[i])]
            error = mlp.calculate_total_error(err_param)
            errors.append(error)

        total_error = sum(errors) / len(errors)
        print(f'Epoch: {epoch}, Error: {total_error}')
        errorsPlot.append(float(total_error))
        epochsPlot.append(epoch)

    plotting(epochsPlot, errorsPlot)

def plotting(epochs, errors):
    plt.plot(epochs, errors)
    plt.xlabel("Epoka")
    plt.ylabel("Wartość Błędu")
    plt.title("Zależność wartości błędu od epoki")
    plt.xticks(np.arange(start=0, stop=(len(epochs) + 1), step=(len(epochs) / 10)))
    plt.show()


def test(mlp, test_inputs, test_targets):
    # Initialize the confusion matrix with 0's
    if exercise == 0:
        confusion_matrix = np.zeros((3, 3))
    else:
        confusion_matrix = np.zeros((4, 4))
    print("———————————————————————————————————————————————————————————————————————————————————————————————————")
    for i in range(0, len(test_inputs)):
        mlp.forwardPropagation(test_inputs[i])
        output = mlp.output_layer.get_outputs()
        print("INPUT: ", test_inputs[i])
        print("OUTPUT: ", output)
        print("DESIRED: ", test_targets[i])
        print("———————————————————————————————————————————————————————————————————————————————————————————————————")

        # Get the indices of the maximum predicted class and the actual class
        predicted_class = np.argmax(output)
        actual_class = np.argmax(test_targets[i])

        # Update the confusion matrix
        confusion_matrix[actual_class][predicted_class] += 1

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix)

    if exercise == 0:
        # Calculate precision, recall and f1-score for each class
        for i, class_name in enumerate(['setosa', 'versicolor', 'virginica']):
            precision = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
            recall = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
            f1_score = 2 * (precision * recall) / (precision + recall)
            print(f"{class_name} - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

            # If exercise is 0, print the counts of correct predictions

            correct_predictions = np.trace(confusion_matrix)
            accuracy = correct_predictions / np.sum(confusion_matrix)
            print(f"Properly classified: {accuracy}/1.0")
            print(f"Correct setosa: {confusion_matrix[0, 0]}")
            print(f"Correct virginica: {confusion_matrix[1, 1]}")
            print(f"Correct versicolor: {confusion_matrix[2, 2]}")

    else:
        for i in range(0, len(test_inputs)):
            precision = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
            recall = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
            f1_score = 2 * (precision * recall) / (precision + recall)
            print(f"\"1\" on {i} position - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")


if programMode == 0:
    train(mlp, dataVector, target, learningRate, momentumRate, epochsNumber)
    save(networkFile, mlp)
else:
    loadedMLP = load(networkFile)
    test(loadedMLP, testedData, targetsTesting)
