from enum import Enum


class layerType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class Layer:
    def __init__(self, neurons, layertype):
        self.neurons = neurons
        self.type = layertype

    def calculateLayerOutput(self):
        return 0

    def passOutputFurther(self):
        return 0
