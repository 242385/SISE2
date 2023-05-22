class MLP:
    def __init__(self, inputlayer, hiddenlayers, outputlayer):
        self.inputlayer = inputlayer
        self.hiddenlayers = hiddenlayers
        self.outputlayer = outputlayer
        self.correspondingTargets = list()

    def backPropagation(self):
        return 0

    def forwardPropagation(self):
        self.hiddenlayers[0].inputs = self.inputlayer.layerOutput()

        for i in range(1, len(self.hiddenlayers)):
            self.hiddenlayers[i].inputs = self.hiddenlayers[i-1].layerOutput()

        self.outputlayer.inputs = self.hiddenlayers[len(self.hiddenlayers)-1].layerOutput()
        return 0

    def calculateResult(self):
        return self.outputlayer.layerOutput()

    def errorFunction(self):
        return 0
