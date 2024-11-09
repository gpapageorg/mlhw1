import numpy as np

size = 20
numberOfInputs = 2

hiddenLayerWeights = np.zeros((size, numberOfInputs))
hiddenLayerBiases  = np.zeros((size, 1))

outputLayerWeights = np.random.uniform(size = (1, size))  
outputLayerbias = 0

def hiddenLayer(x):
    "Calculates Hidden Layer Matrices"

    hiddenLayerOutput = hiddenLayerWeights @ x + hiddenLayerBiases
    
    return hiddenLayerOutput

def outputLayer(x):
    output = outputLayerWeights @ x + outputLayerbias

    return output

def  sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x):
    return (np.exp(x) / (np.exp(x) + 1) ** 2)
    
def phi(z):
    #phi(z) for cross entropy#
    f = -np.log(1 - z)
    return f

def phiDerivative(z):
    return 1 / (1 - z)

def psi( z):
    c = -np.log(z)
    return c
def psiDerivative(z):
    return -1 / z
