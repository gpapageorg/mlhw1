import numpy as np


class NeuralNetwork():
    def __init__(self, size, numberOfInputs):
        np.random.seed(666)

        self.size = size
        self.numberOfInputs = numberOfInputs

        self.hiddenLayerWeights = np.zeros((self.size, self.numberOfInputs))
        self.hiddenLayerBiases  = np.zeros((self.size, 1))
        
        self.outputLayerWeights = np.random.uniform(size = (1, size))  
        self.outputLayerbias = 0
        # print(self.outputLayerWeights.shape)
        self.input = np.vstack([1, 2])    

    def hiddenLayer(self, x):
        # print(x.shape, self.hiddenLayerWeights.shape)
        self.hiddenLayerOutput = self.hiddenLayerWeights @ x + self.hiddenLayerBiases
        
        # self.hiddenLayerOutput = self.__sigmoid(self.hiddenLayerOutput)
        return self.hiddenLayerOutput

    def output(self, x):
        '1x20 * 20x1 + 0 '
        self.output = self.outputLayerWeights @ x + self.outputLayerbias

        return(self.output)

    def  __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def phi(self, z):
        #phi(z) for cross entropy#
        f = -np.log(1 - z)
        return f

    def phiDerivative(self, z):
        return 1 / (1 - z)

    def psi(self, z):
        c = -np.log(z)
        return c
    def psiDerivative(self, z):
        return -1 / z

    def  sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        return (np.exp(x) / (np.exp(x) + 1) ** 2)   

    def gradient(self):
        w1 = self.hiddenLayer(self.input)
        w2 = self.output(w1)

        z = self.input
        z1 = self.sigmoid(w1)
        z2 = self.sigmoid(w2)        

        u2 = 1
        v2 = u2 * self.sigmoidDerivative(w2)

        u1 = np.transpose(self.outputLayerWeights) @ v2
        v1 = u1 * self.sigmoidDerivative(w1)
            
        
        gradWithRespectToA1B1 = v1 @ np.hstack((np.transpose(z), np.ones((1,1))))
        
        
        gradWithRespectToA2B2 = v2 @ np.hstack((np.transpose(z1), np.ones((1,1))))
        gradWithRespectToA2B2 = gradWithRespectToA2B2.flatten()
        gradWithRespectToA2B2 = np.array([gradWithRespectToA2B2[:20], gradWithRespectToA2B2[20]], dtype = object)

        return [gradWithRespectToA1B1, gradWithRespectToA2B2]


    def sgd(self):
        theta = np.empty(4, dtype=object)

        theta[0] = self.hiddenLayerWeights
        theta[1] = self.hiddenLayerBiases
        theta[2] = self.outputLayerWeights
        theta[3] = self.outputLayerbias

        m = 10E-4
        
        oldTheta = theta

        grad = self.gradient()
        # print(theta[0].shape, grad[0][:, :2].shape)
        # print(theta[1].shape, grad[0][:,2].shape)
        # print(theta[2].shape, grad[1][0].shape)
        dA1 = grad[0][:, :2]
        dB1 = grad[0][:,2]
        dA2 = grad[1][0]
        dB2 = grad[1][0]
        
        theta[0] = oldTheta - m * (self.phiDerivative(u(x[0])))

        

    def u(self, X, theta):
        "Function of Neural Network Takes Input Gives Output"
        # theta = [hiddenLayerWeights, HiddenLayerBiases, OutputlayerWeights, OutputLayerBias]

        self.hiddenLayerWeights = theta[0]
        self.hiddenLayerBiases = theta[1]
        self.outputLayerWeights = theta[2]
        self.outputLayerbias = theta[3]

        w1 = self.hiddenLayer(X)
        z1 = self.sigmoid(w1)
        w2 = self.output(z1)
        y = self.sigmoid(w2)




nn = NeuralNetwork(20, 2)
# nn.output(np.vstack([10, 1]))
# nn.gradient()
nn.sgd()