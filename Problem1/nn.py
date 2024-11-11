import numpy as np
from main1 import Bayes
import matplotlib.pyplot as plt
class Functions():
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def sigmoidDerivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def phiExponential(self, x):
        f = 2 * np.exp(0.5*x)

        return f

    def phiExponentialDerivative(self, x):
        f = np.exp(0.5*x)

        return f
    
    def psiExponential(self, x):
        f = 2 * np.exp(-0.5 * x)
        
        return f
    def psiExponentialDerivative(self, x):
        f = -np.exp(-0.5 * x)

        return f
    
    def phiCE(self, x):
        return -np.log(1 - x)
    
    def phiCEDerivative(self, x):
        return 1 / (1-x)

    def psiCE(self, x):
        return -np.log(x)

    def psiCEDerivative(self, x):
        return -1 / x

    def relu(self, x):
        return np.maximum(0, x)

    def reluDerivative(self, x):
        return np.where(x > 0, 1, 0)


class NeuralNetwork(Functions):
    def __init__(self, numberOfInputs, sizeOfHiddenLayer, numberOfOutputs):
        self.numberOfInputs = numberOfInputs
        self.sizeOfHiddenLayer = sizeOfHiddenLayer
        self.numberOfOutputs = numberOfOutputs
        
        self.hiddenLayerWeights = np.random.normal(0, 1/20, size = (self.sizeOfHiddenLayer, self.numberOfInputs)) 
        self.hiddenLayerBiases = np.zeros((self.sizeOfHiddenLayer, 1))

        self.outputLayerWeights = np.random.normal(0, 1/20, size = (self.numberOfOutputs, self.sizeOfHiddenLayer)) 
        self.outputLayerBias = 0

    def hiddenLayer(self, x):
        
        hiddenLayerOutput = self.hiddenLayerWeights @ x + self.hiddenLayerBiases

        return hiddenLayerOutput
    
    def outputLayer(self, x):

        output = self.outputLayerWeights @ x + self.outputLayerBias
        return output


    def gradient(self, x):
        w1 = self.hiddenLayer(x)
        z1 = self.relu(w1)
        w2 = self.outputLayer(z1)

        y = self.sigmoid(w2)
        
        u2 = 1
        v2 = u2 * self.sigmoidDerivative(w2)
        
        u1 = self.outputLayerWeights.T @ v2

        v1 = u1 * self.reluDerivative(w1)

        dA2 = v2 @ z1.T
        dB2 = v2

        dA1 = v1 @ x.T
        dB1 = v1

        fin = np.array([dA1, dB1, dA2, dB2], dtype = object)

        return fin

    def train(self, x_H0, x_H1):
        m = 0.0002
        theta = np.empty(4, dtype=object)

        theta[0] = self.hiddenLayerWeights.copy()
        theta[1] = self.hiddenLayerBiases.copy()
        theta[2] = self.outputLayerWeights.copy()
        theta[3] = self.outputLayerBias
        oldTheta = np.copy(theta)
        
        index = 0
        epoch = 0
        est = np.zeros(20)
        estIndex = 0
        
        # b = Bayes(200)
        # x_H0 = b.generateH0Pairs()
        # x_H1 = b.generateH0Pairs()

        X1 = np.vstack(x_H0[0])
        X2 = np.vstack(x_H0[1])

        gradsX1 = self.gradient(X1)
        gradsX2 = self.gradient(X2)
        px1 = gradsX1 ** 2
        px2 = gradsX2 ** 2
        
        c = 10**(-8)

        for i in range(30000):
        # while True:
            if estIndex == 19:
                estIndex = 0
                


            if index == 199:
                index = 0
                epoch +=1
                plt.scatter(epoch, np.mean(est))
                print("Epoch:", epoch, "Cost:", np.mean(est))
                

            X1 = np.vstack(x_H0[index])
            X2 = np.vstack(x_H1[index])

            x1 = self.u(X1)
            x2 = self.u(X2)
            # print((self.phiCE(x1) + self.psiCE(x2)))

            gradsX1 = self.gradient(X1)
            gradsX2 = self.gradient(X2)

            phiDev = self.phiCEDerivative(x1)
            psiDev = self.psiCEDerivative(x2)

            # print(gradsX1, gradsX2)
            # print(theta[0].shape, theta[1].shape, theta[2].shape)
            # print(theta[3])
            #---Theta without ADAM ---#
            # theta[0] = oldTheta[0] - m * (phiDev * gradsX1[0] + psiDev * gradsX2[0])
            # theta[1] = oldTheta[1] - m * (phiDev * gradsX1[1] + psiDev * gradsX2[1])
            # theta[2] = oldTheta[2] - m * (phiDev * gradsX1[2] + psiDev * gradsX2[2])
            # theta[3] = oldTheta[3] - m * (phiDev * gradsX1[3] + psiDev * gradsX2[3])

            #---Theta with ADAM ---#
            theta[0] = oldTheta[0] - m * (phiDev * gradsX1[0] / np.sqrt(c + px1[0]) + psiDev * gradsX2[0] / np.sqrt(c + px2[0]))
            theta[1] = oldTheta[1] - m * (phiDev * gradsX1[1] / np.sqrt(c + px1[1]) + psiDev * gradsX2[1] / np.sqrt(c + px2[1]))
            theta[2] = oldTheta[2] - m * (phiDev * gradsX1[2] / np.sqrt(c + px1[2]) + psiDev * gradsX2[2] / np.sqrt(c + px2[2]))
            theta[3] = oldTheta[3] - m * (phiDev * gradsX1[3] / np.sqrt(c + px1[3]) + psiDev * gradsX2[3] / np.sqrt(c + px2[3]))
            # print(theta[0].shape, theta[1].shape, theta[2].shape)
            # print(theta[3])

            self.hiddenLayerWeights = theta[0].copy()
            self.hiddenLayerBiases  = theta[1].copy()
            self.outputLayerWeights = theta[2].copy()
            self.outputLayerBias    = theta[3].copy()
            # print(self.outputLayerBias)

            est[estIndex] = (self.phiCE(x1) + self.psiCE(x2)).item()
            oldTheta = theta.copy()
            px1 = self.adam(px1, gradsX1)
            px2 = self.adam(px2, gradsX2)

            estIndex += 1
            index +=1
        plt.show()
        # print(theta)

    def adam(self, oldPower, grad):
        self.lamda = 0.1
        power = (1 - self.lamda) * oldPower + self.lamda * (grad **2)

        return power

        
    def u(self, x):
        w1 = self.hiddenLayerWeights @ x + self.hiddenLayerBiases
        z1 = self.relu(w1)
        
        w2 = self.outputLayerWeights @ z1 + self.outputLayerBias

        y = self.sigmoid(w2)

        return y

    def testing(self, data):
        fails = 0
        for k in data:
            f = self.u(np.vstack(k))
            if f <= 1/2:
                fails +=1
        print(fails / len(data))

        

if __name__ == "__main__":
    nn = NeuralNetwork(2, 20, 1)
    x = np.vstack([1,2])


    # nn.train()
    c = Bayes(1000000)
    x0 = c.generateH0Pairs()
    x1 = c.generateH1Pairs()
    print(x0[0], x1[0])
    # nn.testing(x0)    
    # nn.testing(x1)    
        
        

        

