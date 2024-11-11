from main1 import Bayes
from nn import NeuralNetwork 


if __name__ == "__main__":
    b = Bayes(1000000)

    x_H0 = b.generateH0Pairs()
    x_H1 = b.generateH1Pairs()

    e1 = b.testUnderH0(x_H0)


    e2 = b.testUnderH1(x_H1)

 
    nn = NeuralNetwork(2, 20, 1)
    b = Bayes(200)
    x_H0_train = b.generateH0Pairs()
    x_H1_train = b.generateH1Pairs()

    nn.train(x_H0_train, x_H1_train)

    nn.testing(x_H0)
    nn.testing(x_H1)
    
    print("Bayes Error from H0", str(e1 *100),"%")
    print("Bayes Error from H1", str(e2 *100),"%")
   


    

