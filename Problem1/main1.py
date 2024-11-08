import numpy as np


class Bayes:
    def __init__(self, size) -> None:
        self.size = size

    def generateH0Pairs(self):
        # Generate pairs with mean 0 and sigma^2 1

        x1_H0 =  np.random.normal(0, 1, size = self.size) 
        x2_H0 =  np.random.normal(0, 1, size = self.size)
        # print(x1_H0, x2_H0)
        x = np.column_stack((x1_H0, x2_H0))
        
        # print(x)
        return x

    def generateH1Pairs(self):

        x11_H1 = np.random.normal(-1, 1, size = self.size) # Generate data for x1 form both gaussians and choose later based on uniform
        x12_H1 = np.random.normal(1, 1, size = self.size)
        
        x21_H1 = np.random.normal(-1, 1, size = self.size) # Generate data for x2 form both gaussians and choose later based on uniform
        x22_H1 = np.random.normal(1, 1, size = self.size)

        indices_1 = np.random.uniform(0,1, size = self.size) # Indices array to choose from x11, x12 with prob 0.5
        indices_2 = np.random.uniform(0,1, size = self.size) # Indices array to choose from x21, x22 with prob 0.5
        
        x_0 = np.where(indices_1 <= 0.5, x11_H1, x12_H1)
        x_1 = np.where(indices_2 <= 0.5, x21_H1, x22_H1)
        x = np.column_stack((x_0,x_1))

        return x
    

    def gaussianPDF(self, x, mean, sigma):
        g = np.exp(-(x - mean)**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
        return g

    def probDensityH0Improoved(self, x):
        mean = 0
        sigma = 1

        f_00 = self.gaussianPDF(x[:, 0], mean, sigma)
        f_01 = self.gaussianPDF(x[:, 1], mean, sigma)

        f_0 = f_00 * f_01
        # returns and array with F(x1,x2)=f(x1)*f(x2) for all x1, x2
        return f_0
    
    
    def probDensityH1Improoved(self,x):
        f_00 = self.gaussianPDF(x[:, 0], -1, 1)
        f_01 = self.gaussianPDF(x[:, 0], 1, 1)

        f_10 = self.gaussianPDF(x[:, 1], -1, 1)
        f_11 = self.gaussianPDF(x[:, 1], 1, 1)
        
        f_0 = 0.5*(f_00 + f_01)
        f_1 = 0.5*(f_10 + f_11)
        
        return f_0 * f_1 # returns and array with F(x1,x2)=f(x1)*f(x2) for all x1, x2

    def testUnderH0(self, set):
        L = self.probDensityH0Improoved(set) / self.probDensityH1Improoved(set)

        fails = np.sum(L <= 1)
        return fails / self.size
    
    def testUnderH1(self, set):
        L = self.probDensityH1Improoved(set) / self.probDensityH0Improoved(set)

        fails = np.sum(L <= 1)
        return fails / self.size

if __name__ == "__main__":
    size = int(1E6) # How many pairs i want

    b = Bayes(size)

    #Generating Random Pairs
    x_H0 = b.generateH0Pairs()
    x_H1 = b.generateH1Pairs()
    
    #Testing

    e1 = b.testUnderH0(x_H0)

    e2 = b.testUnderH1(x_H1)

    print("Error from H0", str(e1 *100),"%")
    print("Error from H1", str(e2 *100),"%")