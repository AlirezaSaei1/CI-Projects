import numpy as np


class SOM:
    def __init__(self, x, y, input_dim, sigma=1.0, lr=0.1):
        self.x = x
        self.y = y
        self.input_dim = input_dim  
        self.sigma = sigma  
        self.lr = lr
        self.weights = np.random.rand(x, y, input_dim) 

    def neighborhood(self, c, r):
        d = np.sqrt((np.arange(self.x) - c)**2 + (np.arange(self.y) - r)**2)
        return np.exp(-(d**2) / (2 * self.sigma**2))
        

    def train():
        pass
