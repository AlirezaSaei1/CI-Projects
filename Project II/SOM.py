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
        

    def train(self, data, epochs):
        for epoch in range(epochs):
            print(f'Epoch: {epoch}')
            np.random.shuffle(data)
            for d in data:
                # Find best matching unit
                bmu_idx = np.argmin(np.linalg.norm(self.weights - d, axis=-1))
                bmu_x, bmu_y = np.unravel_index(bmu_idx, (self.x, self.y))

                # Update weights
                for i in range(self.x):
                    for j in range(self.y):
                        h = self._neighborhood(i, j)
                        delta = self.lr * h[:, np.newaxis] * (d - self.weights[i, j])
                        self.weights[i, j] += delta
                        
            # Decay
            self.sigma = self.sigma * 0.9
            self.lr = self.lr * 0.9
