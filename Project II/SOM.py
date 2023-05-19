import numpy as np
from sklearn.preprocessing import StandardScaler


class SOM:
    def __init__(self, x, y, input_dim, sigma=1.0, lr=0.1):
        self.x = x
        self.y = y
        self.input_dim = input_dim  
        self.sigma = sigma  
        self.lr = lr
        self.weights = None


    def neighborhood(self, c, r):
        d = np.sqrt((np.arange(self.x) - c)**2 + (np.arange(self.y) - r)**2)
        return np.exp(-(d**2) / (2 * self.sigma**2))


    def train(self, data, epochs, early_stopping=True, verbose=True):
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.x*self.y, n_init=10, max_iter=100)
        kmeans.fit(data)
        self.weights = kmeans.cluster_centers_.reshape(self.x, self.y, -1)

        best_qe = float('inf')
        for epoch in range(epochs):
            if verbose:
                print(f'Epoch: {epoch}')

            np.random.shuffle(data)

            
            lr = self.lr * np.exp(-epoch/epochs)
            sigma = self.sigma * np.exp(-epoch/epochs)

            
            for d in data:
                # Find best matching unit
                bmu_idx = np.argmin(np.linalg.norm(self.weights - d, axis=-1))
                bmu_x, bmu_y = np.unravel_index(bmu_idx, (self.x, self.y))

                # Update weights for each unit
                for i in range(self.x):
                    for j in range(self.y):
                        h = self.neighborhood(i, j)
                        delta = lr * h[:, np.newaxis] * (d - self.weights[i, j])
                        self.weights[i, j] += delta

            self.sigma = sigma
            self.lr = lr

            qe = np.mean(np.linalg.norm(self.weights.reshape(-1, self.input_dim) - data, axis=-1))
            if verbose:
                print(f'Quantization error: {qe}')

            
            if early_stopping and qe < best_qe:
                best_qe = qe
            elif early_stopping:
                print(f'Stopping early at epoch {epoch}')
                break


    def predict(self, data):
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        bmu_idxs = np.argmin(np.linalg.norm(self.weights.reshape(-1, self.input_dim) - data[:, np.newaxis, :], axis=-1), axis=(1, 2))
        return np.unravel_index(bmu_idxs, (self.x, self.y))
        

    def visualize(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, ax = plt.subplots(figsize=(self.x, self.y))
        im = ax.imshow(np.zeros((self.x, self.y)), cmap='viridis')

        for i in range(self.x):
            for j in range(self.y):
                c = self.weights[i, j]
                ax.text(j, i, str(np.round(c, 2)),
                        va='center', ha='center')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
