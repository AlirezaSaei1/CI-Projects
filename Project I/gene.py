import numpy as np

class Gene:
    def __init__(self, towers, n, m) -> None:
        self.towers = towers
        self.n = n
        self.m = m
        self.map = self.assign_to_towers()
    

    def assign_to_towers(self):
        return np.random.randint(0, (len(self.towers)), size=(self.n, self.m))