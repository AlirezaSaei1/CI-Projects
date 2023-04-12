import numpy as np

class Genotype:
    def __init__(self, towers, n, m, map = list()) -> None:            
        self.towers = towers
        self.n = n
        self.m = m
        if len(map) == 0:
            self.map = self.assign_to_towers()
        else:
            self.map = map
    

    def assign_to_towers(self):
        return np.random.randint(0, (len(self.towers)), size=(self.n, self.m))