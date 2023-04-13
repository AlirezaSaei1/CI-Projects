import numpy as np

class Genotype:
    def __init__(self, towers, n, m, map) -> None:            
        self.towers = towers
        self.n = n
        self.m = m
        if len(map) == 0:
            self.map = self.assign_to_towers()
        else:
            self.map = np.array(map)
        self.correct_tower_ids()
    

    def assign_to_towers(self):
        return np.random.randint(0, (len(self.towers)), size=(self.n, self.m))
    
    
    def get_tower_by_id(self, id):
        for twr in self.towers:
            if id == twr.id:
                return twr
            
    def correct_tower_ids(self):
        for i in range(len(self.towers)):
            self.towers[i].id = i