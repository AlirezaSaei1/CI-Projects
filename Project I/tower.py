from config import Configurations as cnf
import itertools
class Tower:  
    id_iter = itertools.count()
    def __init__(self, x, y, bw) -> None:
        self.id = next(Tower.id_iter)
        self.x = x
        self.y = y
        self.bw = bw
        self.cost = self.calculate_cost()


    def calculate_cost(self):
        return self.bw * cnf.maintenance + cnf.cost
    
    
    def calculate_total_cost(self, towers):
        return sum([tower.cost for tower in towers])
    

