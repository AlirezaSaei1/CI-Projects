from config import Configurations as cnf
import itertools
class Tower:  
    id_iter = itertools.count()
    def __init__(self, x, y, bw) -> None:
        self.id = 0
        self.x = x
        self.y = y
        self.bw = bw

    def calculate_cost(self):
        return self.bw * cnf.maintenance + cnf.cost

