import random
from tower import Tower

class GeneticAlgorithm:
    def __init__(self, max_towers, max_bandwidth, x_range, y_range, population_size) -> None:
        self.population = self.initialize_population(max_towers, max_bandwidth, x_range, y_range, population_size)


    def initialize_population(self, max_towers, max_bandwidth, x_range, y_range, population_size):
        population = []
        for _ in range(population_size):
            num_towers = random.randint(1, max_towers)
            
            # Build random towers
            towers = []
            for _ in range(num_towers):
                x = random.uniform(0, x_range)
                y = random.uniform(0, y_range)
                bandwidth = random.uniform(0, max_bandwidth)
                twr = Tower(x, y, bandwidth)
                towers.append(twr)
            
            population.append(towers)

        return population
    
    # must implement
    def fitness(self):
        pass

    def selection(self, population, num_parents):
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[:num_parents]
    
    
