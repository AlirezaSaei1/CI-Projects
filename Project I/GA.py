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
    

    def one_point_crossover(self, parents, offspring_size):
        offspring = []
        for _ in range(offspring_size):
            parent1 = parents[random.randint(0, len(parents)-1)]
            parent2 = parents[random.randint(0, len(parents)-1)]
            point = random.randint(1, len(parent1)-1)
            offspring1 = parent1[:point] + parent2[point:]
            offspring2 = parent2[:point] + parent1[point:]
            offspring.append(offspring1)
            offspring.append(offspring2)
            
        return offspring
    
    def two_point_crossover(parent1, parent2):
        point1 = random.randint(0, len(parent1)-1)
        point2 = random.randint(point1+1, len(parent1))
        
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

        return child1, child2
    

