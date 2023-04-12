import random
from tower import Tower
from genotype import Genotype
import numpy as np
from config import Configurations as cnf

class GeneticAlgorithm:
    def __init__(self, population, max_towers, max_bandwidth, x_range, y_range, population_size, mutation_probability=0.4) -> None:
        self.map = population
        self.max_towers = max_towers
        self.max_bandwidth = max_bandwidth
        self.x_range = x_range
        self.y_range = y_range
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.population = self.initialize_population()


    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            num_towers = random.randint(1, self.max_towers)
            
            # Build random towers
            towers = []
            for _ in range(num_towers):
                x = random.uniform(0, self.x_range)
                y = random.uniform(0, self.y_range)
                bandwidth = random.uniform(0, self.max_bandwidth)
                twr = Tower(x, y, bandwidth)
                towers.append(twr)

            population.append(Genotype(np.array(towers), self.x_range, self.y_range))

        return population
    

    def fitness(self, genotype):
        # calculate total cost of towers
        cost = sum([tower.calculate_cost() for tower in genotype.towers])
        print(cost)
        
        # calculate total score of customers
        scores = 0.0
        for i in range(self.x_range):
            for j in range(self.y_range):
                id = genotype.map[i][j]
                T = genotype.get_tower_by_id(id)

                tower_covered_population = 0
                for ii in range(self.x_range):
                    for jj in range(self.y_range):
                        if id == genotype.map[ii][jj]:
                            tower_covered_population += self.map[ii][jj]

                real_block_bw = calculate_real_block_bandwidth((T.x, T.y), (i, j), T.bw, self.map[i][j], tower_covered_population)
                user_bw = calculate_user_bandwidth(real_block_bw, self.map[i][j])
                scores += cnf.get_score(user_bw) * self.map[i][j]
        print(scores)
                

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
    

    def mutate(self, tower):
        attribute_to_mutate = random.choice(["X", "Y", "BW"])
        mutated_tower = Tower(tower.x, tower.y, tower.bw)

        if attribute_to_mutate == "X":
            mutated_tower.x = random.uniform(0, self.x_range)
        elif attribute_to_mutate == "Y":
            mutated_tower.y = random.uniform(0, self.y_range)
        else:
            mutated_tower.bw = random.uniform(0, self.max_bandwidth)

        return mutated_tower
    
    
    def mutate_map(self, map, n, m):
        new_map = np.array([0]*n*m).reshape(n, m)
        for i in range(n):
            for j in range(m):
                if random.random() < 0.1:
                    new_map[i][j] = random.randint(0, self.max_towers - 1)
                else:
                    new_map[i][j] = map[i][j]
        return new_map


    def mutate_population(self, population):
        mutated_population = []
        for genotype in population:
            mutated_towers = []
            for tower in genotype.towers:
                if random.random() < self.mutation_probability:
                    mutated_towers.append(self.mutate(tower))
                else:
                    mutated_towers.append(tower)
            mutated_map = self.mutate_map(genotype.map, self.x_range, self.y_range)
            mutated_genotype = Genotype(mutated_towers, self.x_range, self.y_range, mutated_map)
            mutated_population.append(mutated_genotype)
        return mutated_population


    # We can use other methods later 
    def replace_population(self, population, offspring):
        combined_population = population + offspring
        return sorted(combined_population, key=lambda x: x.fitness, reverse=True)[:self.population_size]



# This function calculates assigned bandwidth for each user in a block
def calculate_user_bandwidth(block_bandwidth, block_population):
    return block_bandwidth/block_population


# This function calculates assigned bandwith for each block
# Notice that this function's return value is nominal
# We need to calculate population for blocks that are covered by a tower
def calcualte_nominal_block_bandwidth(tower_bandwith, block_population, tower_covered_blocks_population):
    return block_population * tower_bandwith / tower_covered_blocks_population


# This function calculates assigned bandwith for each block
# Notice that this function's return value is a real value
def calculate_real_block_bandwidth(tower_coordinates, block_coordinates, tower_bandwidth, block_population, tower_covered_blocks_population):
    mat = np.array([[8, 0], [0, 8]])
    
    mat = np.linalg.inv(mat)
    nominal_bw = calcualte_nominal_block_bandwidth(tower_bandwidth, block_population, tower_covered_blocks_population)
    diff = np.array(block_coordinates) - np.array(tower_coordinates)
    return np.exp(-0.5 * np.matmul(diff, np.matmul(mat, np.transpose(diff)))) * nominal_bw
