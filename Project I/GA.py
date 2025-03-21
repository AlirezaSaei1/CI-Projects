import random
from tower import Tower
from genotype import Genotype
import numpy as np
from config import Configurations as cnf

class GeneticAlgorithm:
    def __init__(self, population, max_towers, max_bandwidth, x_range, y_range, population_size, mutation_probability=0.1, crossover_probability=0.9) -> None:
        self.map = population
        self.max_towers = max_towers
        self.max_bandwidth = max_bandwidth
        self.x_range = x_range
        self.y_range = y_range
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
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

            population.append(Genotype(np.array(towers), self.x_range, self.y_range, []))

        return population
    

    def fitness(self, genotype):
        # calculate total cost of towers
        cost = sum([tower.calculate_cost() for tower in genotype.towers])
        
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

        return scores - cost
                

    def selection(self, population, num_parents):
        parents = []
        for _ in range(num_parents):
            k = 3
            individuals = random.sample(population, k)
            fittest = max(individuals, key=lambda x: self.fitness(x))
            parents.append(fittest)

        return parents
    

    def normalize_genotype(self, geno, towers):
        twrs_size = len(geno.towers)
        new_twrs_size = len(towers)

        # correction of tower IDs
        for i in range(new_twrs_size):
            towers[i].id = i

        if  twrs_size > new_twrs_size:
            max_id = new_twrs_size - 1
            indices = np.where(geno.map > max_id)
            new_values = np.random.randint(low=0, high=max_id+1, size=len(indices[0]))
            geno.map[indices] = new_values
            return Genotype(towers, self.x_range, self.y_range, geno.map)

        elif twrs_size < new_twrs_size:
            for t in range(twrs_size + 1, new_twrs_size):
                for i in range(max(0, int(towers[t].x)-1), min(self.x_range, int(towers[t].x)+2)):
                    for j in range(max(0, int(towers[t].y)-1), min(self.y_range, int(towers[t].y)+2)):
                        geno.map[i][j] = towers[t].id
            return Genotype(towers, self.x_range, self.y_range, geno.map)
        else:
            return Genotype(towers, self.x_range, self.y_range, geno.map)
    

    def one_point_crossover(self, genotypes):
        genotype1, genotype2 = random.sample(genotypes, 2)
        if random.random() < self.crossover_probability:
            crossover_point = random.randint(0, min(len(genotype2.towers), len(genotype1.towers)-1))
            new_genotype1 = Genotype(np.concatenate((genotype1.towers[:crossover_point], 
                                                     genotype2.towers[crossover_point:])), self.x_range, self.y_range, [])
            new_genotype2 = Genotype(np.concatenate((genotype2.towers[:crossover_point], 
                                                     genotype1.towers[crossover_point:])), self.x_range, self.y_range, [])
            genotype1 = new_genotype1
            genotype2 = new_genotype2
        genotype1 = self.normalize_genotype(genotype1, genotype1.towers)
        genotype2 = self.normalize_genotype(genotype2, genotype2.towers)
        return genotype1, genotype2

    

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
    
    
    def mutate_map(self, map, n, m, towers_num):
        new_map = np.array([0]*n*m).reshape(n, m)
        for i in range(n):
            for j in range(m):
                if random.random() < 0.1:
                    new_map[i][j] = random.randint(0, towers_num - 1)
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
            mutated_map = self.mutate_map(genotype.map, self.x_range, self.y_range, len(genotype.towers))
            mutated_genotype = Genotype(mutated_towers, self.x_range, self.y_range, mutated_map)
            mutated_population.append(mutated_genotype)
        return mutated_population


    # We can use other methods later 
    def replace_population(self, population, offspring):
        population.extend(offspring)
        combined_population = population
        return sorted(combined_population, key=lambda x: self.fitness(x), reverse=True)[:self.population_size]



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
    mat = np.array([[0, 8], [8, 0]])
    
    mat = np.linalg.inv(mat)
    nominal_bw = calcualte_nominal_block_bandwidth(tower_bandwidth, block_population, tower_covered_blocks_population)
    diff = np.array(block_coordinates) - np.array(tower_coordinates)
    return np.exp(-0.5 * np.matmul(diff, np.matmul(mat, np.transpose(diff)))) * nominal_bw
