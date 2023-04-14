import json
import numpy as np
from config import Configurations as cnf
from GA import GeneticAlgorithm
from matplotlib import pyplot as plt


def read_population(file):
    population_data = []
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        population_data.append(list(map(int, line.split(','))))
    
    return np.array(population_data)


def read_config(file):
    with open(file) as f:
        json_data = json.load(f)
    return json_data


if __name__ == '__main__':
    cnf.load_config('data/problem_config.txt')
    print(cnf.show_configurations())

    population = read_population('data/blocks_population.txt')
    
    shape = population.shape
    ga = GeneticAlgorithm(population, 100, max(cnf.levels)*10000, shape[0], shape[1], 50, mutation_probability=0.1, crossover_probability=0.9)

    answers = list()

    for _ in range(100):
        selected_parents = ga.selection(ga.population, len(ga.population))
        
        recombined_genotypes = ga.one_point_crossover(selected_parents)

        mutated_genotypes = ga.mutate_population(recombined_genotypes)

        ga.population = ga.replace_population(ga.population, mutated_genotypes)

        answers.append(np.mean([ga.fitness(genotype) for genotype in ga.population]))

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.plot(answers)
    plt.show()