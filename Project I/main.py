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
    generations = 200
    answers_all = []
    mutation_probab = 0.9
    crossover_probab = 0.9

    iterate = 0

    for _ in range(5):
        answers = []
        ga = GeneticAlgorithm(population, 100, max(cnf.levels)*10000, shape[0], shape[1], 50, mutation_probability=mutation_probab, crossover_probability=crossover_probab)

        for _ in range(generations):
            selected_parents = ga.selection(ga.population, len(ga.population))
            
            recombined_genotypes = ga.one_point_crossover(selected_parents)

            mutated_genotypes = ga.mutate_population(recombined_genotypes)

            ga.population = ga.replace_population(ga.population, mutated_genotypes)

            fitness_scores = np.array([ga.fitness(genotype) for genotype in ga.population])
            mean_fitness = np.mean(fitness_scores)
            answers.append(mean_fitness)
        print('i')

    mean_answers_all = np.array(answers_all)
    mean_fitness_scores_mean = np.mean(mean_answers_all, axis=0)

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.plot(answers)
    plt.title(f'P_mutation = {mutation_probab} and P_crossover = {crossover_probab}')
    plt.show()