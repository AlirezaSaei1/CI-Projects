import json
import pandas as pd
from config import Configurations as cnf
from GA import GeneticAlgorithm



def read_population(file):
    population_data = pd.read_csv(file, sep=',', header=0)
    return population_data.to_numpy()


def read_config(file):
    with open(file) as f:
        json_data = json.load(f)
    return json_data


if __name__ == '__main__':
    cnf.load_config('data/problem_config.txt')
    print(cnf.show_configurations())

    population = read_population('data/blocks_population.txt')
    #print(population)

    shape = population.shape
    ga = GeneticAlgorithm(10, max(cnf.levels)*5, shape[0], shape[1], 30)
    print('Population size:', len(ga.population))
    print(ga.population.towers)