import json
import pandas as pd
from config import Configurations as cnf


def read_population(file):
    population_data = pd.read_csv(file, sep=',', header=0)
    return population_data.to_numpy()


def read_config(file):
    with open(file) as f:
        json_data = json.load(f)
    return json_data


if __name__ == '__main__':
    cnf.load_config('problem_config.txt')
    print(cnf.show_configurations())

    population = read_population('blocks_population.txt')
    print(population)

