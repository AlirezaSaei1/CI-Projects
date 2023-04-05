import json
import pandas as pd
from config import Configurations


def read_population(file):
    population_data = pd.read_csv(file, sep=',')
    return population_data.to_numpy()


def read_config(file):
    with open(file) as f:
        json_data = json.load(f)
    return json_data


if __name__ == '__main__':
    json_data = read_config('problem_config.txt')
    config = Configurations(json_data)
    print(str(config))

    population = read_population('blocks_population.txt')
    print(population)