import json
from tower import Tower

def read_population(file):
    pass

def read_config(file):
    pass

if __name__ == '__main__':
    json_data = read_config('problem_config.txt')
    tower = Tower(json_data)

    population = read_population('blocks_population.txt')