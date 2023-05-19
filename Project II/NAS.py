import random
from NeuralNetwork import Dense
import tensorflow as tf
layer_numbers = [1, 2, 3]
layer_sizes_all = [10, 20, 30]
activations_all = ['ReLU', 'Sigmoid']
learning_rates_all = [0.1, 0.01, 0.001, 0.0001]
feature_extractors = ['ResNet18', 'ResNet34', 'Vgg11']

class NAS:
    def __init__(self) -> None:
        pass

    def random_network(self):
        # setup
        feature_extractor = random.choice(feature_extractors)
        num_layers = random.choice(layer_numbers)
        layer_sizes = [random.choice(layer_sizes_all) for _ in range(num_layers)]
        activations = [random.choice(activations_all) for _ in range(num_layers)]
        learning_rate = random.choice(learning_rates_all)

        input_size = 512
        output_size = 10
        layers = []

        network = None
        return network

    def evaluate(self, network):
        pass

    def selection(self, population, k=3):
        selected = []
        while len(selected) < len(population):
            individuals = random.sample(population, k)
            print(individuals)
            # min
            fittest = min(individuals, key=lambda network: self.evaluate(network))
            selected.append(fittest)

        return selected

    def crossover(self):
        pass

    def mutation(self):
        pass

    def replacement(self, population, offspring):
        combined_population = population + offspring
        # min sort
        sorted_networks = sorted(combined_population, key=lambda x: self.evaluate(x))
        population = sorted_networks[:len(population)]

        return population

    def run(self, generations=10, population_size=10):

        population = [self.random_network() for _ in range(population_size)]

        for generation in range(generations):
            print(f'--- Generation {generation + 1} ---')

            parents = self.selection(population)

            offspring = self.crossover(parents)

            offspring = self.mutation(offspring)

            population = self.replacement(population, offspring)

        report = self.evaluate(population[0])
        print(f'Report: {report}')
