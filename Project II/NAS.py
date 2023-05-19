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
        self.search_space = {
            'num_layers': layer_numbers,
            'layer_sizes': layer_sizes_all,
            'activations': activations_all,
            'learning_rate': learning_rates_all,
            'feature_extractor': feature_extractors
        }

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

    def selection(self):
        pass

    def crossover(self):
        pass

    def mutation(self):
        pass

    def replacement(self):
        pass

    def run(self, generations=10):
        for generation in range(generations):
            print(f'--- Generation {generation + 1} ---')

            # TODO: selection
            parents = []

            # TODO: crossover
            offspring = []

            # TODO: mutation
            offspring = []

            # TODO: eval
            fitness_scores = [self.evaluate(network) for network in offspring]

            # TODO: replacement
            population = []

        report = self.evaluate(population[0])
        print(f'Report: {report}')
