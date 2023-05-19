import random
from NeuralNetwork import Dense
import torchvision.models as models

class NAS:
    def __init__(self) -> None:
        self.search_space = {
            'num_layers': [1, 2, 3],
            'layer_sizes': [64, 128, 256, 512],
            'activations': ['ReLU', 'Sigmoid'],
            'feature_extractor': [models.resnet18(), models.resnet34(), models.vgg11()]
        }

    def random_network(self):
        # setup
        feature_extractor = random.choice(self.search_space['feature_extractor'])
        num_layers = random.choice(self.search_space['num_layers'])
        layer_sizes = [random.choice(self.search_space['layer_sizes']) for _ in range(num_layers)]
        activations = [random.choice(self.search_space['activations']) for _ in range(num_layers)]

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
