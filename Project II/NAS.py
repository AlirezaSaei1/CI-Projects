import random

activations = ['relu', 'sigmoid']
learning_rates = [0.1, 0.01, 0.001, 0.0001]

class NAS:
    def __init__(self) -> None:
        pass


    def random_network(self):
        # setup
        layer_size = random.choice(range(128))
        activation = random.choice(activations)
        learning_rates = random.choice(learning_rates)

        # TODO: neural network construction
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
            print(f'--- Generation {generation+1} ---')

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
