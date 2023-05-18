import random

activations = ['ReLU', 'Sigmoid']
learning_rates = [0.1, 0.01, 0.001, 0.0001]

class NAS:
    def __init__(self) -> None:
        pass


    def random_network(self):
        # setup
        feature_extractor = ['ResNet18', 'ResNet34', 'Vgg11']
        num_layers = random.choice([1, 2, 3])
        layer_sizes = [random.choice([10, 20, 30]) for _ in range(num_layers)]
        activations = [random.choice(activations) for _ in range(num_layers)]
        learning_rate = random.choice(learning_rates)

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
