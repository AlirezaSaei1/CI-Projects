import random

activations = ['relu', 'sigmoid']

class NeuroEvolution:
    def __init__(self, input_size, output_size, pop_size=50) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.pop_size = pop_size
        self.population = []
        self.fitness_scores = []

        # Initialize population
        for i in range(pop_size):
            network = self.random_network()
            self.population.append(network)


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

            # Selection
            parents = self.selection()

            # Crossover
            offspring = self.crossover(parents)

            # Mutation
            offspring = self.mutation(offspring)

            # Evaluation
            fitness_scores = [self.evaluate(network) for network in offspring]

            # Replacement
            self.replacement(offspring, fitness_scores)

            # Print progress
            print(f"Generation {generation+1}: Best fitness = {max(self.fitness_scores)}")