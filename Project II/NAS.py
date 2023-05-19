import random
import torchvision.models as models
import torch.nn as nn
from NeuralNetwork import MLP
from main import x_train_features, y_train

class NAS:
    def __init__(self) -> None:
        self.search_space = {
            'num_layers': [1, 2, 3],
            'layer_sizes': [10, 20, 30],
            'activations': ['ReLU', 'Sigmoid'],
            'feature_extractor': [
                nn.Sequential(*list(models.resnet18().children())[:-1]),
                nn.Sequential(*list(models.resnet34().children())[:-1]),
                nn.Sequential(*list(models.vgg11().features.children())[:-1])
            ]
        }

    def random_network(self):
        # setup
        feature_extractor = random.choice(self.search_space['feature_extractor'])
        num_layers = random.choice(self.search_space['num_layers'])
        layer_sizes = [random.choice(self.search_space['layer_sizes']) for _ in range(num_layers)]
        activations = [random.choice(self.search_space['activations']) for _ in range(num_layers)]

        input_size = 512
        output_size = 10
        return feature_extractor, MLP(input_size, output_size, layer_sizes, activations, loss_function="cross_entropy", learning_rate=0.001)
        

    def evaluate(self, network: MLP, X=x_train_features, Y=y_train):
        fitness_list = []
        n_samples = X.shape[0]
        indices = list(range(n_samples))
        random.shuffle(indices)
        subsample_size = int(n_samples / 5)
        
        for i in range(5):
            subset_indices = indices[i*subsample_size:(i+1)*subsample_size]
            X_subset = X[subset_indices]
            Y_subset = Y[subset_indices]
            
            y_hat = network.forward(X_subset)
            loss = network.loss.forward(y_hat, Y_subset)
            fitness = 1.0 / (1.0 + loss)
            fitness_list.append(fitness)
        
        avg_fitness = sum(fitness_list) / len(fitness_list)
        return avg_fitness


    def selection(self, population, k=3):
        selected = []
        while len(selected) < len(population):
            individuals = random.sample(population, k)
            print(individuals)
            # min
            fittest = min(individuals, key=lambda network: self.evaluate(network))
            selected.append(fittest)

        return selected


    def crossover(self, parents):
        offspring = []
        for i in range(len(parents)):
            parent1 = parents[i]
            parent2 = parents[(i+1)%len(parents)]

            feature_extractor1, network1 = parent1
            feature_extractor2, network2 = parent2

            if random.random() < 0.5:
                feature_extractor = feature_extractor1
            else:
                feature_extractor = feature_extractor2

            layer_sizes = []
            activations = []

            for j in range(min(len(network1.layer_sizes), len(network2.layer_sizes))):
                if random.random() < 0.5:
                    layer_sizes.append(network1.layer_sizes[j])
                else:
                    layer_sizes.append(network2.layer_sizes[j])

            for j in range(min(len(network1.activations), len(network2.activations))):
                if random.random() < 0.5:
                    activations.append(network1.activations[j])
                else:
                    activations.append(network2.activations[j])

            num_layers = len(layer_sizes)

            input_size = 512
            output_size = 10
            offspring_network = MLP(input_size, output_size, layer_sizes, activations, loss_function="cross_entropy", learning_rate=0.001)
            offspring.append((feature_extractor, offspring_network))

            return offspring


    def mutation(self, offspring, mutation_rate=0.1):
        mutated_offspring = []
        for i in range(len(offspring)):
            feature_extractor, network = offspring[i]

            for j in range(len(network.layer_sizes)):
                if random.random() < mutation_rate:
                    network.layer_sizes[j] = random.choice(self.search_space['layer_sizes'])

            for j in range(len(network.activations)):
                if random.random() < mutation_rate:
                    network.activations[j] = random.choice(self.search_space['activations'])

            input_size = 512
            output_size = 10
            mutated_network = MLP(input_size, output_size, network.layer_sizes, network.activations, loss_function="cross_entropy", learning_rate=0.001)
            mutated_offspring.append((feature_extractor, mutated_network))

        return mutated_offspring
    

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
