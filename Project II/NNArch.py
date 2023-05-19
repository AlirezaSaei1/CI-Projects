import itertools


class NNArch:
    id_iter = itertools.count()

    def __init__(self, num, layers, af, feature_ext, lr):
        self.id = 0
        self.num_layers = num
        self.layers = layers
        self.activation_func = af
        self.feature_ext = feature_ext
        self.learning_rate = lr
