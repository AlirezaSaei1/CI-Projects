import numpy as np


class SOM:
    def __init__(self, input_size=512, output_size=10):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(*output_size, input_size)
        
