import numpy as np


class Network:
    def __init__(self, layers, activations):
        self.sizes_of_layers = layers
        self.number_of_layers = len(layers)
        self.weights = self.weights_initialize()

    def weights_initialize(self):
        return [np.random.random((y, x)) for x, y in zip(self.sizes_of_layers[:-1], self.sizes_of_layers[1:])]
