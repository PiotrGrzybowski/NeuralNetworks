import numpy as np


class Initializer:
    def initialize_weights(self, inputs, units):
        raise NotImplementedError("Should have implemented this!")

    def initialize_bias(self, inputs):
        raise NotImplementedError("Should have implemented this!")


class RandomInitializer(Initializer):
    def __init__(self, weights_range, bias_range):
        self.weight_range = weights_range
        self.bias_range = bias_range

    def initialize_weights(self, units, inputs):
        return np.random.uniform(*self.weight_range, (units, inputs.units))

    def initialize_bias(self, units):
        return np.random.uniform(*self.bias_range, units)
