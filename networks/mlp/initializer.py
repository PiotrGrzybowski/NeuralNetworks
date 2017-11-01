import numpy as np


class Initializer:
    def initialize_weights(self, inputs, units):
        raise NotImplementedError("Should have implemented this!")

    def initialize_bias(self, inputs):
        raise NotImplementedError("Should have implemented this!")


class GaussianNormal(Initializer):
    def initialize_weights(self, units, inputs):
        return np.random.randn(units, inputs.units)

    def initialize_bias(self, units):
        return np.random.randn(units, 1)


class GaussianNormalScaled(GaussianNormal):
    def initialize_weights(self, units, inputs):
        return np.random.randn(units, inputs.units) / np.sqrt(inputs.units)
