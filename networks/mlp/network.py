import numpy as np


class Layer:
    def __init__(self, units):
        self.units = units


class Input(Layer):
    def __init__(self, units):
        super().__init__(units)
        self.inputs = None

    def load_inputs(self, inputs):
        self.inputs = inputs


class Dense(Layer):
    def __init__(self, inputs, units, activation, initializer):
        super().__init__(units)
        self.inputs = inputs
        self.units = units
        self.activation = activation
        self.initializer = initializer
        self.weights = self.initialize_weights(initializer)
        self.bias = self.initialize_bias(initializer)

    def initialize_weights(self, initializer):
        return initializer.initialize_weights(self.units, self.inputs)

    def initialize_bias(self, initializer):
        return initializer.initialize_bias(self.units)


class Network:
    def __init__(self, layers):
        self.layers = layers

