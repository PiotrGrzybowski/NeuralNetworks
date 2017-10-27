import numpy as np


class Layer:
    def __init__(self, units):
        self.units = units

    def calculate_output(self):
        raise NotImplementedError("Should have implemented this!")


class Input(Layer):
    def __init__(self, units):
        super().__init__(units)
        self.inputs = None
        self.outputs = None

    def load_inputs(self, inputs):
        self.inputs = inputs

    def calculate_output(self):
        self.outputs = self.inputs
        return self.outputs


class Dense(Layer):
    def __init__(self, input_layer, units, activation, initializer):
        super().__init__(units)
        self.input_layer = input_layer
        self.units = units
        self.activation = activation
        self.initializer = initializer
        self.weights = self.initialize_weights(initializer)
        self.bias = self.initialize_bias(initializer)
        self.outputs = None
        self.neuron_activations = None

    def initialize_weights(self, initializer):
        return initializer.initialize_weights(self.units, self.input_layer)

    def initialize_bias(self, initializer):
        return initializer.initialize_bias(self.units)

    def calculate_output(self):
        self.outputs = self.activation(np.dot(self.weights, self.input_layer.outputs))
        return self.outputs


class Network:
    def __init__(self, layers):
        self.layers = layers
        self.outputs = None

    def load_inputs(self, inputs):
        self.layers[0].load_inputs(inputs)

    def propagate_forward(self):
        for layer in self.layers:
            layer.calculate_output()
        self.outputs = self.layers[-1].outputs
        return self.outputs





