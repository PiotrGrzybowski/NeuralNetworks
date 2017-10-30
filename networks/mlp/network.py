import numpy as np

from mlp.activation_functions import get_activation_derivative


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
    def __init__(self, input_layer, units, activation_function, initializer):
        super().__init__(units)
        self.input_layer = input_layer
        self.units = units
        self.activation_function = activation_function
        self.activation_derivative = get_activation_derivative(activation_function)
        self.initializer = initializer
        self.weights = self.initialize_weights(initializer)
        self.biases = self.initialize_bias(initializer)
        self.activations = None
        self.outputs = None
        self.derivative_outputs = None

    def initialize_weights(self, initializer):
        return initializer.initialize_weights(self.units, self.input_layer)

    def initialize_bias(self, initializer):
        return initializer.initialize_bias(self.units)

    def calculate_output(self):
        self.activations = np.dot(self.weights, self.input_layer.outputs) + self.biases
        self.outputs = self.activation_function(self.activations)
        self.derivative_outputs = self.activation_derivative(self.activations)

    @property
    def inputs(self):
        return self.input_layer.outputs


class Network:
    def __init__(self, layers):
        self.layers = layers

    def load_inputs(self, inputs):
        self.layers[0].load_inputs(inputs)

    def propagate_forward(self):
        for layer in self.layers:
            layer.calculate_output()

    def get_layer(self, i):
        return self.layers[i]

    def is_last_layer(self, index):
        return index == len(self.layers) - 1

    def get_weights_of_layer(self, i):
        return self.layers[i].weights

    @property
    def last_layer(self):
        return self.layers[-1]

    @property
    def outputs(self):
        return self.last_layer.outputs

    @property
    def activations(self):
        return self.last_layer.activations

    @property
    def number_of_layers(self):
        return len(self.layers)
