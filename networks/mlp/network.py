import numpy as np

from mlp.activation_functions import get_activation_derivative


class Layer:
    def __init__(self, units):
        self.units = units

    def calculate_output(self):
        raise NotImplementedError("Should have implemented this!")


class InputLayer(Layer):
    def __init__(self, units):
        super().__init__(units)
        self.inputs = None
        self.outputs = None

    def load_inputs(self, inputs):
        self.inputs = inputs

    def calculate_output(self):
        self.outputs = self.inputs
        return self.outputs


class DenseLayer(Layer):
    def __init__(self, input_layer, units, activation_function, initializer):
        super().__init__(units)
        self.input_layer = input_layer
        self.units = units
        self.activation_function = activation_function
        self.activation_derivative = get_activation_derivative(activation_function)
        self.initializer = initializer
        self.weights = self.initialize_weights(initializer)
        self.biases = self.initialize_bias(initializer)
        self.error = None
        self.next_layer = None
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

    def calculate_error(self, loss_function):
        self.error = np.dot(self.next_layer.weights.T, self.next_layer.error) * self.derivative_outputs
        return self.error

    @property
    def inputs(self):
        return self.input_layer.outputs


class OutputLayer(DenseLayer):
    def __init__(self, input_layer, units, activation_function, initializer):
        super().__init__(input_layer, units, activation_function, initializer)
        self.expected_value = None
        self.predicted_class = None

    def calculate_error(self, loss_function):
        self.error = loss_function.calculate_cost_gradient(self.expected_value, self.outputs, self.derivative_outputs)
        return self.error

    def calculate_cost(self, loss_function):
        return loss_function.calculate_cost(self.expected_value, self.outputs)

    def calculate_output(self):
        super().calculate_output()
        self.predicted_class = np.argmax(self.outputs)


class Network:
    def __init__(self, layers):
        self.layers = layers
        self.connect_layers()

    def connect_layers(self):
        for i in range(len(self.layers) - 1):
            self.layers[i].next_layer = self.layers[i + 1]

    def load_sample(self, inputs):
        self.layers[0].load_inputs(inputs[0])
        self.last_layer.expected_value = inputs[1]
        self.last_layer.expected_class = np.argmax(inputs[1])

    def propagate_forward(self):
        for layer in self.layers:
            layer.calculate_output()

    def propagate_backward(self, sample, loss_function):
        self.load_sample(sample)
        self.propagate_forward()
        biases_error = [layer.calculate_error(loss_function) for layer in reversed(self.layers[1:])]
        weights_error = [np.dot(error, layer.inputs.T) for error, layer in zip(biases_error, reversed(self.layers[1:]))]

        return weights_error, biases_error

    def calculate_cost(self, sample, loss_function):
        self.load_sample(sample)
        self.propagate_forward()

        return loss_function.calculate_cost(self.last_layer.expected_value, self.last_layer.outputs)

    def update_weights(self, weights_gradient):
        for layer, gradient in zip(reversed(self.layers[1:]), weights_gradient):
            layer.weights -= gradient

    def update_biases(self, biases_gradient):
        for layer, gradient in zip(reversed(self.layers[1:]), biases_gradient):
            layer.biases -= gradient

    @property
    def last_layer(self):
        return self.layers[-1]

    @property
    def number_of_layers(self):
        return len(self.layers)

    @property
    def correct_prediction(self):
        return self.last_layer.expected_class == self.last_layer.predicted_class
