import numpy as np

from mlp.activation_functions import get_activation_derivative


class Layer:
    def __init__(self, units):
        self.units = units
        self.outputs = None
        self.inputs = None

    def load_inputs(self, inputs):
        self.inputs = inputs

    def calculate_output(self):
        raise NotImplementedError("Should have implemented this!")


class Input(Layer):
    def __init__(self, units):
        super().__init__(units)

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
        dot = np.dot(self.next_layer.weights.T, self.next_layer.error)
        self.error = dot * self.derivative_outputs
        return self.error

    @property
    def input(self):
        return self.input_layer.outputs


class Dropout(Layer):
    def __init__(self, units, probability):
        super().__init__(units)
        self.probability = probability

    def calculate_output(self):
        pass


class Output(Dense):
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

    def load_batch(self, samples):
        inputs = np.zeros(shape=(70, len(samples)))
        expected_values = np.zeros(shape=(10, len(samples)))
        expected_class = []

        for i in range(len(samples)):
            inputs[:, i] = samples[i][0].squeeze()
            expected_values[:, i] = samples[i][1].squeeze()
            expected_class.append(np.argmax(samples[i][1]))

        self.layers[0].load_inputs(inputs)
        self.last_layer.expected_value = expected_values
        self.last_layer.expected_class = expected_class

    def propagate_forward(self):
        for layer in self.layers:
            layer.calculate_output()

    def propagate_backward(self, loss_function):
        biases_error = []
        weights_error = []

        for layer in self.reversed_layers:
            biases_error.append(layer.calculate_error(loss_function))
            weights_error.append(np.dot(biases_error[-1], layer.input.T))

        return biases_error, weights_error

    def calculate_cost(self, sample, loss_function):
        self.load_sample(sample)
        self.propagate_forward()

        return loss_function.calculate_cost(self.last_layer.expected_value, self.last_layer.outputs)

    def update_weights(self, weights_gradient):
        for layer, gradient in zip(self.reversed_layers, weights_gradient):
            layer.weights -= gradient

    def update_biases(self, biases_gradient):
        for layer, gradient in zip(self.reversed_layers, biases_gradient):
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

    @property
    def reversed_layers(self):
        return reversed(self.layers[1:])

