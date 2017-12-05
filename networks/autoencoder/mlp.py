import numpy as np

from autoencoder.activation_functions import sigmoid, sigmoid_derivative
from random import shuffle


class MeanSquaredError:
    @staticmethod
    def calculate_cost(expected, outputs):
        return 0.5 * np.power(np.linalg.norm(expected - outputs), 2)

    @staticmethod
    def calculate_cost_gradient(expected, outputs):
        return outputs - expected


class Network:
    def __init__(self, hidden):
        self.input = None
        self.hidden_weights = np.random.randn(hidden, 784) / np.sqrt(784)
        self.output_weights = np.random.randn(784, hidden) / np.sqrt(hidden)

        self.hidden_bias = np.ones(shape=(hidden, 1))
        self.output_bias = np.ones(shape=(784, 1))

        self.hidden_net = None
        self.output_net = None

        self.hidden_activation = None
        self.output_activation = None

        self.hidden_activation_derivative = None
        self.output_activation_derivative = None

        self.hidden_error = None
        self.output_error = None

        self.output_weights_gradient = None
        self.hidden_weights_gradient = None

        self.output_bias_gradient = None
        self.hidden_bias_gradient = None

        self.loss_function = MeanSquaredError()

    def load_input(self, data):
        self.input = data

    def feed_forward(self):
        self.hidden_net = np.dot(self.hidden_weights, self.input)
        self.hidden_activation = sigmoid(self.hidden_net)
        self.hidden_activation_derivative = sigmoid_derivative(self.hidden_net)
        self.output_net = np.dot(self.output_weights, self.hidden_activation)
        self.output_activation = sigmoid(self.output_net)
        self.output_activation_derivative = sigmoid_derivative(self.output_net)

    def propagate_backward(self):
        self.output_error = np.multiply(self.loss_function.calculate_cost_gradient(self.input, self.output_activation),
                                        self.output_activation_derivative)

        self.hidden_error = np.multiply(np.dot(self.output_weights.T, self.output_error),
                                        self.hidden_activation_derivative)

    def calculate_parameters_gradients(self):
        self.output_weights_gradient = np.dot(self.output_error, self.hidden_activation.T)
        self.hidden_weights_gradient = np.dot(self.hidden_error, self.input.T)

        self.output_bias_gradient = np.expand_dims(np.sum(self.output_error, axis=1), axis=1)
        self.hidden_bias_gradient = np.expand_dims(np.sum(self.hidden_error, axis=1), axis=1)

    def update_parameters(self, eta, batch_size, l2):
        self.output_weights = (1 - eta * l2) * self.output_weights - eta / batch_size * self.output_weights_gradient
        self.hidden_weights = (1 - eta * l2) * self.hidden_weights - eta / batch_size * self.hidden_weights_gradient

        self.output_bias -= eta / batch_size * self.output_bias_gradient
        self.hidden_bias -= eta / batch_size * self.hidden_bias_gradient

    def train(self, training_data, learning_rate, epochs, batch_size, l2, test_data=None):
        for epoch in range(epochs):
            training_data = self.shuffle_data(training_data)
            print("Epoch: {}".format(epoch))

            for batch in self.batch_generator(training_data, batch_size):
                self.load_input(batch)
                self.feed_forward()
                self.propagate_backward()
                self.calculate_parameters_gradients()
                self.update_parameters(learning_rate, batch_size, l2)

            print("  Training cost = {}".format(self.calculate_cost(training_data, l2)))
            print("  Test     cost = {}".format(self.calculate_cost(test_data, l2)))

    def calculate_cost(self, data, l2):
        self.load_input(data)
        self.feed_forward()

        cost = self.loss_function.calculate_cost(data, self.output_activation) / data.shape[1]
        cost += l2 / (2 * data.shape[1]) * np.power(np.linalg.norm(self.output_weights), 2)
        cost += l2 / (2 * data.shape[1]) * np.power(np.linalg.norm(self.hidden_weights), 2)

        return cost

    @staticmethod
    def batch_generator(data, batch_size):
        for i in np.arange(0, data.shape[1] - batch_size, batch_size):
            yield data[:, i: i + batch_size]

    @staticmethod
    def shuffle_data(data):
        data = data.T
        shuffle(data)
        return data.T
