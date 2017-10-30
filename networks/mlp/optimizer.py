import numpy as np
from random import shuffle


class Optimizer:
    def __init__(self, epochs, learning_rate, loss_function):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.layer_errors = None
        self.epoch_train_cost = 0.0

    def train(self, network, training_data):
        raise NotImplementedError("Should have implemented this!")

    def calculate_epoch_cost(self, network, data):
        epoch_train_cost = 0.0
        for sample in data:
            network.load_inputs(sample[0])
            network.propagate_forward()
            epoch_train_cost += self.loss_function.calculate_cost(outputs=network.last_layer.outputs, expected_value=sample[1]) / len(data)
        return epoch_train_cost

    def calculate_hidden_layer_error(self, network, index):
        current_layer = network.get_layer(index)
        next_layer = network.get_layer(index + 1)
        t = next_layer.weights.T
        errors_ = self.layer_errors[-1]
        outputs = current_layer.derivative_outputs
        return np.dot(t, errors_) * outputs

    def calculate_last_layer_error(self, layer):
        return self.loss_function.calculate_cost_gradient(
            outputs=layer.outputs,
            derivative_outputs=layer.derivative_outputs,
            expected_value=self.layer_errors[-1])

    def calculate_last_layer_cost(self, layer):
        return self.loss_function.calculate_cost(
            outputs=layer.outputs,
            expected_value=self.layer_errors[-1])

    def calculate_layer_error(self, index, network):
        return self.calculate_last_layer_error(network.get_layer(index)) if network.is_last_layer(index) \
            else self.calculate_hidden_layer_error(network, index)


class GradientDescent(Optimizer):
    def train(self, network, training_data):
        for epoch in range(self.epochs):
            # shuffle(training_data)

            for sample in training_data:
                network.load_inputs(sample[0])
                network.propagate_forward()
                self.layer_errors = [sample[1]]
                weights_gradients = []
                biases_gradients = []

                for index in reversed(range(1, network.number_of_layers)):
                    layer = network.get_layer(index)
                    layer_error = self.calculate_layer_error(index, network)
                    self.layer_errors.append(layer_error)

                    weights_gradients.append(np.dot(layer_error, layer.inputs.T))
                    biases_gradients.append(layer_error)

                for layer, weights_change, biases_change in zip(reversed(network.layers[1:]), weights_gradients, biases_gradients):
                    layer.weights = layer.weights - self.learning_rate * weights_change
                    layer.biases = layer.biases - self.learning_rate * biases_change

            epoch_cost = self.calculate_epoch_cost(network, training_data)
            print("Epoch: {}, error = {}".format(epoch, epoch_cost))


class MomentumGradientDescent(GradientDescent):
    pass
