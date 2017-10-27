import numpy as np


class Optimizer:
    def __init__(self, epochs, learning_rate, loss_function):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.layer_errors = None
        self.epoch_train_cost = 0.0

    def train(self, network, training_data):
        raise NotImplementedError("Should have implemented this!")

    def calculate_hidden_layer_error(self, weights):
        return np.dot(weights.T, self.layer_errors[-1])

    def calculate_last_layer_error(self, layer):
        return self.loss_function.calculate_cost_gradient(
            outputs=layer.activations,
            derivative_outputs=layer.derivative_outputs,
            expected_value=self.layer_errors[-1])

    def calculate_layer_error(self, index, network):
        return self.calculate_last_layer_error(network.get_layer(index)) if network.is_last_layer(index) \
            else self.calculate_hidden_layer_error(network.get_weights_of_layer(index + 1))


class GradientDescent(Optimizer):
    def train(self, network, training_data):
        for epoch in range(self.epochs):
            self.epoch_train_cost = 0.0
            for sample in training_data:
                network.load_inputs(sample[0])
                network.propagate_forward()
                self.layer_errors = [sample[1]]

                for index in reversed(range(1, network.number_of_layers)):
                    layer = network.get_layer(index)
                    layer_error = self.calculate_layer_error(index, network)
                    self.layer_errors.append(layer_error)

                    weights_gradient = np.dot(layer_error, layer.inputs.T)
                    bias_gradient = layer_error

                    layer.weights = layer.weights - self.learning_rate * weights_gradient
                    layer.bias = layer.bias - self.learning_rate * bias_gradient

                self.epoch_train_cost += self.loss_function.calculate_cost(sample[1], network.layers[-1].outputs)
            print("Epoch: {}, training_cost = {}".format(epoch, self.epoch_train_cost))

class MomentumGradientDescent(GradientDescent):
    pass
