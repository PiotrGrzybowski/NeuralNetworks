import numpy as np
from random import shuffle


class Optimizer:
    def __init__(self, epochs, learning_rate, loss_function, mini_batch):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.mini_batch = mini_batch

    def train(self, network, training_data):
        for epoch in range(self.epochs):
            # shuffle(training_data)

            for i in np.arange(0, len(training_data) - self.mini_batch, self.mini_batch):
                network.load_batch(training_data[i: i + self.mini_batch])
                network.propagate_forward()
                biases_error, weights_error = network.propagate_backward(self.loss_function)
                # print("Error = {}".format(np.sum(biases_error[0])))
                gradient = self.calculate_gradient(weights_error)
                network.update_weights(gradient)
                calculate_gradient = self.calculate_gradient(biases_error)
                network.update_biases(calculate_gradient)

            epoch_cost, epoch_accuracy = self.calculate_epoch_cost(network, training_data)

            print("Epoch: {}, cost = {}".format(epoch, epoch_cost))
            print("Accuracy: {} / {}".format(epoch_accuracy, len(training_data)))

    @staticmethod
    def mini_batch_error(biases_error, weights_error):
        biases_error = [np.expand_dims(np.mean(error, axis=1), axis=1) for error in biases_error]
        weights_error = [np.expand_dims(np.mean(error, axis=1), axis=1) for error in weights_error]
        return biases_error, weights_error

    def calculate_epoch_cost(self, network, data):
        cost = 0.0
        accuracy = 0

        for sample in data:
            cost += network.calculate_cost(sample, self.loss_function) / len(data)
            accuracy += network.correct_prediction
        return cost, accuracy

    def calculate_gradient(self, error):
        raise NotImplementedError("Should have implemented this!")


class GradientDescent(Optimizer):
    def calculate_gradient(self, error):
        return np.multiply(self.learning_rate, error)


class MomentumGradientDescent(GradientDescent):
    def __init__(self, epochs, learning_rate, loss_function, momentum):
        super().__init__(epochs, learning_rate, loss_function)
        self.momentum = momentum
        self.previous_gradient = None

    def calculate_gradient(self, error):
        if self.previous_gradient is None:
            self.previous_gradient = super().calculate_gradient(error)
            return self.previous_gradient
        else:
            return self.previous_gradient * self.momentum + super().calculate_gradient(error)

