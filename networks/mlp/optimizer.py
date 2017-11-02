import numpy as np
from random import shuffle
from collections import namedtuple
from mlp.utils import generate_mini_batches

Stats = namedtuple('Stats',
                   'training_cost training_accuracy validation_cost validation_accuracy test_cost test_accuracy')


class Optimizer:
    def __init__(self, epochs, learning_rate, loss_function, mini_batch, verbose):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.batch_size = mini_batch
        self.epoch_stats = []
        self.verbose = verbose

    def train(self, network, training_data, validation_data, test_data):
        for epoch in range(self.epochs):
            shuffle(training_data)

            for mini_batch in generate_mini_batches(training_data, self.batch_size):
                network.load_batch(mini_batch)
                network.propagate_forward()
                biases_error, weights_error = self.mini_batch_error(*network.propagate_backward(self.loss_function))

                network.update_weights(self.calculate_gradient(weights_error))
                network.update_biases(self.calculate_gradient(biases_error))

            self.epoch_stats.append(self.calculate_epoch_stats(network, training_data, validation_data, test_data))

            if self.verbose:
                self.print_log(epoch, training_data, validation_data, test_data)

    def print_log(self, epoch, training_data, validation_data, test_data):
        print("Cost = {}".format(self.epoch_stats[-1].training_cost))
        print("Accuracy: {} / {}".format(int(self.epoch_stats[-1].training_accuracy), len(training_data)))

        print("Cost = {}".format(self.epoch_stats[-1].validation_cost))
        print("Accuracy: {} / {}".format(int(self.epoch_stats[-1].validation_accuracy), len(validation_data)))

        print("Cost = {}".format(epoch, self.epoch_stats[-1].test_cost))
        print("Accuracy: {} / {}\n".format(int(self.epoch_stats[-1].test_accuracy), len(test_data)))

    def mini_batch_error(self, biases_error, weights_error):
        return self.mean_biases_by_samples(biases_error), weights_error

    @staticmethod
    def mean_biases_by_samples(biases_error):
        return [np.expand_dims(np.mean(error, axis=1), axis=1) for error in biases_error]

    def calculate_epoch_stats(self, network, training_data, validation_data, test_data):
        validation_cost = validation_accuracy = test_cost = test_accuracy = None
        training_cost, training_accuracy = self.calculate_cost_and_accuracy(training_data, network)

        if validation_data is not None:
            validation_cost, validation_accuracy = self.calculate_cost_and_accuracy(validation_data, network)

        if test_data is not None:
            test_cost, test_accuracy = self.calculate_cost_and_accuracy(test_data, network)

        return Stats(training_cost, training_accuracy, validation_cost, validation_accuracy, test_cost, test_accuracy)

    def calculate_cost_and_accuracy(self, data, network):
        return np.sum([[network.calculate_cost(sample, self.loss_function) / len(data),
                        network.correct_prediction] for sample in data], axis=0)

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
