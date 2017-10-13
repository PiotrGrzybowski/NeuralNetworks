import numpy as np
import matplotlib.pyplot as plt
from drawnow import drawnow
import time

from neurons.activations import bipolar

LEAST_MEAN_ERROR = 'least_mean_square'
DISCRETE_ERROR = 'discrete'


class Optimizer:
    def __init__(self, loss, learning_rate, epochs, stop_error):
        self.loss = loss
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.validation_error = 0
        self.continue_learning = True
        self.stop_error = stop_error

    def train(self, neuron, training_data, validation_data, visualize):
        if visualize:
            self.initialize_plotting(neuron, training_data, validation_data)
        epoch = 0
        while epoch < self.epochs and self.continue_learning:
            training_error = 0
            validation_error = 0
            np.random.shuffle(training_data)

            if visualize:
                self.plot_output(neuron)

            for sample in training_data:
                input_values, expected_value = Optimizer.split_sample(neuron, sample)
                error, training_error = self.calculate_error(expected_value, input_values, neuron, training_error)

                self.update_weights(error, input_values, neuron)
                self.update_bias(error, neuron)

            validation_error = self.calculate_validation_error(neuron, validation_data, validation_error)

            epoch_error_training = self.calculate_epoch_error(training_error, len(training_data))
            epoch_error_validation = self.calculate_epoch_error(validation_error, len(training_data))

            print("Training Error = {}".format(epoch_error_training))
            print("Validation Error = {}".format(epoch_error_validation))

            self.check_stop_condition(epoch_error_training)

            epoch += 1

        if visualize:
            input("Press Enter to continue...")

    def check_stop_condition(self, epoch_error_training):
        if self.loss == DISCRETE_ERROR:
            if epoch_error_training == self.stop_error:
                self.continue_learning = False
        else:
            if epoch_error_training < self.stop_error:
                self.continue_learning = False

    def calculate_validation_error(self, neuron, validation_data, validation_error):
        for sample in validation_data:
            input_values, expected_value = Optimizer.split_sample(neuron, sample)
            error, validation_error = self.calculate_error(expected_value, input_values, neuron, validation_error)
        return validation_error

    @staticmethod
    def split_sample(neuron, sample):
        return sample[:neuron.get_number_of_inputs], sample[-1]

    def update_bias(self, error, neuron):
        neuron.bias += self.learning_rate * error

    def update_weights(self, error, input_values, neuron):
        for i in range(neuron.get_number_of_inputs):
            neuron.weights[i] += self.learning_rate * error * input_values[i]

    def calculate_error(self, expected_value, input_values, neuron, cumulative_error):
        if self.loss == DISCRETE_ERROR:
            error = expected_value - neuron.predict(input_values)
            cumulative_error += error
        elif self.loss == LEAST_MEAN_ERROR:
            error = 2 * (expected_value - neuron.raw_output(input_values))
            cumulative_error += np.power(error, 2)
        else:
            raise ValueError('Wrong loss function')
        return error, cumulative_error

    def calculate_epoch_error(self, error, number_of_samples):
        return error if self.loss == DISCRETE_ERROR else error / number_of_samples

    def update_line(self, new_x, new_y):
        self.plot.set_xdata(new_x)
        self.plot.set_ydata(new_y)
        plt.draw()

    @staticmethod
    def get_x_axis(neuron):
        return [-1, 1] if neuron.get_activation == bipolar else [0, 1]

    @staticmethod
    def get_y_axis(neuron):
        x = Optimizer.get_x_axis(neuron)
        return [(-neuron.bias - (neuron.weights[0]) * x[0]) / neuron.weights[1],
                (-neuron.bias - neuron.weights[0]) / neuron.weights[1]]

    def draw_fig(self):
        plt.xlim((-2.1, 2.1))
        plt.ylim((-2.1, 2.1))

        for sample in self.validation_data:
            result = self.neuron.predict(sample)
            if result == 1:
                plt.plot(sample[0], sample[1], 'or')
            else:
                plt.plot(sample[0], sample[1], 'ob')
        plt.plot(self.x, self.y)

    def plot_output(self, neuron):
        self.x = self.get_x_axis(neuron)
        self.y = self.get_y_axis(neuron)
        drawnow(self.draw_fig)

    def initialize_plotting(self, neuron, training_data, validation_data):
        self.plot, = plt.plot([], [])
        plt.ion()
        self.training_data = training_data
        self.validation_data = validation_data
        self.neuron = neuron
