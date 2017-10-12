import numpy as np

from neurons.activations import UNIPOLAR, BIPOLAR
from neurons.data_generator import LogicalFunctionsGenerator
from neurons.perceptron import Perceptron
from matplotlib import pyplot as plt


class Optimizer:
    @staticmethod
    def error(expected, predicted):
        return expected - predicted

    @staticmethod
    def mean_squared_error(expected, output):
        error = np.power(expected - output, 2)
        print("Outpus = {}".format(output))
        # print("Expected = {}, Output = {}, Error = {}".format(expected, output, error))
        return error

    @staticmethod
    def train(perceptron, data, learning_rate, epochs, loss=None):
        for epoch in range(epochs):
            print("Epoch {}".format(epoch))
            epoch_error = 0
            np.random.shuffle(data)

            for sample in data:
                # print("Weights = {}, bias = {}".format(perceptron.weights, perceptron.bias))
                input_values, expected_value = sample[:perceptron.inputs], sample[-1]

                if loss == 'discrete':
                    error = Optimizer.error(expected_value, perceptron.predict(input_values))
                elif loss == 'mse':
                    error = Optimizer.mean_squared_error(expected_value, perceptron.raw_output(input_values))
                else:
                    raise ValueError('Wrong loss function')
                for i in range(perceptron.inputs):
                    perceptron.weights[i] += learning_rate * error * input_values[i]
                perceptron.bias += learning_rate * error
                epoch_error += error
            print("Epoch error = {}".format(epoch_error))

    @staticmethod
    def trainAdaline(perceptron, data, learning_rate, epochs, loss=None):
        for epoch in range(epochs):
            print("Epoch {}".format(epoch))
            epoch_error = 0
            np.random.shuffle(data)

            for sample in data:
                print("Weights = {}, bias = {}".format(perceptron.weights, perceptron.bias))
                input_values, expected_value = sample[:perceptron.inputs], sample[-1]

                if loss == 'discrete':
                    error = Optimizer.error(expected_value, perceptron.predict(input_values))
                elif loss == 'mse':
                    error = expected_value - perceptron.raw_output(input_values)
                    # print("loss = {}".format(error))
                else:
                    raise ValueError('Wrong loss function')
                print("Inputs = {},  output = {}, prediction = {}, error = {}".format(input_values, perceptron.raw_output(input_values), perceptron.predict(input_values), error))
                for i in range(perceptron.inputs):
                    perceptron.weights[i] += 2 * learning_rate * error * input_values[i]
                perceptron.bias += learning_rate * error
                epoch_error += np.abs(error)

            print("Epoch error = {}".format(epoch_error))


AND_UNIPOLAR_PATTERN = [1, 0, 0, 0]
AND_BIPOLAR_PATTERN = [1, -1, -1, -1]

OR_UNIPOLAR_PATTERN = [1, 1, 1, 0]
OR_BIPOLAR_PATTERN = [1, 1, 1, -1]

samples = 10
epsilon = 0.1
high_range = (1 - epsilon, 1)

low_range, low_output_value, loss_type, outputs_pattern = (0, epsilon), 0, 'unipolar', AND_UNIPOLAR_PATTERN
# low_range, low_output_value, loss_type, outputs_pattern = (-1, -1 + epsilon), -1, 'bipolar', AND_BIPOLAR_PATTERN

perceptron = Perceptron(2, (-0.1, 0.5), loss_type)


def build_data_config(samples, high_range, low_range, outputs):
    return [[samples, high_range, high_range, outputs[0]],
            [samples, high_range, low_range, outputs[1]],
            [samples, low_range, high_range, outputs[2]],
            [samples, low_range, low_range, outputs[3]]]


def build_data_set(config):
    data = []
    for cfg in config:
        data.append(LogicalFunctionsGenerator.generate_logical_function(*cfg))

    return np.asarray(data).reshape((samples * len(training_data_config), 3))


training_data_config = build_data_config(samples, high_range, low_range, outputs_pattern)
data = build_data_set(training_data_config)


optimizer = Optimizer()
# optimizer.trainAdaline(perceptron, data, 0.01, 50, 'mse')
optimizer.train(perceptron, data, 0.01, 50, 'discrete')


# test_data_config = build_data_config(samples, high_range, low_range, outputs_pattern)
# data = build_data_set(test_data_config)

# for sample in data:
#     result = perceptron.predict(sample)
#     # print("{} -> {}".format(sample, result))
#     if result == 1:
#         plt.plot(sample[0], sample[1], 'or')
#     else:
#         plt.plot(sample[0], sample[1], 'ob')
#
# # plt.show()
#
print(perceptron.weights)
print(perceptron.bias)