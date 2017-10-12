import numpy as np
from neurons.data_generator import LogicalFunctionsGenerator
from neurons.perceptron import Perceptron
from prettytable import PrettyTable

class Optimizer:
    @staticmethod
    def error(expected, predicted):
        return expected - predicted

    @staticmethod
    def train(perceptron, data, learning_rate, epochs):
        for epoch in range(epochs):
            print("Epoch {}".format(epoch))
            table = PrettyTable(['x1', 'x2', 'output', 'Y', 'y', 'error', 'w1_old', 'w2_old', 'w1_change', 'w2_change', 'w1_new', 'w2_new'])
            epoch_error = 0
            # np.random.shuffle(data)
            for sample in data:
                input_values, expected_value = sample[:perceptron.inputs], sample[-1]
                prediction = perceptron.predict(input_values)
                error = Optimizer.error(expected_value, prediction)

                w1 = perceptron.weights[0]
                w2 = perceptron.weights[1]
                output = perceptron.raw_output(input_values)

                for i in range(perceptron.inputs):
                    perceptron.weights[i] += learning_rate * error * input_values[i]

                table.add_row([input_values[0], input_values[1], output, expected_value, prediction, error, w1, w2, learning_rate * error * input_values[0], learning_rate * error * input_values[1], perceptron.weights[0], perceptron.weights[1]])
                print(table)
                epoch_error += error
            print("Epoch error = {}".format(epoch_error))


perceptron = Perceptron(2, (-1, 1), 'unipolar')
perceptron.weights = np.array([0.0, 0])
# positive = LogicalFunctionsGenerator.generate_logical_function(100, np.logical_and, (0.9, 1))
# negative = LogicalFunctionsGenerator.generate_logical_function(100, np.logical_and, (0, 0.1))
# data = np.append(positive, negative, axis=0)
# np.random.shuffle(data)


data = np.array([[1, 1, 1],[1, 0, 0],[0, 1, 0],[0, 0, 0]])
optimizer = Optimizer()
optimizer.train(perceptron, data, 0.001, 1000)

