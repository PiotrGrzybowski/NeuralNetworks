import numpy as np

LEAST_MEAN_SQUARE = 'least_mean_square'


class Optimizer:
    def __init__(self, loss):
        self.epoch_error = 0
        self.loss = loss

    def train(self, neuron, data, learning_rate, epochs):
        for epoch in range(epochs):
            print("Epoch {}".format(epoch))
            epoch_error = 0
            np.random.shuffle(data)

            for sample in data:
                input_values, expected_value = sample[:neuron.get_number_of_inputs], sample[-1]
                error = self.calculate_error(expected_value, input_values, neuron)

                for i in range(neuron.get_number_of_inputs):
                    neuron.weights[i] += learning_rate * error * input_values[i]
                neuron.bias += learning_rate * error
                epoch_error += error
            print("Epoch error = {}".format(epoch_error))

    def calculate_error(self, expected_value, input_values, neuron):
        if self.loss == 'discrete':
            error = expected_value - neuron.predict(input_values)
        elif self.loss == LEAST_MEAN_SQUARE:
            error = 2 * (expected_value - neuron.raw_output(input_values))
        else:
            raise ValueError('Wrong loss function')

        return error
