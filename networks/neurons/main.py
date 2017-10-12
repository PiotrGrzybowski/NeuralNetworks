from neurons.data_generator import LogicalFunctionsGenerator
from neurons.neuron import Neuron
from neurons.optimizer import Optimizer
import matplotlib.pyplot as plt
import numpy as np

AND_UNIPOLAR_PATTERN = [1, 0, 0, 0]
AND_BIPOLAR_PATTERN = [1, -1, -1, -1]

OR_UNIPOLAR_PATTERN = [1, 1, 1, 0]
OR_BIPOLAR_PATTERN = [1, 1, 1, -1]

samples = 10
epsilon = 0.1
high_range = (1 - epsilon, 1)

# low_range, low_output_value, loss_type, outputs_pattern = (0, epsilon), 0, 'unipolar', AND_UNIPOLAR_PATTERN
low_range, low_output_value, activation, outputs_pattern = (-1, -1 + epsilon), -1, 'bipolar', OR_BIPOLAR_PATTERN

perceptron = Neuron(2, (-0.1, 0.5), activation)


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


optimizer = Optimizer(loss='least_mean_square')
optimizer.train(perceptron, data, 0.01, 50)


# test_data_config = build_data_config(samples, high_range, low_range, outputs_pattern)
# data = build_data_set(test_data_config)

for sample in data:
    result = perceptron.predict(sample)
    # print("{} -> {}".format(sample, result))
    if result == 1:
        plt.plot(sample[0], sample[1], 'or')
    else:
        plt.plot(sample[0], sample[1], 'ob')

plt.show()

print(perceptron.weights)
print(perceptron.bias)