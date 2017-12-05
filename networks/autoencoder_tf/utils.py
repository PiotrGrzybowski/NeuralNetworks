from random import shuffle

import tensorflow as tf
import numpy as np


def sigmoid_derivative(x):
    return (1 - tf.sigmoid(x)) * tf.sigmoid(x)


def batch_generator(data, batch_size):
    for i in np.arange(0, data.shape[1] - batch_size, batch_size):
        yield data[:, i: i + batch_size]


def shuffle_data(data):
    data = data.T
    shuffle(data)
    return data.T


class MeanSquaredError:
    @staticmethod
    def calculate_cost(expected, outputs):
        return 0.5 * tf.pow(tf.norm(expected - outputs), 2)

    @staticmethod
    def calculate_cost_gradient(expected, outputs):
        return outputs - expected
