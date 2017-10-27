import numpy as np


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


ACTIVATION_DERIVATIVES_MAP = {
    sigmoid: sigmoid_derivative
}


def get_activation_derivative(activation):
    return ACTIVATION_DERIVATIVES_MAP[activation]