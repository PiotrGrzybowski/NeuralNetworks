import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    a = np.zeros(shape=x.shape)
    a[x <= 0] = 0
    a[x > 0] = 1
    return a


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - x ** 2


ACTIVATION_DERIVATIVES_MAP = {
    sigmoid: sigmoid_derivative,
    relu: relu_derivative,
    tanh: tanh_derivative
}


def get_activation_derivative(activation):
    return ACTIVATION_DERIVATIVES_MAP[activation]
