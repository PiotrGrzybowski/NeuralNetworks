import numpy as np


class LossFunction:
    @staticmethod
    def calculate_error(expected, predicted):
        raise NotImplementedError("Should have implemented this!")

    @staticmethod
    def calculate_error_derivative(expected, activations, derivative_activations):
        raise NotImplementedError("Should have implemented this!")


class MeanSquaredError(LossFunction):
    @staticmethod
    def calculate_error(expected, activations):
        return 0.5 * np.power(expected - activations, 2)

    @staticmethod
    def calculate_error_derivative(expected, activations, derivative_activations):
        return (activations - expected) * derivative_activations
