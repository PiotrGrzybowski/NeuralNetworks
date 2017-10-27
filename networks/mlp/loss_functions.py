import numpy as np


class LossFunction:
    @staticmethod
    def calculate_error(expected, predicted):
        raise NotImplementedError("Should have implemented this!")

    @staticmethod
    def calculate_error_derivative(expected, predicted):
        raise NotImplementedError("Should have implemented this!")


class MeanSquaredError(LossFunction):
    @staticmethod
    def calculate_error(expected, activations):
        return 0.5 * np.power(expected - activations, 2)

    @staticmethod
    def calculate_error_derivative(expected, raw_outputs, activation, activation_derivative):
        return expected - raw_outputs
