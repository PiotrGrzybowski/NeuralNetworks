import numpy as np


class LossFunction:
    @staticmethod
    def calculate_cost(expected_value, predicted):
        raise NotImplementedError("Should have implemented this!")

    @staticmethod
    def calculate_cost_gradient(expected_value, outputs, derivative_outputs):
        raise NotImplementedError("Should have implemented this!")


class MeanSquaredError(LossFunction):
    @staticmethod
    def calculate_cost(expected_value, outputs):
        return 0.5 * np.power(np.linalg.norm(expected_value - outputs), 2)

    @staticmethod
    def calculate_cost_gradient(expected_value, outputs, derivative_outputs):
        return (outputs - expected_value) * derivative_outputs
