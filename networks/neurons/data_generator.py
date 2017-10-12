import numpy as np


class LogicalFunctionsGenerator:
    @staticmethod
    def generate_logical_function(samples, first_scope, second_scope, output_value):
        result = np.empty(shape=(samples, 3))
        for i in range(samples):
            x1 = np.random.uniform(*first_scope)
            x2 = np.random.uniform(*second_scope)
            result[i, :] = [x1, x2, output_value]
        return result

