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

    @staticmethod
    def build_data_config(samples, high_range, low_range, outputs):
        return [[samples, high_range, high_range, outputs[0]],
                [samples, high_range, low_range, outputs[1]],
                [samples, low_range, high_range, outputs[2]],
                [samples, low_range, low_range, outputs[3]]]

    @staticmethod
    def generate_data_set(samples, high_range, low_range, outputs):
        config = LogicalFunctionsGenerator.build_data_config(samples, high_range, low_range, outputs)
        data = []
        for cfg in config:
            data.append(LogicalFunctionsGenerator.generate_logical_function(*cfg))

        return np.asarray(data).reshape((samples * len(config), 3))
