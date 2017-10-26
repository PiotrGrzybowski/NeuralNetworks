import unittest
import numpy as np

from mlp.initializer import RandomInitializer
from mlp.network import Input


class TestInitializer(unittest.TestCase):
    def test_random_initializer_shape(self):
        inputs = Input(784)
        initializer = RandomInitializer([-1, 1], [1, 1])

        self.assertEqual(initializer.initialize_weights(units=100, inputs=inputs).shape, (100, 784))
        self.assertEqual(initializer.initialize_bias(units=100).shape, (100,))

    def test_random_initializer_value_ranges(self):
        inputs = Input(784)
        initializer = RandomInitializer([-1, 1], [1, 1])

        weights = initializer.initialize_weights(units=100, inputs=inputs)
        bias = initializer.initialize_bias(units=100)

        self.assertTrue(initializer.weight_range[0] <= np.all(weights) <= initializer.weight_range[1])
        self.assertTrue(initializer.bias_range[0] <= np.all(bias) <= initializer.bias_range[1])

if __name__ == '__main__':
    unittest.main()
