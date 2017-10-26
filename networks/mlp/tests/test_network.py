import unittest

from mlp.activations import sigmoid
from mlp.initializer import RandomInitializer
from mlp.network import Network, Input, Dense


class TestNetwork(unittest.TestCase):
    def test_three_layer_network_build(self):
        initializer = RandomInitializer([-1, 1], [1, 1])
        activation = sigmoid
        input_layer = Input(784)

        dense_1 = Dense(input_layer, 100, activation, initializer)
        dense_2 = Dense(dense_1, 50, activation, initializer)
        dense_3 = Dense(dense_2, 10, activation, initializer)

        network = Network([input_layer, dense_1, dense_2, dense_3])

        self.assertEqual(network.layers[1].weights.shape, (100, 784))
        self.assertEqual(network.layers[2].weights.shape, (50, 100))
        self.assertEqual(network.layers[3].weights.shape, (10, 50))


if __name__ == '__main__':
    unittest.main()