import unittest

from mlp.network import Network


class TestNetwork(unittest.TestCase):
    def test_initialized_weights_shape_no_hidden_layers(self):
        network = Network([8, 6])
        self.assertEqual(network.weights[0].shape, (6, 8))

    def test_initialized_weights_shape_one_hidden_layer(self):
        network = Network([8, 6, 4])
        self.assertEqual(network.weights[0].shape, (6, 8))
        self.assertEqual(network.weights[1].shape, (4, 6))

    def test_initialized_weights_shape_two_hidden_layer(self):
        network = Network([8, 6, 4, 2])
        self.assertEqual(network.weights[0].shape, (6, 8))
        self.assertEqual(network.weights[1].shape, (4, 6))
        self.assertEqual(network.weights[2].shape, (4, 6))
if __name__ == '__main__':
    unittest.main()