import unittest
import numpy as np

from mlp.activation_functions import sigmoid
from mlp.initializer import RandomInitializer
from mlp.network import Network, Input, Dense


class TestLayers(unittest.TestCase):
    def setUp(self):
        self.initializer = RandomInitializer([-1, 1], [1, 1])
        self.activation = sigmoid
        self.input_layer = Input(784)
        self.inputs = np.random.rand(784)
        self.input_layer.load_inputs(self.inputs)

        self.dense_1 = Dense(self.input_layer, 100, self.activation, self.initializer)
        self.dense_2 = Dense(self.dense_1, 50, self.activation, self.initializer)
        self.dense_3 = Dense(self.dense_2, 10, self.activation, self.initializer)

        self.network = Network([self.input_layer, self.dense_1, self.dense_2, self.dense_3])

    def test_three_layer_network_build(self):
        self.input_layer.calculate_output()
        self.assertEqual(self.network.layers[1].weights.shape, (100, 784))
        self.assertEqual(self.network.layers[2].weights.shape, (50, 100))
        self.assertEqual(self.network.layers[3].weights.shape, (10, 50))

    def test_input_layer_outputs_shape(self):
        self.input_layer.calculate_output()
        self.assertEqual(self.dense_1.input_layer.outputs.shape, (784, ))

        self.input_layer.load_inputs(np.random.rand(784, 2))
        self.input_layer.calculate_output()
        self.assertEqual(self.dense_1.input_layer.outputs.shape, (784, 2))

        self.input_layer.load_inputs(np.random.rand(784, 10000))
        self.input_layer.calculate_output()
        self.assertEqual(self.dense_1.input_layer.outputs.shape, (784, 10000))

    def test_dense_1_outputs_shape_with_calculate_trigger(self):
        self.input_layer.calculate_output()
        self.dense_1.calculate_output()
        self.assertEqual(self.dense_1.outputs.shape, (100,))

        self.input_layer.load_inputs(np.random.rand(784, 2))
        self.input_layer.calculate_output()
        self.dense_1.calculate_output()
        self.assertEqual(self.dense_1.outputs.shape, (100, 2))

        self.input_layer.load_inputs(np.random.rand(784, 10000))
        self.input_layer.calculate_output()
        self.dense_1.calculate_output()
        self.assertEqual(self.dense_1.outputs.shape, (100, 10000))


class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.initializer = RandomInitializer([-1, 1], [1, 1])
        self.activation = sigmoid
        self.input_layer = Input(784)
        self.inputs = np.random.rand(784)
        self.input_layer.load_inputs(self.inputs)

        self.dense_1 = Dense(self.input_layer, 100, self.activation, self.initializer)
        self.dense_2 = Dense(self.dense_1, 50, self.activation, self.initializer)
        self.dense_3 = Dense(self.dense_2, 10, self.activation, self.initializer)

        self.network = Network([self.input_layer, self.dense_1, self.dense_2, self.dense_3])

    def test_forward_propagation(self):
        self.network.propagate_forward()
        self.assertEqual(self.network.layers[-1].outputs.shape, (10,))

        self.input_layer.load_inputs(np.random.rand(784, 2))
        self.network.propagate_forward()
        self.assertEqual(self.network.layers[-1].outputs.shape, (10, 2))

        self.input_layer.load_inputs(np.random.rand(784, 10000))
        self.network.propagate_forward()
        self.assertEqual(self.network.layers[-1].outputs.shape, (10, 10000))

if __name__ == '__main__':
    unittest.main()