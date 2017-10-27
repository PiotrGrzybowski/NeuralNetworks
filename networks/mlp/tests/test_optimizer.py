import unittest
import numpy as np

from mlp.activation_functions import sigmoid
from mlp.initializer import RandomInitializer
from mlp.loss_functions import MeanSquaredError
from mlp.network import Network, Input, Dense
from mlp.optimizer import GradientDescent


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.initializer = RandomInitializer([0.5, 0.5], [1, 1])
        self.activation = sigmoid
        self.input_layer = Input(6)

        self.dense_1 = Dense(self.input_layer, 4, self.activation, self.initializer)
        self.dense_2 = Dense(self.dense_1, 4, self.activation, self.initializer)
        self.dense_3 = Dense(self.dense_2, 2, self.activation, self.initializer)
        self.network = Network([self.input_layer, self.dense_1, self.dense_2, self.dense_3])

        self.optimizer = GradientDescent(5, 0.01, MeanSquaredError)

    def test_gradient_descent(self):
        training_data = [(np.array([[-0.2], [-0.1], [0.0], [0.1], [0.2], [0.3]]), np.array([[1], [0]]))]
        print("")
        self.optimizer.train(self.network, training_data)


if __name__ == '__main__':
    unittest.main()