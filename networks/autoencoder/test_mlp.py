import unittest
import numpy as np

from autoencoder.mlp import Network


class TestMlp(unittest.TestCase):
    def test_weights_shapes(self):
        mlp = Network(32)

        self.assertEqual(mlp.output_weights.shape, (784, 32))
        self.assertEqual(mlp.hidden_weights.shape, (32, 784))

    def test_bias_shapes(self):
        mlp = Network(32)

        self.assertEqual(mlp.output_bias.shape, (784, 1))
        self.assertEqual(mlp.hidden_bias.shape, (32, 1))

    def test_feed_forward_single_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 1)

        mlp.load_input(data)
        mlp.feed_forward()

        self.assertEqual(mlp.hidden_net.shape, (32, 1))
        self.assertEqual(mlp.hidden_activation.shape, (32, 1))
        self.assertEqual(mlp.hidden_activation_derivative.shape, (32, 1))

        self.assertEqual(mlp.output_net.shape, (784, 1))
        self.assertEqual(mlp.output_activation.shape, (784, 1))
        self.assertEqual(mlp.output_activation_derivative.shape, (784, 1))

    def test_feed_forward_batch_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 200)

        mlp.load_input(data)
        mlp.feed_forward()

        self.assertEqual(mlp.hidden_net.shape, (32, 200))
        self.assertEqual(mlp.hidden_activation.shape, (32, 200))
        self.assertEqual(mlp.hidden_activation_derivative.shape, (32, 200))

        self.assertEqual(mlp.output_net.shape, (784, 200))
        self.assertEqual(mlp.output_activation.shape, (784, 200))
        self.assertEqual(mlp.output_activation_derivative.shape, (784, 200))

    def test_propagate_backward_single_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 1)

        mlp.load_input(data)
        mlp.feed_forward()
        mlp.propagate_backward()

        self.assertEqual(mlp.output_error.shape, (784, 1))
        self.assertEqual(mlp.hidden_error.shape, (32, 1))

    def test_propagate_backward_batch_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 200)

        mlp.load_input(data)
        mlp.feed_forward()
        mlp.propagate_backward()

        self.assertEqual(mlp.output_error.shape, (784, 200))
        self.assertEqual(mlp.hidden_error.shape, (32, 200))

    def test_calculate_gradients_single_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 1)

        mlp.load_input(data)
        mlp.feed_forward()
        mlp.propagate_backward()
        mlp.calculate_parameters_gradients()

        self.assertEqual(mlp.output_weights_gradient.shape, mlp.output_weights.shape)
        self.assertEqual(mlp.hidden_weights_gradient.shape, mlp.hidden_weights.shape)

        self.assertEqual(mlp.output_bias_gradient.shape, mlp.output_bias.shape)
        self.assertEqual(mlp.hidden_bias_gradient.shape, mlp.hidden_bias.shape)

    def test_calculate_gradients_batch_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 200)

        mlp.load_input(data)
        mlp.feed_forward()
        mlp.propagate_backward()
        mlp.calculate_parameters_gradients()

        self.assertEqual(mlp.output_weights_gradient.shape, mlp.output_weights.shape)
        self.assertEqual(mlp.hidden_weights_gradient.shape, mlp.hidden_weights.shape)

        self.assertEqual(mlp.output_bias_gradient.shape, mlp.output_bias.shape)
        self.assertEqual(mlp.hidden_bias_gradient.shape, mlp.hidden_bias.shape)

    def test_batch_generator_single_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 200)
        batch_size = 1

        for batch in mlp.batch_generator(data, batch_size):
            self.assertEqual(batch.shape, (784, 1))

    def test_batch_generator_batch_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 200)
        batch_size = 32

        for batch in mlp.batch_generator(data, batch_size):
            self.assertEqual(batch.shape, (784, 32))

    def test_train_single_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 200)
        batch_size = 1
        learning_rate = 0.01
        epochs = 10
        l2 = 0.0001

        mlp.train(data, learning_rate, epochs, batch_size, l2)

    def test_train_batch_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 200)
        batch_size = 32
        learning_rate = 0.01
        epochs = 10
        l2 = 0.0001

        mlp.train(data, learning_rate, epochs, batch_size, l2)


if __name__ == '__main__':
    unittest.main()
