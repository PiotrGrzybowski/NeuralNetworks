import unittest

import numpy as np
import tensorflow as tf

from autoencoder_tf.encoder import Network
from autoencoder_tf.utils import batch_generator


class TestMlp(unittest.TestCase):
    def test_weights_shapes(self):
        mlp = Network(32)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            self.assertEqual(mlp.hidden_weights.get_shape(), tf.TensorShape([tf.Dimension(32), tf.Dimension(784)]))
            self.assertEqual(mlp.output_weights.get_shape(), tf.TensorShape([tf.Dimension(784), tf.Dimension(32)]))

    def test_bias_shapes(self):
        mlp = Network(32)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            self.assertEqual(mlp.hidden_bias.get_shape(), tf.TensorShape([tf.Dimension(32), tf.Dimension(1)]))
            self.assertEqual(mlp.output_bias.get_shape(), tf.TensorShape([tf.Dimension(784), tf.Dimension(1)]))

    def test_feed_forward_single_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 1)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            hidden_net, hidden_activation, hidden_activation_derivative, \
            output_net, output_activation, output_activation_derivative = sess.run([mlp.hidden_net,
                                                                                    mlp.hidden_activation,
                                                                                    mlp.hidden_activation_derivative,
                                                                                    mlp.output_net,
                                                                                    mlp.output_activation,
                                                                                    mlp.output_activation_derivative],
                                                                                   feed_dict={mlp.input: data})

            self.assertEqual(hidden_net.shape, (32, 1))
            self.assertEqual(hidden_activation.shape, (32, 1))
            self.assertEqual(hidden_activation_derivative.shape, (32, 1))

            self.assertEqual(output_net.shape, (784, 1))
            self.assertEqual(output_activation.shape, (784, 1))
            self.assertEqual(output_activation_derivative.shape, (784, 1))

    def test_feed_forward_batch_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 200)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            hidden_net, hidden_activation, hidden_activation_derivative, \
            output_net, output_activation, output_activation_derivative = sess.run([mlp.hidden_net,
                                                                                    mlp.hidden_activation,
                                                                                    mlp.hidden_activation_derivative,
                                                                                    mlp.output_net,
                                                                                    mlp.output_activation,
                                                                                    mlp.output_activation_derivative],
                                                                                   feed_dict={mlp.input: data})

            self.assertEqual(hidden_net.shape, (32, 200))
            self.assertEqual(hidden_activation.shape, (32, 200))
            self.assertEqual(hidden_activation_derivative.shape, (32, 200))

            self.assertEqual(output_net.shape, (784, 200))
            self.assertEqual(output_activation.shape, (784, 200))
            self.assertEqual(output_activation_derivative.shape, (784, 200))

    def test_propagate_backward_single_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 1)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            output_error, hidden_error = sess.run([mlp.output_error, mlp.hidden_error], feed_dict={mlp.input: data})

            self.assertEqual(output_error.shape, (784, 1))
            self.assertEqual(hidden_error.shape, (32, 1))

    def test_propagate_backward_batch_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 200)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            output_error, hidden_error = sess.run([mlp.output_error, mlp.hidden_error], feed_dict={mlp.input: data})

            self.assertEqual(output_error.shape, (784, 200))
            self.assertEqual(hidden_error.shape, (32, 200))

    def test_calculate_gradients_single_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 1)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            output_weights_gradient, hidden_weights_gradient, output_bias_gradient, hidden_bias_gradient = \
                sess.run([mlp.output_weights_gradient, mlp.hidden_weights_gradient,
                          mlp.output_bias_gradient, mlp.hidden_bias_gradient], feed_dict={mlp.input: data})

            self.assertEqual(output_weights_gradient.shape, (784, 32))
            self.assertEqual(hidden_weights_gradient.shape, (32, 784))

            self.assertEqual(output_bias_gradient.shape, (784, 1))
            self.assertEqual(hidden_bias_gradient.shape, (32, 1))

    def test_calculate_gradients_batch_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 200)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            output_weights_gradient, hidden_weights_gradient, output_bias_gradient, hidden_bias_gradient = \
                sess.run([mlp.output_weights_gradient, mlp.hidden_weights_gradient,
                          mlp.output_bias_gradient, mlp.hidden_bias_gradient], feed_dict={mlp.input: data})

            self.assertEqual(output_weights_gradient.shape, (784, 32))
            self.assertEqual(hidden_weights_gradient.shape, (32, 784))

            self.assertEqual(output_bias_gradient.shape, (784, 1))
            self.assertEqual(hidden_bias_gradient.shape, (32, 1))

    def test_batch_generator_single_sample(self):
        data = np.random.rand(784, 200)
        batch_size = 1

        for batch in batch_generator(data, batch_size):
            self.assertEqual(batch.shape, (784, 1))

    def test_batch_generator_batch_sample(self):
        data = np.random.rand(784, 200)
        batch_size = 64

        for batch in batch_generator(data, batch_size):
            self.assertEqual(batch.shape, (784, 64))

    def test_train_single_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 200)
        batch_size = 1
        learning_rate = 0.01
        epochs = 2
        l2 = 0.0001

        # mlp.train(data, learning_rate, epochs, batch_size, l2)

    def test_train_batch_sample(self):
        mlp = Network(32)
        data = np.random.rand(784, 200)
        batch_size = 2
        learning_rate = 0.01
        epochs = 2
        l2 = 0.0001

        # mlp.train(data, learning_rate, epochs, batch_size, l2)


if __name__ == '__main__':
    unittest.main()
