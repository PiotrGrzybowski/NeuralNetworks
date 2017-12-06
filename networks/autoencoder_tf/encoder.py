import numpy as np
import tensorflow as tf

from autoencoder_tf.utils import MeanSquaredError, sigmoid_derivative, shuffle_data, batch_generator


class Network:
    def __init__(self, hidden):
        self.input = tf.placeholder(tf.float32, shape=[784, None], name="input")
        self.loss_function = MeanSquaredError()

        self.hidden_weights = tf.Variable(tf.random_normal(shape=[hidden, 784], stddev=0.35), name="hidden_weights")
        self.output_weights = tf.Variable(tf.random_normal(shape=[784, hidden], stddev=0.35), name="output_weights")

        self.hidden_bias = tf.Variable(tf.ones(shape=[hidden, 1], name="hidden_bias"))
        self.output_bias = tf.Variable(tf.ones(shape=[784, 1], name="output_bias"))

        self.hidden_net = tf.matmul(self.hidden_weights, self.input)
        self.hidden_activation = tf.sigmoid(self.hidden_net)
        self.hidden_activation_derivative = sigmoid_derivative(self.hidden_net)

        self.output_net = tf.matmul(self.output_weights, self.hidden_activation)
        self.output_activation = tf.sigmoid(self.output_net)
        self.output_activation_derivative = sigmoid_derivative(self.output_net)

        self.output_error = tf.multiply(self.loss_function.calculate_cost_gradient(self.input, self.output_activation),
                                        self.output_activation_derivative)

        self.hidden_error = tf.multiply(tf.matmul(tf.transpose(self.output_weights), self.output_error),
                                        self.hidden_activation_derivative)

        self.output_weights_gradient = tf.matmul(self.output_error, tf.transpose(self.hidden_activation))
        self.hidden_weights_gradient = tf.matmul(self.hidden_error, tf.transpose(self.input))

        self.output_bias_gradient = tf.expand_dims(tf.reduce_sum(self.output_error, axis=1), axis=1)
        self.hidden_bias_gradient = tf.expand_dims(tf.reduce_sum(self.hidden_error, axis=1), axis=1)

    def train(self, training_data, eta, epochs, batch_size, l2, test_data=None):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(epochs):
                training_data = shuffle_data(training_data)
                print("Epoch: {}".format(epoch))
                k = 0
                for batch in batch_generator(training_data, batch_size):
                    print(k)
                    k += 1
                    steps = [
                        tf.assign(self.output_weights, tf.subtract(
                            tf.multiply(tf.subtract(tf.constant(1.0), tf.multiply(eta, l2)), self.output_weights),
                            tf.multiply(tf.div(eta, batch_size), self.output_weights_gradient))),

                        tf.assign(self.hidden_weights, tf.subtract(
                            tf.multiply(tf.subtract(tf.constant(1.0), tf.multiply(eta, l2)), self.hidden_weights),
                            tf.multiply(tf.div(eta, batch_size), self.hidden_weights_gradient))),

                        tf.assign(self.output_bias, tf.subtract(self.output_bias, tf.multiply(tf.div(eta, batch_size),
                                                                                              self.output_bias_gradient))),

                        tf.assign(self.hidden_bias, tf.subtract(self.hidden_bias, tf.multiply(tf.div(eta, batch_size),
                                                                                              self.hidden_bias_gradient)))]

                    sess.run(steps, feed_dict={self.input: batch})

                cost = sess.run([self.calculate_cost(training_data, l2)], feed_dict={self.input: training_data})
                print("  Training cost = {}".format(cost[0]))

    def calculate_cost(self, data, l2):
        l2_factor = l2 / (2 * data.shape[1])
        output_l2 = tf.scalar_mul(l2_factor, tf.pow(tf.norm(self.output_weights), 2))
        hidden_l2 = tf.scalar_mul(l2_factor, tf.pow(tf.norm(self.hidden_weights), 2))

        return tf.add(tf.add(tf.div(self.loss_function.calculate_cost(data, self.output_activation), data.shape[1]),
                             output_l2), hidden_l2)
