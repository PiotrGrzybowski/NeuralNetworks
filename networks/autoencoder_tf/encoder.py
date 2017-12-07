import numpy as np
import tensorflow as tf

from autoencoder_tf.utils import MeanSquaredError, sigmoid_derivative, shuffle_data, batch_generator


class Network:
    def __init__(self, hidden):
        self.input = tf.placeholder(tf.float32, shape=[784, None], name="input")
        self.loss_function = MeanSquaredError()
        
        self.weights = [tf.Variable(tf.random_normal(shape=[hidden, 784], stddev=0.35), name="hidden_weights"),
                        tf.Variable(tf.random_normal(shape=[784, hidden], stddev=0.35), name="output_weights")]
        
        self.biases = [tf.Variable(tf.ones(shape=[hidden, 1], name="hidden_bias")),
                       tf.Variable(tf.ones(shape=[784, 1], name="output_bias"))]
        
        self.nets = [tf.matmul(self.weights[0], self.input),
                     tf.matmul(self.weights[1], self.activations[0])]
        self.activations = [tf.sigmoid(self.nets[0]),
                            tf.sigmoid(self.nets[1])]
        self.activation_derivatives = [sigmoid_derivative(self.nets[0]), sigmoid_derivative(self.nets[1])]

        self.errors = [tf.multiply(self.loss_function.calculate_cost_gradient(self.input, self.activations[1]),
                                   self.activation_derivatives[1]),
                       tf.multiply(tf.matmul(tf.transpose(self.weights[1]), self.errors[1]),
                                   self.activation_derivatives[0])]
        
        self.weights_gradient = [tf.matmul(self.errors[0], tf.transpose(self.input)),
                                 tf.matmul(self.errors[1], tf.transpose(self.activations[0]))]
        
        self.bias_gradient = [tf.expand_dims(tf.reduce_sum(self.errors[0], axis=1), axis=1),
                              tf.expand_dims(tf.reduce_sum(self.errors[1], axis=1), axis=1)]
        
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
                        tf.assign(self.weights[1], tf.subtract(
                            tf.multiply(tf.subtract(tf.constant(1.0), tf.multiply(eta, l2)), self.weights[1]),
                            tf.multiply(tf.div(eta, batch_size), self.weights_gradient[1]))),

                        tf.assign(self.weights[0], tf.subtract(
                            tf.multiply(tf.subtract(tf.constant(1.0), tf.multiply(eta, l2)), self.weights[0]),
                            tf.multiply(tf.div(eta, batch_size), self.weights_gradient[0]))),

                        tf.assign(self.biases[1], tf.subtract(self.biases[1], tf.multiply(tf.div(eta, batch_size),
                                                                                              self.bias_gradient[1]))),

                        tf.assign(self.biases[0], tf.subtract(self.biases[0], tf.multiply(tf.div(eta, batch_size),
                                                                                              self.bias_gradient[0])))]

                    sess.run(steps, feed_dict={self.input: batch})

                cost = sess.run([self.calculate_cost(training_data, l2)], feed_dict={self.input: training_data})
                print("  Training cost = {}".format(cost[0]))

    def calculate_cost(self, data, l2):
        l2_factor = l2 / (2 * data.shape[1])
        output_l2 = tf.scalar_mul(l2_factor, tf.pow(tf.norm(self.weights[1]), 2))
        hidden_l2 = tf.scalar_mul(l2_factor, tf.pow(tf.norm(self.weights[0]), 2))

        return tf.add(tf.add(tf.div(self.loss_function.calculate_cost(data, self.activations[1]), data.shape[1]),
                             output_l2), hidden_l2)
