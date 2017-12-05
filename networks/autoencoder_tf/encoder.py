import tensorflow as tf


def sigmoid_derivative(x):
    return (1 - tf.sigmoid(x)) * tf.sigmoid(x)


class Network:
    def __init__(self, hidden):
        self.input = tf.placeholder(tf.float32, shape=[784, None], name="input")

        self.hidden_weights = tf.Variable(tf.random_normal(shape=[hidden, 784], stddev=0.35), name="hidden_weights")
        self.output_weights = tf.Variable(tf.random_normal(shape=[784, hidden], stddev=0.35), name="output_weights")

        self.hidden_bias = tf.ones(shape=[hidden, 1], name="hidden_bias")
        self.output_bias = tf.ones(shape=[784, 1], name="output_bias")

        self.hidden_net = tf.matmul(self.hidden_weights, self.input)
        self.hidden_activation = tf.sigmoid(self.hidden_net)
        self.hidden_activation_derivative = sigmoid_derivative(self.hidden_net)

        self.output_net = tf.matmul(self.output_weights, self.hidden_activation)
        self.output_activation = tf.sigmoid(self.output_net)
        self.output_activation_derivative = sigmoid_derivative(self.output_net)


        self.output_error = np.multiply(self.loss_function.calculate_cost_gradient(self.input, self.output_activation),
                                        self.output_activation_derivative)