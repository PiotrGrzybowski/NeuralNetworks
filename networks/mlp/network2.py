import numpy as np
import json
import random
import sys


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y.T * np.log(a) - (1 - y).T * np.log(1 - a)))

    @staticmethod
    def output_error(z, a, y):
        return a - y


class Network(object):
    def __init__(self, sizes_of_layers, cost=CrossEntropyCost):
        self.number_of_layers = len(sizes_of_layers)
        self.sizes_of_layers = sizes_of_layers
        self.default_weight_initializer()
        self.cost_function = cost

    def default_weight_initializer(self):
        self.biases = [np.asmatrix(np.random.randn(y, 1)) for y in self.sizes_of_layers[1:]]
        self.weights = [np.asmatrix(np.random.randn(y, x) / np.sqrt(x)) for x, y in
                        zip(self.sizes_of_layers[:-1], self.sizes_of_layers[1:])]

    def large_weight_initializer(self):
        self.biases = [np.asmatrix(np.random.randn(y, 1)) for y in self.sizes_of_layers[1:]]
        self.weights = [np.asmatrix(np.random.randn(y, x)) for x, y in
                        zip(self.sizes_of_layers[:-1], self.sizes_of_layers[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        if evaluation_data is not None:
            n_data = np.shape(evaluation_data[0])[1]
        n = np.shape(training_data[0])[1]

        evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = [], [], [], []

        for epoch in range(epochs):
            for k in range(0, n, mini_batch_size):
                mini_batch_x = (training_data[0])[:, k:k + mini_batch_size]
                mini_batch_y = (training_data[1])[:, k:k + mini_batch_size]

                self.update_mini_batch((mini_batch_x, mini_batch_y), eta, lmbda, (np.shape(training_data[0])[1]))

            print("Epoch %s training complete" % epoch)
            # if monitor_training_cost:
            #     cost = self.total_cost((training_data[0], training_data[1]), lmbda)
            #     training_cost.append(cost)
            #     print( "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy((training_data[0], training_data[1]), convert=False)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {} = {}".format(int(accuracy * 1.5), n,
                                                                       float(int(accuracy * 1.5)) / float(n)))
                print()
            # if monitor_evaluation_cost:
            #     cost = self.total_cost(evaluation_data, lmbda, convert=True)
            #     evaluation_cost.append(cost)
            #     print( "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy((evaluation_data[0], evaluation_data[1]), convert=True)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {} = {}".format(accuracy, n_data,
                                                                         float(accuracy) / float(n_data)))

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b, nabla_w = self.backprop(mini_batch[0], mini_batch[1])

        mini_batch_size = np.shape(mini_batch[0])[1]
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / mini_batch_size) * nw for w, nw in
                        zip(self.weights, nabla_w)]
        self.biases = [b - (eta / mini_batch_size) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        delta_bias = [np.asmatrix(np.zeros(b.shape)) for b in self.biases]
        delta_weight = [np.asmatrix(np.zeros(w.shape)) for w in self.weights]

        activation = x
        neurons_activations_on_layer_l = [x]
        pre_sigmoid_activations_on_layer_l = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            pre_sigmoid_activations_on_layer_l.append(z)
            activation = sigmoid(z)
            neurons_activations_on_layer_l.append(activation)





        delta = (self.cost_function).output_error(pre_sigmoid_activations_on_layer_l[-1], neurons_activations_on_layer_l[-1], y)

        delta_bias[-1] = delta
        delta_weight[-1] = delta * neurons_activations_on_layer_l[-2].T

        for l in range(2, self.number_of_layers):
            z = pre_sigmoid_activations_on_layer_l[-l]
            sp = sigmoid_prime(z)
            delta = np.multiply(self.weights[-l + 1].T * delta, sp)
            delta_bias[-l] = delta
            delta_weight[-l] = np.dot(delta, neurons_activations_on_layer_l[-l - 1].transpose())

        for i in range(0, len(self.sizes_of_layers) - 1):
            delta_bias[i] = np.sum(delta_bias[i], axis=1)
        return (delta_bias, delta_weight)

    def accuracy(self, data, convert=False):
        aa = self.feedforward(data[0])

        a = np.squeeze(np.asarray(np.argmax(aa, axis=0)[0].T))
        y = np.squeeze(np.asarray(np.argmax(data[1], axis=0)[0].T), 1)

        for i in range(0, 18):
            where_i_y = np.where(y == i)[0]
            where_i_a = np.where(a == i)[0]

            # print( sum((a[where_i_y] == i).astype(int))
            # print( str(sum((a[where_i_y] == i).astype(int))) + " / " + str(len(where_i_y))

        return np.sum((np.all(a == np.argmax(data[1], axis=0), axis=0)).astype(int), axis=1).item(0)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for i in range((np.shape(data[0])[1])):
            a = self.feedforward((data[0])[:, i])
            y = (data[1])[:, i]

            cost += self.cost_function.fn(a, y) / (np.shape(data[0])[1])
        cost += 0.5 * (lmbda / (np.shape(data[0])[1])) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes_of_layers,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost_function.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


#### Loading a Network
def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


nn = Network([784, 100, 10])

for w in nn.weights:
    print(w.shape)
