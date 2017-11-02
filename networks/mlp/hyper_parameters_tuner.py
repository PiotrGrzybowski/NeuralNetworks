import numpy as np

from mlp.activation_functions import sigmoid
from mlp.initializer import GaussianNormalScaled
from mlp.loss_functions import MeanSquaredError
from mlp.network import Input, Dense, Output, Network
from mlp.optimizer import GradientDescent


class Tuner:
    def __init__(self, learning_rates):
        self.learning_rates = learning_rates

    def tune(self, times):
        training_data, validation_data, test_data = self.build_data()
        optimizer = self.build_optimizer()
        network = self.build_network()

        # for learning_rate in self.learning_rates:
        #     optimizer.train(network, training_data, validation_data, test_data)

    def build_network(self):
        initializer = GaussianNormalScaled()
        activation = sigmoid

        input_layer = Input(70)
        dense_1 = Dense(input_layer, 20, activation, initializer)
        dense_2 = Dense(dense_1, 10, activation, initializer)
        output_layer = Output(dense_2, 10, activation, initializer)

        return Network([input_layer, dense_1, dense_2, output_layer])

    def build_optimizer(self):
        return GradientDescent(10, 0.5, MeanSquaredError, 1, True)

    def build_data(self):
        images = np.load('/Users/Piotr/Workspace/NeuralNetworks/networks/mlp/images.npy')
        np.random.shuffle(images)
        train = 1444
        validation = 1644
        print(sorted([np.argmax(s[1]) for s in images[validation:]]))
        return images[:train], images[train:validation], images[validation:]