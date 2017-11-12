import numpy as np

from mlp.activation_functions import sigmoid
from mlp.hyper_parameters_tuner import Tuner
from mlp.initializer import GaussianNormal, GaussianNormalScaled
from mlp.loss_functions import MeanSquaredError
from mlp.network import Input, Dense, Network, Output
from mlp.optimizer import GradientDescent, MomentumGradientDescent
from mlp.utils import build_data, parse_images


def predict_directory(network, data):
    s = 0
    for sample in data:
        network.load_sample(sample)
        network.propagate_forward()

        print("Expected = {},\nPredicte = {}\n".format(network.last_layer.expected_class, network.last_layer.predicted_class))
        s += network.correct_prediction
    print("Correct = {}".format(s))

path = 'images.npy'
training_data, validation_data, test_data = build_data(path)
initializer = GaussianNormalScaled()
activation = sigmoid

input_layer = Input(70)
dense_1 = Dense(input_layer, 64, activation, initializer)
output_layer = Output(dense_1, 10, activation, initializer)

network = Network([input_layer, dense_1, output_layer])
optimizer = GradientDescent(8, 1.3, MeanSquaredError, 1, True)
optimizer.train(network, training_data, validation_data, None)
#
# np.save('www1.npy', network.layers[1].weights)
# np.save('www2.npy', network.layers[2].weights)
#
# np.save('bbb1.npy', network.layers[1].biases)
# np.save('bbb2.npy', network.layers[2].biases)

network.layers[1].weights = np.load('www1.npy')
network.layers[1].biases = np.load('bbb1.npy')
network.layers[2].weights = np.load('www2.npy')
network.layers[2].biases = np.load('bbb2.npy')

predict_directory(network, np.load('/Users/Piotr/Workspace/NeuralNetworks/networks/images_2.npy'))