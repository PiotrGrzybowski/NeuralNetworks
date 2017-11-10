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

path = '/home/piotr/Workspace/MachineLearning/NeuralNetworks/networks/mlp/images.npy'
training_data, validation_data, test_data = build_data(path)
initializer = GaussianNormalScaled()
activation = sigmoid

input_layer = Input(70)
dense_1 = Dense(input_layer, 64, activation, initializer)
output_layer = Output(dense_1, 10, activation, initializer)

network = Network([input_layer, dense_1, output_layer])

#
# print(network.layers[1].weights.shape)
# print(network.layers[1].biases.shape)
# print(network.layers[2].weights.shape)
# print(network.layers[2].biases.shape)
# optimizer = GradientDescent(7, 1.3, MeanSquaredError, 1, True)
# optimizer.train(network, training_data, validation_data, None)

network.layers[1].weights = np.load('w1.npy').T
network.layers[1].biases = np.expand_dims(np.load('b1.npy'), axis=1)
network.layers[2].weights = np.load('w2.npy').T
network.layers[2].biases = np.expand_dims(np.load('b2.npy'), axis=1)

# print()
# print()
# print(np.load('w1.npy').T.shape)
# print(network.layers[1].weights.shape)
# print(network.layers[1].biases.shape)
# print(network.layers[2].weights.shape)
# print(network.layers[2].biases.shape)
predict_directory(network, np.load('/home/piotr/Workspace/MachineLearning/NeuralNetworks/networks/images_0.npy'))