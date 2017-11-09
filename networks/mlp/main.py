import numpy as np

from mlp.activation_functions import sigmoid
from mlp.hyper_parameters_tuner import Tuner
from mlp.initializer import GaussianNormal, GaussianNormalScaled
from mlp.loss_functions import MeanSquaredError
from mlp.network import Input, Dense, Network, Output
from mlp.optimizer import GradientDescent
from mlp.utils import build_data

path = '/Users/Piotr/Workspace/NeuralNetworks/networks/mlp/images.npy'
training_data, validation_data, test_data = build_data(path)
initializer = GaussianNormalScaled()
activation = sigmoid

input_layer = Input(70)
dense_1 = Dense(input_layer, 5, activation, initializer)
output_layer = Output(dense_1, 10, activation, initializer)

network = Network([input_layer, dense_1, output_layer])

optimizer = GradientDescent(10, 0.25, MeanSquaredError, 1, True)
optimizer.train(network, training_data, validation_data, None)
