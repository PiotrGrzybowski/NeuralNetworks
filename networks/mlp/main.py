import numpy as np

from mlp.activation_functions import sigmoid
from mlp.hyper_parameters_tuner import Tuner
from mlp.initializer import GaussianNormal, GaussianNormalScaled
from mlp.loss_functions import MeanSquaredError
from mlp.network import Input, Dense, Network, Output
from mlp.optimizer import GradientDescent

# images = np.load('/Users/Piotr/Workspace/NeuralNetworks/networks/mlp/images.npy')
# initializer = GaussianNormalScaled()
# activation = sigmoid
#
# input_layer = Input(70)
# dense_1 = Dense(input_layer, 20, activation, initializer)
# dense_2 = Dense(dense_1, 10, activation, initializer)
# output_layer = Output(dense_2, 10, activation, initializer)
#
# network = Network([input_layer, dense_1, dense_2, output_layer])
# optimizer = GradientDescent(10, 0.5, MeanSquaredError, 1, True)
# optimizer.train(network, images, None, None)

# # data = np.zeros(shape=(70, 3))
# # data[:, 0] = ima
#
# samples = images[:3]
# # network.load_sample(samples[0])
# network.load_batch(samples)
# network.propagate_forward()
# weights_error, biases_error = network.propagate_backward(optimizer.loss_function)
# # print(biases_error)
# # print(weights_error[0])
#
# network.update_weights(optimizer.calculate_gradient(weights_error))
# network.update_biases(optimizer.calculate_gradient(biases_error))

tuner = Tuner([0.5])

tuner.tune(1)