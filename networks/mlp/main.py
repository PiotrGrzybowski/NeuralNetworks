import numpy as np

from mlp.activation_functions import sigmoid
from mlp.initializer import GaussianNormal, GaussianNormalScaled
from mlp.loss_functions import MeanSquaredError
from mlp.network import Input, Dense, Network, Output
from mlp.optimizer import GradientDescent

images = np.load('/Users/Piotr/Workspace/NeuralNetworks/networks/mlp/images.npy')
initializer = GaussianNormalScaled()
activation = sigmoid

input_layer = Input(70)
dense_1 = Dense(input_layer, 20, activation, initializer)
dense_2 = Dense(dense_1, 10, activation, initializer)
dense_3 = Output(dense_2, 10, activation, initializer)

# dense_1.weights = np.full(dense_1.weights.shape, 0.5)
# dense_1.biases = np.full(dense_1.biases.shape, 1.0)
#
# dense_2.weights = np.full(dense_2.weights.shape, 0.5)
# dense_2.biases = np.full(dense_2.biases.shape, 1.0)

network = Network([input_layer, dense_1, dense_2, dense_3])
optimizer = GradientDescent(10, 0.5, MeanSquaredError, 1)
optimizer.train(network, images)

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
