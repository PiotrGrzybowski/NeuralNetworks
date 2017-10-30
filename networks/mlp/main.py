import numpy as np

from mlp.activation_functions import sigmoid
from mlp.initializer import RandomInitializer
from mlp.loss_functions import MeanSquaredError
from mlp.network import InputLayer, DenseLayer, Network, OutputLayer
from mlp.optimizer import GradientDescent

images = np.load('/home/piotr/Workspace/Repositories/NeuralNetworks/networks/mlp/images.npy')
initializer = RandomInitializer()
activation = sigmoid

input_layer = InputLayer(70)
dense_1 = DenseLayer(input_layer, 5, activation, initializer)
dense_2 = OutputLayer(dense_1, 10, activation, initializer)

network = Network([input_layer, dense_1, dense_2])
optimizer = GradientDescent(20, 0.5, MeanSquaredError, 20)
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
