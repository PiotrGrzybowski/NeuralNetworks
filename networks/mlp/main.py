import numpy as np

from mlp.activation_functions import sigmoid
from mlp.initializer import RandomInitializer
from mlp.loss_functions import MeanSquaredError
from mlp.network import Input, Dense, Network
from mlp.optimizer import GradientDescent

images = np.load('/home/piotr/Workspace/MachineLearning/NeuralNetworks/data/images.npy')
initializer = RandomInitializer([-5, 5], [1, 1])
activation = sigmoid

input_layer = Input(70)
dense_1 = Dense(input_layer, 40, activation, initializer)
dense_2 = Dense(dense_1, 10, activation, initializer)

network = Network([input_layer, dense_1, dense_2])


############MOCKING##################
weights = np.load('/home/piotr/Workspace/MachineLearning/NeuralNetworks/networks/trash/neural-networks-and-deep-learning-master/src/weights.npy', encoding = 'latin1')
biases = np.load('/home/piotr/Workspace/MachineLearning/NeuralNetworks/networks/trash/neural-networks-and-deep-learning-master/src/biases.npy', encoding = 'latin1')

dense_1.weights = weights[0]
dense_1.biases = biases[0]

dense_2.weights = weights[1]
dense_2.biases = biases[1]

print(dense_1.weights[0][0])
####################################################
optimizer = GradientDescent(1, 0.5, MeanSquaredError)
optimizer.train(network, images)

