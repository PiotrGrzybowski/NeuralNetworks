import numpy as np

from mlp.activation_functions import sigmoid, relu
from mlp.initializer import RandomInitializer
from mlp.loss_functions import MeanSquaredError
from mlp.network import InputLayer, DenseLayer, Network, OutputLayer
from mlp.optimizer import GradientDescent

images = np.load('/home/piotr/Workspace/MachineLearning/NeuralNetworks/data/images.npy')
initializer = RandomInitializer()
activation = sigmoid

input_layer = InputLayer(70)
dense_1 = DenseLayer(input_layer, 5, activation, initializer)
dense_2 = OutputLayer(dense_1, 10, activation, initializer)

network = Network([input_layer, dense_1, dense_2])
optimizer = GradientDescent(20, 0.5, MeanSquaredError)
optimizer.train(network, images)

