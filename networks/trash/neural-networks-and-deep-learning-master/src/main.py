import mnist_loader
import network2
import numpy as np

# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = np.load('/home/piotr/Workspace/MachineLearning/NeuralNetworks/networks/mlp/images.npy')
net = network2.Network([70, 40, 10], cost=network2.QuadraticCost)
net.large_weight_initializer()

net.SGD(training_data, 10, 1, 0.5, evaluation_data=None, lmbda=0, monitor_evaluation_cost=False, monitor_evaluation_accuracy=False, monitor_training_cost=True, monitor_training_accuracy=True)
