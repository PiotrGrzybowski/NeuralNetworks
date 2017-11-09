import numpy as np

# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
from mlp import network2

data = np.load('/Users/Piotr/Workspace/NeuralNetworks/networks/mlp/images.npy')
np.random.shuffle(data)
split = 1500
training_data = data[:split]
test_data = data[split:]

net = network2.Network([70, 40, 10], cost=network2.QuadraticCost)
net.default_weight_initializer()

net.SGD(training_data=training_data,
        evaluation_data=test_data,
        epochs=10,
        mini_batch_size=1,
        eta=0.01,
        lmbda=1.8,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
