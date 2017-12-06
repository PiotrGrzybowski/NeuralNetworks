import numpy as np
from autoencoder_tf.encoder import Network

test_data = np.load('test.npy')
train_data = np.load('train.npy')

mlp = Network(100)
batch_size = 1000
learning_rate = 0.1
epochs = 10
l2 = 0

mlp.train(train_data, learning_rate, epochs, batch_size, l2)
