import pickle

import matplotlib.pyplot as plt
import numpy as np

from autoencoder.mlp import Network
from autoencoder_tf.utils import shuffle_data

test_data = shuffle_data(np.load('test.npy'))
train_data = shuffle_data(np.load('train.npy'))
mlp = Network(100)
batch_size = 32
learning_rate = 0.2
epochs = 10
l2 = 0.001
mlp.drop = 0

mlp.train(train_data[:, :60000], learning_rate, epochs, batch_size, l2, test_data)

train, test = mlp.train_cost, mlp.test_cost

plt.title("Wartość funkcji kosztu zależna od epoki, l2 = {}".format(l2))

line_train = plt.plot(np.arange(1, epochs + 1), train, color='blue', label='Treningowy')
line_test = plt.plot(np.arange(1, epochs + 1), test, color='green', label='Testowy')
plt.legend()
plt.xlabel('Epoka')
plt.ylabel('Koszt')
plt.show()
# with open('mlp.pickle', 'wb') as handle:
#     pickle.dump(mlp, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('mlp.pickle', 'rb') as handle:
#     mlp = pickle.load(handle)
#     experiment_data = test_data[:, 100:200]
#     mlp.load_input(experiment_data)
#     mlp.feed_forward()
#
#     fig = plt.figure()
#     plt.gray()
#
#     for i in range(50):
#         ax = fig.add_subplot(10, 10, 10 * (i // 10) + i + 1)
#         ax.imshow(experiment_data[:, i].reshape(28, 28))
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#
#         ax = fig.add_subplot(10, 10, 10 * (i // 10) + i + 11)
#         ax.imshow(mlp.output_activation[:, i].reshape(28, 28))
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.show()
#
#     # print(mlp.output_weights.shape)
#     # print(mlp.hidden_weights.shape)
#     weights = mlp.hidden_weights
#     weights /= np.linalg.norm(weights)
#     weights = weights.T
#     # print(weights.shape)
#
#     fig = plt.figure()
#     plt.gray()
#
#     for i in range(100):
#         ax = fig.add_subplot(10, 10, i + 1)
#         ax.imshow(weights[:, i].reshape(28, 28))
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.show()
