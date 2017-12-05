import pickle

import matplotlib.pyplot as plt
import numpy as np

from autoencoder.mlp import Network

test_data = np.load('test.npy')
train_data = np.load('train.npy')
mlp = Network(32)
batch_size = 32
learning_rate = 0.1
epochs = 5
l2 = 0.0

mlp.train(train_data, learning_rate, epochs, batch_size, l2, test_data)

with open('mlp.pickle', 'wb') as handle:
    pickle.dump(mlp, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('mlp.pickle', 'rb') as handle:
#     mlp = pickle.load(handle)
#     experiment_data = test_data[:, 100:200]
#     mlp.load_input(experiment_data)
#     mlp.feed_forward()
#
#     fig = plt.figure()
#     plt.gray()
#
#     # for i in range(50):
#     #     ax = fig.add_subplot(10, 10, 10 * (i // 10) + i + 1)
#     #     ax.imshow(experiment_data[:, i].reshape(28, 28))
#     #     ax.get_xaxis().set_visible(False)
#     #     ax.get_yaxis().set_visible(False)
#     #
#     #     ax = fig.add_subplot(10, 10, 10 * (i // 10) + i + 11)
#     #     ax.imshow(mlp.output_activation[:, i].reshape(28, 28))
#     #     ax.get_xaxis().set_visible(False)
#     #     ax.get_yaxis().set_visible(False)
#     # plt.show()
#
#
#     weights = mlp.hidden_weights
#     print(weights.shape)
#     weights /= np.linalg.norm(weights)
#     weights = weights.T
#
#
# fig = plt.figure()
# plt.gray()
#
# for i in range(100):
#     ax = fig.add_subplot(10, 10, i + 1)
#     ax.imshow(weights[:, i].reshape(28, 28))
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()
