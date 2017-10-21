import network2
import numpy as np

net = network2.Network([6, 100, 18], cost = network2.CrossEntropyCost)
train_data = ((np.asmatrix(np.load('allX.npy'))), (np.asmatrix(np.load('allY.npy'))))
# test_data = (np.asmatrix(np.load('X_test.npy').T), np.asmatrix(np.load('Y_test.npy')).T)


evaluation_cost, evaluation_accuracy,training_cost, training_accuracy = \
    net.SGD(train_data, 20, 5, 0.05,
        lmbda= 5,
        evaluation_data=None,
        monitor_evaluation_accuracy=False,
        monitor_evaluation_cost=False,
        monitor_training_accuracy=True,
        monitor_training_cost=False)
