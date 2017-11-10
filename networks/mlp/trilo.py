from __future__ import print_function

import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.initializers import RandomNormal, Ones
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
import os
batch_size = 1
num_classes = 10
epochs = 10
#
# # the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print(x_train.shape, 'train samples')
# print(x_test.shape, 'test samples')
#
# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
#
# print(y_train.shape)
# print(y_test.shape)
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
dataset = np.load('/home/piotr/Workspace/MachineLearning/NeuralNetworks/networks/mlp/images.npy')
np.random.shuffle(dataset)

X = np.zeros(shape=(1744, 70))
Y = np.zeros(shape=(1744, 10))

for i in range(1744):
    X[i, :] = dataset[i][0].squeeze()
    Y[i, :] = dataset[i][1].squeeze()


print(X.shape)
print(Y.shape)
split = 1600

x_train = X[:split]
y_train = Y[:split]

x_test = X[split:]
y_test = Y[split:]

model = Sequential()
model.add(Dense(64, activation='sigmoid', input_shape=(70,)))
# model.add(Dropout(0.2))
# model.add(Dense(20, activation='sigmoid'))
# model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

optimizer = SGD(lr=0.5, momentum=0.9, decay=0.0, nesterov=False)

model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['accuracy'])


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    shuffle=False)

weights_1 = model.layers[0].get_weights()[0]
biases_1 = model.layers[0].get_weights()[1]

weights_2 = model.layers[1].get_weights()[0]
biases_2 = model.layers[1].get_weights()[1]

print(weights_1.shape)
print(biases_1.shape)

print(weights_2.shape)
print(biases_2.shape)

np.save('w1.npy', weights_1)
np.save('b1.npy', biases_1)
np.save('w2.npy', weights_2)
np.save('b2.npy', biases_2)