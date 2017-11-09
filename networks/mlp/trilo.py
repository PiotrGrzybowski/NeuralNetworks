from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.initializers import RandomNormal, Ones
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD

batch_size = 1
num_classes = 10
epochs = 5
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

import numpy as np
dataset = np.load('/Users/Piotr/Workspace/NeuralNetworks/networks/mlp/images.npy')
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
kernel_initializer = RandomNormal(mean=0.0, stddev=0.1, seed=None)
bias_initializer = Ones()

model = Sequential()
model.add(Dense(20, activation='sigmoid', input_shape=(70,), kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
# model.add(Dropout(0.2))
# model.add(Dense(20, activation='sigmoid'))
# model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

optimizer = SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False)

model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['accuracy'])


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
