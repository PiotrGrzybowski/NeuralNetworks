import numpy as np
import os
from PIL import Image
from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()
lb.fit(range(10))

DIRECTORY = "/Users/Piotr/Workspace/NeuralNetworks/data/03_11_18_cyfry_zima_17-18_ostateczny_zbior/"
IMAGE_ROW_SHAPE = (70,)


def get_label_from_filename(filename):
    return int(filename.split('_')[0])


def parse_images(path):
    l = []
    for filename in os.listdir(path):
        img = np.invert(np.asarray(Image.open(path + filename).convert('1'))).astype(int)
        if img.shape == (10, 7):
            # print(img)
            # print(img.reshape(IMAGE_ROW_SHAPE, order='F'))

            l.append((img.reshape(IMAGE_ROW_SHAPE), get_label_from_filename(filename)))
        else:
            print("Shape = {}, index = {}".format(img.shape, filename))

    return l


def list_to_arrays(data):
    images = np.empty((len(data), IMAGE_ROW_SHAPE[0]))
    labels = np.empty((len(data)))
    for i in range(len(data)):
        images[i, :] = data[i][0]
        labels[i] = data[i][1]

    return images, labels


def generate_mini_batches(data, mini_batch):
    for i in np.arange(0, len(data) - mini_batch, mini_batch):
        yield data[i: i + mini_batch]


def build_data(path):
    images = np.load(path)
    np.random.shuffle(images)
    train = 1444
    validation = 1644
    return images[:train], images[train:], images[train:]

import Augmentor
p = Augmentor.Pipeline(DIRECTORY)
p.rotate(0.7, 10, 10)

g = p.keras_generator(batch_size=128)

# images = parse_images(DIRECTORY)
# print(images[0][0])
# train_x, train_y = list_to_arrays(images)
#
# print(train_x.shape)
# print(train_y.shape)
#
# training_data = []
# print(train_y[0])
# ty = lb.transform(train_y)
# print()
# for x, y in zip(train_x, ty):
#     training_data.append(((np.expand_dims(x, axis=1)), np.expand_dims(y, axis=1)))
#
# print(len(training_data))
# np.save('images.npy', training_data)
