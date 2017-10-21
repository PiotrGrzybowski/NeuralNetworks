import numpy as np
import os
from PIL import Image

DIRECTORY = '/Users/Piotr/Workspace/NeuralNetworks/data/data_set/'
IMAGE_ROW_SHAPE = (70,)


def get_label_from_filename(filename):
    return int(filename.split('_')[0])


def parse_images(path):
    return [(np.asarray(Image.open(path + filename).convert('1')).astype(int).reshape(IMAGE_ROW_SHAPE),
            get_label_from_filename(filename)) for filename in os.listdir(path)]


def list_to_arrays(data):
    images = np.empty((len(data), IMAGE_ROW_SHAPE[0]))
    labels = np.empty((len(data)))
    for i in range(len(data)):
        images[i, :] = data[i][0]
        labels[i] = data[i][1]

    return images, labels


images = parse_images(DIRECTORY)
train_x, train_y = list_to_arrays(images)

print(train_x.shape)
print(train_y.shape)


