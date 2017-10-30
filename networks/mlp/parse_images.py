import numpy as np
import os
from PIL import Image
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(range(10))

DIRECTORY = '/home/piotr/Workspace/MachineLearning/NeuralNetworks/data/digits/'
IMAGE_ROW_SHAPE = (70,)


def get_label_from_filename(filename):
    return int(filename.split('_')[0])


def parse_images(path):
    l = []
    for filename in os.listdir(path):
        img = np.asarray(Image.open(path + filename).convert('1')).astype(int)
        if img.shape == (10, 7):
            l.append((img.reshape(IMAGE_ROW_SHAPE), get_label_from_filename(filename)))
        else:
            print("Shape = {}, index = {}".format(img.shape, filename))

    return l
    # return [(np.asarray(Image.open(path + filename).convert('1')).astype(int).reshape(IMAGE_ROW_SHAPE),
    #         get_label_from_filename(filename)) for filename in os.listdir(path)]


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

training_data = []
print(train_y[0])
ty = lb.transform(train_y)
print()
for x, y in zip(train_x, ty):
    training_data.append(((np.expand_dims(x, axis=1)), np.expand_dims(y, axis=1)))

print(len(training_data))
np.save('images.npy', training_data)