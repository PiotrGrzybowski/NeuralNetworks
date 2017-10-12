import numpy as np

from neurons.activations import get_activation


class Perceptron:
    def __init__(self, inputs, weights_range, activation):
        self.inputs = inputs
        self.weights = np.random.uniform(*weights_range, inputs)
        self.bias = 1
        self.activation = get_activation(activation)

    def predict(self, features):
        return self.activation(self.raw_output(features))

    def raw_output(self, features):
        features_weights_product = self.bias
        for i in range(self.inputs):
            features_weights_product += self.weights[i] * features[i]
        return features_weights_product
