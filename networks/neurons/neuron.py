import numpy as np

from neurons.activations import get_activation


class Neuron:
    def __init__(self, number_of_inputs, weights_range, bias, activation):
        self.number_of_inputs = number_of_inputs
        self.weights = np.random.uniform(*weights_range, number_of_inputs)
        self.bias = bias
        self.activation = get_activation(activation)

    def predict(self, features):
        return self.activation(self.raw_output(features))

    def raw_output(self, features):
        features_weights_product = self.bias
        for i in range(self.number_of_inputs):
            features_weights_product += self.weights[i] * features[i]
        return features_weights_product

    @property
    def get_number_of_inputs(self):
        return self.number_of_inputs

    @property
    def get_activation(self):
        return self.activation
