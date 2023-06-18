import sys
import numpy as np
import cupy as cp
from layer import Layer
from activation_switcher import Activation_switcher as switcher
np.set_printoptions(threshold=sys.maxsize)

# Fully connected layer
class Dense(Layer):
    def __init__(self, neurons, activation):
        self.neurons = neurons
        self.activation = switcher.activation_switcher(activation)
        self.activation_string = switcher.get_attribute(activation)
        self.weights = None
        self.bias = None

    def forward(self, input):
        self.input = input
        self.input_size = len(input)
        self.weights = np.random.randn(self.input_size, self.neurons) * 0.01
        self.bias = np.random.randn(1, self.neurons)

        self.output = np.dot(self.input, self.weights) + self.bias
        self.output = np.reshape(self.output, (self.neurons))

        if self.activation_string == 'softmax':
            self.output = self.activation.forward(self.output)
        else:
            for i in range(len(self.output)):
                self.output[i] = self.activation.forward(self.output[i])

        return self.output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input[np.newaxis].T, output_gradient[np.newaxis])
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        input_gradient = np.dot(self.weights, output_gradient)

        return input_gradient
