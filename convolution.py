import sys
import numpy as np
import cupy as cp
from layer import Layer
from scipy import signal
np.set_printoptions(threshold=sys.maxsize)

from activation_switcher import Activation_switcher as switcher

class Convolutional(Layer):

        def __init__(self, num_filters, kernel_size, input_shape, stride, padding, activation):

            input_height, input_width, input_channels = input_shape
            kernel_height, kernel_width = kernel_size

            self.num_filters = num_filters
            self.kernel_size = kernel_size
            self.kernel_height = kernel_height
            self.kernel_width = kernel_width
            self.input_shape = input_shape
            self.input_height = input_height
            self.input_width = input_width
            self.input_channels = input_channels
            self.output = None
            self.padded_output = None
            self.stride = stride
            self.padding = padding
            self.weights = None
            self.bias = None
            self.activation = switcher.activation_switcher(activation)

        def forward(self, input):
            self.input = input
            input_height, input_width, input_channels = input.shape

            #print('Convolutional forward input shape: ', input.shape)
            self.weights = np.random.randn(self.kernel_height, self.kernel_width, self.num_filters)
            self.bias = np.random.randn(self.num_filters)

            # Compute the output tensor
            output_height = int((input_height - self.kernel_height + 2 * self.padding) / self.stride + 1)
            output_width = int((input_width - self.kernel_width + 2 * self.padding) / self.stride + 1)
            output_depth = int(self.num_filters)
            self.output = np.zeros((output_height, output_width, output_depth))
            self.padded_output = np.zeros((output_height + 2 * self.padding, output_width + 2 * self.padding, input_channels))

            # Add padding to the input tensor
            if self.padding > 0:
                for i in range(input_channels):
                    padded_input = np.reshape(input[:, :, i], (input_height, input_width))
                    padded_input = np.pad(padded_input, (self.padding, self.padding))
                    self.padded_output[:, :, i] = np.reshape(padded_input, (input_height + 2 * self.padding, input_width + 2 * self.padding))


            #print('padded out shape: ', self.padded_output.shape)
            #print('output shape: ', self.output.shape)
            #print('weights shape: ', self.weights.shape)

            # Apply convolution
            for i in range(output_depth):
                for j in range(output_height):
                    for k in range(output_width):
                        weights_slice = np.rot90(self.weights[:, :, i], 2, (0, 1))
                        output_slice = self.padded_output[j * self.stride: j * self.stride + self.kernel_height, k * self.stride: k * self.stride + self.kernel_width, :]
                        self.output[j, k, i] = np.sum(np.dot(weights_slice, output_slice)) + self.bias[i]


            # Fill up the remaining channels with random numbers
            for i in range(input_channels, output_depth):
                for j in range(output_height):
                    for k in range(output_width):
                        self.output[j, k, i] = np.random.randn()
       
            # Apply the activation function
            for i in range(output_depth):
                for j in range(output_height):
                    for k in range(output_width):
                        self.output[j, k, i] = self.activation.forward(self.output[j, k, i])

            #print('Convolutional forward output shape: ', self.output.shape)
            return self.output

        def backward(self, output_gradient, learning_rate):
            output_width, output_height, output_depth = output_gradient.shape
            input_width, input_height, input_depth = self.input.shape

            # Compute the gradient for activation function
            input_gradient = np.zeros(self.input.shape)
            weights_gradient = np.zeros(self.weights.shape)
            bias_gradient = np.zeros(self.bias.shape)
            padded_output_gradient = np.zeros((output_height + 2 * self.padding, output_width + 2 * self.padding, output_depth))


            for i in range(output_depth):
                for j in range(output_height):
                    for k in range(output_width):
                        output_gradient[j, k, i] = self.activation.backward(output_gradient[j, k, i], learning_rate)

            for i in range(output_depth):
                    padded_output = np.reshape(output_gradient[:, :, i], (output_height, output_width))
                    padded_output = np.pad(padded_output, (self.padding, self.padding))
                    padded_output_gradient[:, :, i] = np.reshape(padded_output, (input_height + 2 * self.padding, input_width + 2 * self.padding))

            # Compute the gradient of the weights and bias
            for i in range(input_depth):
                for j in range(output_height):
                    for k in range(output_width):
                        weights_slice = np.rot90(padded_output_gradient[j * self.stride: j * self.stride + self.kernel_height, k * self.stride: k * self.stride + self.kernel_width, i], 2, (0, 1))
                        padded_output_slice = self.padded_output[j * self.stride: j * self.stride + self.kernel_height, k * self.stride: k * self.stride + self.kernel_width, i]
                        weights_gradient[:, :, i] += np.sum(np.dot(padded_output_slice, weights_slice))
                        bias_gradient[i] += output_gradient[j, k, i]
                        input_gradient[:, :, i] += np.sum(np.dot(weights_gradient[:, :, i], padded_output_slice))
            self.weights -= learning_rate * weights_gradient
            self.bias -= learning_rate * bias_gradient
            
            return input_gradient

