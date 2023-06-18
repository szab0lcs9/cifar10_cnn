import numpy as np
import cupy as cp

class Flatten():
    def __init__(self):
        self.input = None
        self.input_height = None
        self.input_width = None
        self.input_depth = None


    def forward(self, input):
        self.input = input
        self.input_height, self.input_width, self.input_depth = input.shape
        output = np.zeros((self.input_height * self.input_width * self.input_depth))

        output = np.reshape(self.input, (self.input_height * self.input_width * self.input_depth))

        return output
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = np.reshape(output_gradient, (self.input_height, self.input_width, self.input_depth))
        return input_gradient
