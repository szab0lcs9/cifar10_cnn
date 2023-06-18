import numpy as np
import cupy as cp
from layer import Layer

class Pooling(Layer):
    def __init__(self, pool_size, stride):
        self.input = None
        self.pool_size = pool_size
        self.stride = stride
        self.indices = None

    def forward(self, input):
        self.input = input

        return self.maxPooling(self.input, self.pool_size, self.stride)

    def backward(self, output_gradient, learning_rate):
        output_height, output_width, output_depth = output_gradient.shape
        input_height, input_width, input_depth = self.input.shape
        input_gradient = np.zeros(self.input.shape)

        output_gradient = np.reshape(output_gradient, (output_height * output_width * output_depth))

        for i in range(len(self.indices)):
                indices = self.indices[i]
                input_gradient[indices] = output_gradient[i]
            
        return input_gradient
    
    def maxPooling(self, input, pool_size, stride):
        input_height, input_width, input_depth = input.shape
        pool_width = pool_size
        pool_height = pool_size
        output_depth = input_depth
        output_height = input_height // pool_height
        output_width = input_width // pool_width

        output = np.zeros((output_height, output_width, output_depth))
        self.indices = ()
        
        for i in range(output_depth):
            for j in range(output_height):
                for k in range(output_width):
                    input_slice = input[j * stride: j * stride + pool_height, k * stride: k * stride + pool_width, i]
                    max_value = np.max(input_slice)
                    output[j, k, i] = max_value
                    index_is_added = False
                    # Save the indices of the max values
                    for m in range(input_slice.shape[0]):
                        for n in range(input_slice.shape[1]):
                            if input_slice[m, n] == max_value and index_is_added == False:
                                self.indices += ((j * stride + m, k * stride + n, i),)
                                index_is_added = True

        return output