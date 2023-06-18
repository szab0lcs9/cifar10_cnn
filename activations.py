from activation import Activation
import numpy as np

# Activation functions and their derivatives
class LeakyReLU(Activation):
    def __init__(self):
        super().__init__(self.leaky_relu, self.derivative_leaky_relu)

    def leaky_relu(self, x):
        if x > 0:
            return x
        else:
            return 0.01 * x

    def derivative_leaky_relu(self, y):
        if y > 0:
            return 1
        elif y < 0:
            return 0.01
        else:
            return None
        
class ReLU(Activation):
    def __init__(self):
        super().__init__(self.relu, self.derivative_relu)

    def relu(self, x):
        if x > 0:
            return x
        else:
            return 0

    def derivative_relu(self, y):
        if y > 0:
            return 1
        else:
            return 0
        
class Sigmoid(Activation):
    def __init__(self):
        super().__init__(self.sigmoid, self.derivative_sigmoid)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_sigmoid(self, y):
        return y * (1 - y)

class Tanh(Activation):
    def __init__(self):
        super().__init__(self.tanh, self.derivative_tanh)

    def tanh(self, x):
        return np.tanh(x)

    def derivative_tanh(self, y):
        return 1 - y * y
    
class Softmax(Activation):
    def __init__(self):
        super().__init__(self.softmax, self.derivative_softmax)

    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def derivative_softmax(self, y):
        s = y.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)
    
