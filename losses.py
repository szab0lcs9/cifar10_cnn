import numpy as np

# Loss functions
def MSE(y, y_hat):
    return np.mean(np.power(y - y_hat, 2))

def MSE_prime(y, y_hat):
    return 2 * (y_hat - y) / np.size(y)

def MAE(y, y_hat):
    return np.mean(np.abs(y - y_hat))

def MAE_prime(y, y_hat):
    return np.sign(y_hat - y) / np.size(y)

def CategoricalCrossentropy(y, y_hat):
    return -np.sum(y * np.log(y_hat))

def CategoricalCrossentropy_prime(y, y_hat):
    return -y / y_hat

def BinaryCrossentropy(y, y_hat):
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def BinaryCrossentropy_prime(y, y_hat):
    return -(y / y_hat) + ((1 - y) / (1 - y_hat))

