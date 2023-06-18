import sys
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical

from dense import Dense
from convolution import Convolutional
from pool import Pooling
from flatten import Flatten
from losses import CategoricalCrossentropy, CategoricalCrossentropy_prime

# Load CIFAR10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Constants
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
INPUT_SHAPE = X_train.shape[1:]
KERNEL_SIZE = (3, 3)
STRIDE = 1
PADDING = 0

def print_progress_bar(index, total, label):
    n_bar = 50  # Progress bar width
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress + 1)}%  {label}")
    sys.stdout.flush()    


def reduce_quantity(data, quantity):
    return data[:quantity]

# Transform label indices to one-hot encoded vectors
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Reduce quantity of data
X_train = reduce_quantity(X_train, 5000)
X_test = reduce_quantity(X_test, 1000)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalization of pixel values (to [0-1] range)
X_train /= 255
X_test /= 255


# create network
network = [
    Convolutional(16, KERNEL_SIZE, INPUT_SHAPE, 1, 1, 'leaky_relu'),
    Pooling(2, 2),
    Convolutional(32, KERNEL_SIZE, INPUT_SHAPE, 1, 1, 'leaky_relu'),
    Pooling(2, 2),
    Flatten(),
    Dense(64, 'relu'),
    Dense(10, 'softmax'),
]

epochs = 10
learning_rate = 0.001

# train network
for epoch in range(epochs):
    error = 0
    i = 0
    for x, y in zip(X_train, y_train):
        # forward pass
        output = x/2
        print_progress_bar(i, len(X_train), 'Training in progress...')
        for layer in network:
            output = layer.forward(output)
        error += CategoricalCrossentropy(y, output)
        # backward pass
        grad = CategoricalCrossentropy_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
        i += 1
    error /= len(X_train)

    print(f"\nEpoch {epoch + 1}/{epochs}, error={error:.3f}")

# test
correct = 0
for x, y in zip(X_test, y_test):
    output = x
    for layer in network:
        output = layer.forward(output)
    correct += int(np.argmax(output) == np.argmax(y))
print(f" Accuracy: {correct/len(X_test):.3f}")

