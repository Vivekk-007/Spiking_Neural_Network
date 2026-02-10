import numpy as np
from tensorflow.keras.datasets import mnist

def load_mnist_data():
    """
    Loads MNIST dataset and normalizes pixel values.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    return x_train, y_train, x_test, y_test
