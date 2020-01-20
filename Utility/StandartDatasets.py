import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist


def get_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    x_test.shape = x_test.shape + (1,)
    x_train.shape = x_train.shape + (1,)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train, x_test, y_test)