# python
import math
# 3rd party
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class SquareLoss:
    def __init__(self):
        pass


    def forward(self, input_data):
        self.input_data = input_data.copy()
        return input_data.copy()


    def backward(self, loss, lr):
        if loss.shape != self.input_data.shape:
            raise('wrong shape')

        loss = -(self.input_data - loss)

        return loss


class Sigmoid:
    def __init__(self):
        self.vsigmoid = np.vectorize(sigmoid)


    def forward(self, input_data):
        self.input_data = self.vsigmoid(input_data)

        return self.input_data.copy()


    def backward(self, loss, lr):
        loss = (
            self.input_data * 
            (1 - self.input_data)
        ) * loss

        return loss


class Dense:
    def __init__(
        self, 
        input_size, 
        output_size
    ):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.rand(
            self.input_size,
            self.output_size
        ) 
        self.weights -= 0.5
        self.weights / 500.0
        self.biases = np.zeros(self.output_size)


    def forward(self, input_data):
        if len(input_data.shape) != 1 or input_data.shape[0] != self.input_size:
            raise 'wrong tensor shape of input'

        self.input_data = input_data.copy()

        return self.input_data.dot(self.weights) + self.biases

    
    def backward(self, loss, lr):
        if len(loss.shape) != 1 or loss.shape[0] != self.output_size:
            raise 'wrong tensor shape of loss'

        loss_lr = loss * lr

        delta_weights = np.outer(self.input_data, loss_lr)

        weights_t = self.weights.transpose(1, 0)

        self.weights += delta_weights
        self.biases += loss_lr

        return loss.dot(weights_t)
