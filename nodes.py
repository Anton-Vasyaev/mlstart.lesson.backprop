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


    def backward(self, loss):
        if loss.shape != self.input_data.shape:
            raise('wrong shape')

        loss = -(self.input_data - loss)

        return loss, {}

    
    def get_params(self):
        return {}



class Sigmoid:
    def __init__(self):
        self.vsigmoid = np.vectorize(sigmoid)


    def forward(self, input_data):
        self.input_data = self.vsigmoid(input_data)

        return self.input_data.copy()


    def backward(self, loss):
        loss = (
            self.input_data * 
            (1 - self.input_data)
        ) * loss

        return loss, {}

    
    def get_params(self):
        return {}



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

    
    def backward(self, loss):
        if len(loss.shape) != 1 or loss.shape[0] != self.output_size:
            raise 'wrong tensor shape of loss'

        delta_weights = np.outer(self.input_data, loss)
        delta_biases = loss.copy()

        weights_t = self.weights.transpose(1, 0)
        loss_out = loss.dot(weights_t)

        return loss_out, { 
            'w' : delta_weights, 
            'b' : delta_biases 
        }
    

    def get_params(self):
        return { 'w' : self.weights, 'b' : self.biases }
