# python
import os
import random
# 3rd party
import numpy as np
# project
from trainLoad import load_data, load_labels
from nodes import Dense, Sigmoid, SquareLoss


def forward_net(
    neuro_net_list,
    x_data
):
    x_data = x_data.flatten()
    for layer in neuro_net_list:
        x_data = layer.forward(x_data)

    return x_data

def backward_net(
    neuro_net_list,
    y_data,
    learning_rate
):
    loss_data = y_data
    for i in range(len(neuro_net_list) - 1, -1, -1):
        loss_data = neuro_net_list[i].backward(loss_data, learning_rate)


def SGD_train_example(
    neuro_net_list,
    x_batch, 
    y_batch, 
    learning_rate
):
    if x_batch.shape[0] != y_batch.shape[0]:
        raise 'batch size of x and y not equal'

    for example_idx in range(x_batch.shape[0]):
        x_example = x_batch[example_idx]
        y_example = y_batch[example_idx]

        forward_net(neuro_net_list, x_example)

        backward_loss = y_example
        for layer_idx in range(len(neuro_net_list) - 1, -1, -1):
            backward_loss, delta_params = neuro_net_list[layer_idx].backward(
                backward_loss
            )
            layer_params = neuro_net_list[layer_idx].get_params()

            for param_name, delta in delta_params.items():
                layer_params[param_name] += -delta * learning_rate


def BATCH_train_example(
    neuro_net_list,
    x_batch, 
    y_batch, 
    learning_rate
):
    if x_batch.shape[0] != y_batch.shape[0]:
        raise 'batch size of x and y not equal'
    batch_size = x_batch.shape[0]

    delta_weights_dict = {}

    for layer_idx in range(len(neuro_net_list)):
        params = neuro_net_list[layer_idx].get_params()
        if len(params) != 0:
            delta_params = {}
            for param_name, param in params.items():
                delta_params[param_name] = np.zeros(param.shape)
            delta_weights_dict[layer_idx] = delta_params

    for example_idx in range(x_batch.shape[0]):
        x_example = x_batch[example_idx]
        y_example = y_batch[example_idx]

        forward_net(neuro_net_list, x_example)

        backward_loss = y_example
        for layer_idx in range(len(neuro_net_list) - 1, -1, -1):
            backward_loss, delta_params = neuro_net_list[layer_idx].backward(
                backward_loss
            )
            layer_params = neuro_net_list[layer_idx].get_params()

            for param_name, delta in delta_params.items():
                delta_weights_dict[layer_idx][param_name] += -delta * learning_rate

    for layer_idx, deltas in delta_weights_dict.items():
        for param_name, delta in deltas.items():
            neuro_net_list[layer_idx].get_params()[param_name] += delta


def loss_classifier(
    neuro_net_list,
    x_data,
    y_data
):
    right_labels = 0

    for i in range(x_data.shape[0]):
        x_example = x_data[i]
        y_label = y_data[i] 
        y_net = forward_net(neuro_net_list, x_example)

        if y_net.argmax() == y_label.argmax():
            right_labels += 1

    return right_labels / x_data.shape[0]


if __name__ == '__main__':
    LEARNING_RATE = 0.01
    EPOCHS = 1010
    BATCH_SIZE = 40

    x_train = load_data( # (data_size, 28, 28)
        os.path.join('resources', 'train-images.idx3-ubyte')
    )
    y_train = load_labels( # (data_size, 10)
        os.path.join('resources', 'train-labels.idx1-ubyte')
    )

    x_test = load_data(
        os.path.join('resources', 't10k-images.idx3-ubyte')
    )
    y_test = load_labels(
        os.path.join('resources', 't10k-labels.idx1-ubyte')
    )

    train_size = x_train.shape[0]

    neuronet = list()
    neuronet.append(Dense(784, 100))
    neuronet.append(Sigmoid())

    neuronet.append(Dense(100, 10))
    neuronet.append(Sigmoid())

    neuronet.append(SquareLoss())

    for epoch_i in range(EPOCHS):
        random_batch = np.random.randint(train_size, size=BATCH_SIZE)
        x_batch = x_train[random_batch,:,:]
        y_batch = y_train[random_batch,:]

        rand_idx = random.randrange(0, x_train.shape[0])
        x_img = x_train[rand_idx]
        y_labels = y_train[rand_idx]

        SGD_train_example(
            neuronet,
            x_batch,
            y_batch,
            LEARNING_RATE
        )
        if epoch_i % 100 == 0 and epoch_i != 0:
            loss = loss_classifier(
                neuronet,
                x_test,
                y_test
            )
            print(f'epoch:{epoch_i}, loss:{loss}')
    
