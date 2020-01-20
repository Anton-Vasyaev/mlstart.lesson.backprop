# python
import os
import random
# 3rd party
import cv2 as cv
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


def train_example(
    neuro_net_list,
    x_data, 
    y_data, 
    learning_rate
):
    forward_net(neuro_net_list, x_data)
    backward_net(neuro_net_list, y_data, learning_rate)
    

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
    BATCH_SIZE = 20
    EPOCHS = 2000

    x_train = load_data(
        os.path.join('resources', 'train-images.idx3-ubyte')
    )
    y_train = load_labels(
        os.path.join('resources', 'train-labels.idx1-ubyte')
    )

    x_test = load_data(
        os.path.join('resources', 't10k-images.idx3-ubyte')
    )
    y_test = load_labels(
        os.path.join('resources', 't10k-labels.idx1-ubyte')
    )

    neuronet = list()
    neuronet.append(Dense(784, 100))
    neuronet.append(Sigmoid())

    neuronet.append(Dense(100, 10))
    neuronet.append(Sigmoid())

    neuronet.append(SquareLoss())

    for epoch_i in range(EPOCHS):
        for batch_i in range(BATCH_SIZE):
            rand_idx = random.randrange(0, x_train.shape[0])

            x_img = x_train[rand_idx]
            y_labels = y_train[rand_idx]

            train_example(
                neuronet,
                x_img,
                y_labels,
                LEARNING_RATE
            )
        if epoch_i % 100 == 0:
            loss = loss_classifier(
                neuronet,
                x_test,
                y_test
            )
            print(f'epoch:{epoch_i}, loss:{loss}')
        
    loss = loss_classifier(
        neuronet,
        x_test,
        y_test
    )
    print(f'loss after train:{loss}')
    