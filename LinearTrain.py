import numpy as np
import random
from tqdm import tqdm

from ActivationFn import *
from Dataset import dataset
from Linear import Linear
from LossFn import BinaryCrossEntropyLossFn


class model:

    def __init__(self, num_layers, image_size, channel, num_class, lr, loss_fn,
                 train_set_x, train_set_y):
        self.num_layers = num_layers
        self.layerdims = image_size * image_size * channel
        self.layers = [
            Linear(self.layerdims, self.layerdims, lr, ReLU)
            for i in range(num_layers - 1)
        ]
        self.layers.append(Linear(self.layerdims, num_class, lr, Sigmoid))
        self.loss_fn = loss_fn
        self.costs = []
        # DataSet
        self.train_X = train_set_x
        self.train_Y = train_set_y

    def train(self, X, Y, num_iterations, logfile):
        for i in tqdm(range(num_iterations), "Iter"):
            A = X
            # Forward
            for i in tqdm(range(self.num_layers), "Forward"):
                A, _ = self.layers[i].forward(A)
            cost = self.loss_fn.forward(A, Y)
            # Backward
            dA = self.loss_fn.backward(A, Y)
            for i in tqdm(range(self.num_layers - 1, -1, -1), "Backward"):
                dA, _ = self.layers[i].backward(dA)
            # Update
            for i in tqdm(range(self.num_layers), "Update"):
                self.layers[i].update_parameters()
            # Record
            self.costs.append(cost)
            # if i % 100 == 0:
            with open(logfile, "a") as f:
                f.write(str(cost) + "\n")


if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)
    dataset = dataset('datasets/train_catvnoncat.h5',
                      'datasets/test_catvnoncat.h5')
    train_set_x, train_set_y = dataset.get_train_set()
    test_set_x, test_set_y = dataset.get_test_set()

    model = model(10, 64, 3, 2, 0.01, BinaryCrossEntropyLossFn, train_set_x,
                  train_set_y)
    model.train(train_set_x, train_set_y, 1000, "log.txt")
