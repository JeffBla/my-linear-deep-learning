import h5py
import numpy as np
import h5py


class dataset:

    def __init__(self, raw_train_filename, raw_test_filename):
        raw_train = h5py.File(raw_train_filename, 'r')
        raw_test = h5py.File(raw_test_filename, 'r')

        train_set_x_orig = np.array(raw_train['train_set_x'])
        train_set_y = np.array(raw_train['train_set_y'])
        self.train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))

        test_set_x_orig = np.array(raw_test['test_set_x'])
        test_set_y = np.array(raw_test['test_set_y'])
        self.test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

        train_set_x_flatten = train_set_x_orig.reshape(
            train_set_x_orig.shape[0], -1).T
        test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],
                                                     -1).T
        self.train_set_x = train_set_x_flatten / 255.
        self.test_set_x = test_set_x_flatten / 255.
        self.classes = np.array(raw_train['list_classes'])

    def get_train_set(self):
        return self.train_set_x, self.train_set_y

    def get_test_set(self):
        return self.test_set_x, self.test_set_y
