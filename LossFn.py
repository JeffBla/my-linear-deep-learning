import numpy as np


class BinaryCrossEntropyLossFn:

    @classmethod
    def forward(self, Y_hat: np.ndarray, Y: np.ndarray) -> float:
        """
        Y_hat -- predicted labels, numpy array of shape (1, number of examples)
        Y -- true labels, numpy array of shape (1, number of examples)
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
        return np.squeeze(cost)

    @classmethod
    def backward(self, Y_hat: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Y_hat -- predicted labels, numpy array of shape (1, number of examples)
        Y -- true labels, numpy array of shape (1, number of examples)
        """
        m = Y.shape[1]
        dA = -(np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
        return dA
