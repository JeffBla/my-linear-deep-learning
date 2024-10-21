import numpy as np


class ActivationFn:

    @classmethod
    def forward(self, Z: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def backward(self, dA: np.ndarray) -> np.ndarray:
        pass


class ReLU(ActivationFn):

    @classmethod
    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.cache = Z
        return np.maximum(0, Z)

    @classmethod
    def backward(self, dA: np.ndarray, Z) -> np.ndarray:
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ


class Sigmoid(ActivationFn):

    @classmethod
    def forward(self, Z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-Z))

    @classmethod
    def backward(self, dA: np.ndarray, Z) -> np.ndarray:
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        return dZ
