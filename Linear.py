import numpy as np

from ActivationFn import *


class Linear:

    def __init__(self, in_feat, out_feat, lr, activationFn: ActivationFn):
        self.lr = lr
        self.activationFn = activationFn
        self.parameters = {
            "W": np.random.randn(out_feat, in_feat) * 0.01,
            "b": np.random.randn(out_feat, 1) * 0.01
        }
        self.cache = None
        self.grad = None

    def forward(self, X):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        n -- the index of this layer
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        
        Returns:
        AL -- last post-activation value
        cache -- Activation_cache & Linear_cache
        """
        W, b = self.parameters["W"], self.parameters["b"]
        Z = np.dot(W, X) + b
        Linear_cache = (X, W, b)
        A = self.activationFn.forward(Z)
        Activation_cache = Z
        self.cache = (Linear_cache, Activation_cache)
        return A, self.cache

    def backward(self, dA):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        n -- the index of this layer
        DA -- post-activation gradient for current layer l
        cache -- Activation_cache & Linear_cache

        Returns:
        grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ... 
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ... 
        """
        Linear_cache, Z = self.cache
        dZ = self.activationFn.backward(dA, Z)
        A_prev, W, b = Linear_cache
        m = A_prev.shape[1]
        dW = 1. / m * np.dot(dZ, A_prev.T)
        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        self.grad = {"dA": dA, "dZ": dZ, "dW": dW, "db": db}
        return dA_prev, self.grad

    def update_parameters(self):
        """
        Update parameters using gradient descent
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
                      parameters["W" + str(l)] = ... 
                      parameters["b" + str(l)] = ...
        """
        dW, db = self.grad["dW"], self.grad["db"]
        self.parameters["W"] -= self.lr * dW
        self.parameters["b"] -= self.lr * db
        return self.parameters
