import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import model_selection
from operator import add, sub


class NN(object):

    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.n_layers = len(n_nodes)
        self.weights, self.bias = self.init_weights()

    def feed_forward(self, X, train=False):
        x = np.atleast_2d(X).T
        N = len(self.n_nodes)
        iters = np.arange(1, N)
        activations = [x]
        zs = []
        for i, b, w in zip(iters, self.bias, self.weights):
            a = self.activate_logistic(activations[i - 1], w, b)
            if train:
                activations.append(a)
                zs.append(w @ activations[i - 1] + b)

        if train:
            return activations, zs
        return a

    def back_prop(self, X, Y, debug=False):
        del_w = [np.zeros(w.shape) for w in self.weights]
        del_b = [np.zeros(b.shape) for b in self.bias]

        # feed forward:
        A, Z = self.feed_forward(X, train=True)

        # get error heuristic of output layer:
        delta_l = self.cost_derivative(Y, A[-1]) * self.sigmoid_prime(Z[-1])
        del_w[-1] = delta_l @ A[-2].T
        del_b[-1] = delta_l

        # back propagate:
        for i in np.arange(2, self.n_layers):
            sp = self.sigmoid_prime(Z[-i])
            delta_l = (self.weights[-i + 1].T @ delta_l) * sp
            del_b[-i] = delta_l
            del_w[-i] = delta_l @ A[-i - 1].T

        return del_w, del_b

    # helper functions --------------------
    def update(self, mini_batch, lr):
        N = len(mini_batch)
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_dw, delta_db = self.back_prop(x, y)
            nabla_w = list(map(add, nabla_w, delta_dw))
            nabla_b = list(map(add, nabla_b, delta_db))

        self.weights = list(map(sub, self.weights, map(lambda x: x / N, nabla_w)))
        self.bias = list(map(sub, self.bias, map(lambda x: x / N, nabla_b)))

    def activate_logistic(self, X, weights, bias):
        return expit(weights @ X + bias)

    def activate_softmax(self, X, weights, bias):
        Z = weights @ X + b
        return np.exp(Z[i]) / np.exp(Z).sum()

    def sigmoid_prime(self, X):
        return expit(X) * (1 - expit(X))

    # cross entropy loss derivative wrt final activation layer
    def cost_derivative(self, y, y_hat):
        return (y_hat - y) / (1e7 + y_hat * (1 - y_hat))

    def init_weights(self):
        weights = [np.random.randn(m, n) for m, n in zip(self.n_nodes[1:], self.n_nodes[:-1])]
        bias = [np.zeros(m).reshape(-1, 1) for m in self.n_nodes[1:]]
        return weights, bias
    # ------------------------------------


if __name__ == '__main__':

    net = NN((4, 3, 2))
    # for w in net.weights:
    #     print(w, '\n')
    # print('-------------------------')
    # for b in net.bias:
    #     print(b, '\n')
    x = np.array([1, 2, 3, 4])
    print('\n', net.feed_foward(x))
