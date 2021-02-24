import numpy as np
import pandas as pd

from scipy.special import expit

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt


class BinaryLogisticRegression:
    """
    Binary Logistic regression class from scratch.
    """
    def __init__(self, n):
        """
        Initializes a logistic regression class object with ``n`` weights.

        :param n: The number of weights to initialize.
        """

        self.weights = np.random.random(size=(n, 1))
        self.bias = np.random.random(size=1)

    def infer(self, x):
        """
        Run logistic regression on input vector. Outputs probabilities, not class predictions.
        
        :param x: A numpy array representing the values of the independent variables.
        :return: Probability of dependent class being equal to the positive case.
        """
        return self.sigmoid(np.dot(x, self.weights) + self.bias)

    def train(self, X, y, epochs, lr=1e-4, print_every=5):
        """
        Train this class instances weights based on training data X and y.

        :param X: Independent variables, one sample per row. X should be 2 dimensions.
        :param y: Binary target variable, one sample per row.
        :param epochs: Number of times the model will see the data
        :param lr: Learning rate.
        :param print_every: How often to print loss stats.
        :return: Returns an array containing the average loss for each epoch. 
        """

        loss = []
        for e in range(1, epochs + 1):
            y_hat = self.infer(X)
            y = y.reshape(-1, 1)
            self._step(y, y_hat, X, lr)

            epoch_loss = self.binary_cross_entropy(y, y_hat)
            loss.append(epoch_loss.mean())
            if e % print_every == 0:
                print(f'Loss after {e} epochs: {loss[-1]}')

        return loss

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid (logistic function) applied to x.
        :param x: Input.
        :return:
        """
        return expit(x)

    @staticmethod
    def binary_cross_entropy(p, q):
        """
        Cross entropy loss for a single sample.
        :param p: The true probability q(X=1). Since the classification is binary
            this should value should only be 0 or 1.
        :param q: The model's estimated probability of q(X=1).
        :return: Cross entropy of distribution q with respect to distribution p. 
        """
        epsilon = 1e-15 # to avoid divide by zero errors.
        return -p * np.log(q + epsilon) - (1 - p) * np.log(1 - q + epsilon)

    def _step(self, y, y_hat, X, lr):
        """
        Back propagation step.
        """

        delta_y = (y_hat - y).squeeze()
        dLdw = (X.T * delta_y).T.mean(axis=0)
        dLdb = delta_y.mean()
        self.weights -= lr * dLdw.reshape(self.weights.shape)
        self.bias -= lr * dLdb


if __name__ == '__main__':
    print("Hello!")