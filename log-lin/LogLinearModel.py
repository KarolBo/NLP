import random
import sys
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class LogLinearModel:
    def __init__(
        self,
        feature_function: Callable,
        learning_rate: float,
        iterations: int,
        loss: Callable,
        gradient_loss: Callable,
        verbose: bool,
    ):
        """
        Parameters
        ---
        feature_function : Callable
            Feature function mapping from X x Y -> R^m
        learning_rate : float
            Learning rate parameter eta for gradient descent
        iterations : int
            Number of iterations to run gradient descent for during `fit`
        loss : Callable
            Loss function to be used by this LogLinearModel instance as
            a function of the parameters and the data X and y
        gradient_loss : Callable
            Closed form gradient of the `loss` function used for gradient descent as
            a function of the parameters and the data X and y
        verbose : bool
            Verbosity level of the class. If verbose == True,
            the class will print updates about the gradient
            descent steps during `fit`

        """
        self.feature_function = feature_function
        self.theta = None
        self.alpha = learning_rate
        self.iterations = iterations
        self.loss = loss
        self.gradient_loss = gradient_loss
        self.verbose = verbose

    def gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """Performs one gradient descent step, and update parameters inplace.

        Parameters
        ---
        X : np.ndarray
            Data matrix
        y : np.ndarray
            Binary target values

        Returns
        ---
        None

        """
        grad = self.gradient_loss(X, y, self.theta)
        self.theta -= self.alpha * grad

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits LogLinearModel class using gradient descent.

        Parameters
        ---
        X : np.ndarray
            Input data matrix
        y : np.ndarray
            Binary target values

        Returns
        ---
        None

        """
        self.theta = np.array(X.shape[1] * [0.0])
        for i in range(self.iterations):
            self.gradient_descent(X, y)
            if self.verbose:
                print('step:', i+1, 'loss:', self.loss(X, y, self.theta))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary target labels for input data `X`.

        Parameters
        ---
        X : np.ndarray
            Input data matrix

        Returns
        ---
        np.ndarray
            Predicted binary target labels

        """
        if self.theta is None:
            print('The model is not trained!')
            return;
        z = np.dot(self.theta, np.transpose(x))
        return 1/(1 + np.exp(-z))


###############################################################################################


def feature_function(X, y):
    return X, y

def loss(X, Y, theta):
    z = np.dot(theta, np.transpose(X))
    Y_pred = 1/(1 + np.exp(-z))
    ce = -Y * np.log(Y_pred) - (1-Y) * np.log(1 - Y_pred)
    return np.sum(ce) / ce.shape[0]

def gradient_loss(X, Y, theta):
    z = np.dot(theta, np.transpose(X))
    Y_pred = 1/(1 + np.exp(-z))
    diffs = Y_pred - Y
    return np.dot(np.transpose(X), diffs) / diffs.shape[0]

# Prepare dataset
x, y = make_classification(random_state=101)
print(x)
print(y.shape)

# model = LogLinearModel(feature_function, 1, 100, loss, gradient_loss, True)
# model.fit(x, y)
# pred = model.predict(x)
# acc = accuracy_score(y, np.rint(pred))
# print(acc)