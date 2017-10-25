"""
Implements lienar regression.
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class LinearRegression(LinearModel):
    """Implements a linear regression mode model"""

    def backward(self, f, y):
        """Performs the backward operation.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
            y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
            (numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,).
        """
        #1/N(sum((f-y)*x))
        diff = f - y
        return np.sum(np.multiply(diff.reshape((f.shape[0], 1)), self.x), axis=0) / f.shape[0]

    def loss(self, f, y):
        """The average loss across batch examples.
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
            y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
            (float): average square loss.
        """
        return np.sum((y - f)**2, axis=0) / f.shape[0]

    def predict(self, f):
        """Converts score into predictions in {-1, 1}.
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,).
        """
        y_predict = f > 0
        y_predict = y_predict * 2 -1
        return y_predict
