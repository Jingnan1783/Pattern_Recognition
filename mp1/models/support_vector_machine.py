"""
Implements support vector machine.
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class SupportVectorMachine(LinearModel):
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
        mask = f * y < 1
        gradient = -(mask * y).reshape((f.shape[0],1)) * self.x
        return self.w + np.sum(gradient, axis=0) / f.shape[0]

    def loss(self, f, y):
        """The average loss across batch examples.
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
            y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
            (float): average hinge loss.
        """
        hinge = 1 - y * f
        return np.sum(np.where(hinge > 0, hinge, 0), axis=0) / f.shape[0]


    def predict(self, f):
        """
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,).
        """
        y_predict = f > 0
        y_predict = y_predict * 2 -1
        return y_predict
