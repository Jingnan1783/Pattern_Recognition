"""Linear model base class."""

import abc
import numpy as np
import six


@six.add_metaclass(abc.ABCMeta)
class LinearModel(object):
    """Abstract class for linear models."""
    def __init__(self, ndims, w_init='zeros'):
        """Initialize a linear model.

        This function prepares an uninitialized linear model.
        It will initialize the weight vector, self.w, based on the method
        specified in w_init.

        We assume that the last index of w is the bias term, self.w = [w,b]

        self.w(numpy.ndarray): array of dimension (n_dims+1,)

        w_init needs to support:
          'zeros': initialize self.w with all zeros.
          'ones': initialze self.w with all ones.
          'uniform': initialize self.w with uniform random number between [0,1)

        Args:
            ndims(int): feature dimension
            w_init(str): types of initialization.
        """
        self.ndims = ndims
        self.w_init = w_init
        if self.w_init == 'zeros':
            self.w = np.zeros(self.ndims + 1)
        elif self.w_init == 'ones':
            self.w = np.ones(self.ndims + 1)
        elif self.w_init == 'uniform':
            self.w = np.random.uniform(size=self.ndims + 1)
        self.x = None

    def forward(self, x):
        """Forward operation for linear models.
        Performs the forward operation, f=w^Tx, and return f.
        Args:
            x(numpy.ndarray): Dimension of (N, ndims), N is the number
              of examples.
        Returns:
            f(numpy.ndarray): Dimension of (N,)
        """
        N, ndims = x.shape
        self.x = np.concatenate((x, np.ones((N, 1))), axis=1)
        f = np.dot(self.x, self.w)
        return f

    @abc.abstractmethod
    def backward(self, f, y):
        """Do not need to be implemented here."""
        pass

    @abc.abstractmethod
    def loss(self, f, y):
        """Do not need to be implemented here."""
        pass

    @abc.abstractmethod
    def predict(self, f):
        """Do not need to be implemented here."""
        pass
