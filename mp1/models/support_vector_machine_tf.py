"""Support vector machine model implemented in TensorFlow.
"""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from models.linear_model_tf import LinearModelTf


class SupportVectorMachineTf(LinearModelTf):
    def loss(self, f, y):
        """The average loss across batch examples.
        Computes the average hinge loss.

        Args:
            f: Tensor containing the output of the forward operation.
            y(tf.placeholder): Tensor containing the ground truth label.
        Returns:
            (1): Returns the loss function tensor.
        """
        loss = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(f, y))))
        return loss

    def predict(self, f):
        """Converts score into predictions in {-1, 1}
        Args:
            f: Tensor containing theoutput of the forward operation.
        Returns:
            (1): Converted predictions, tensor of the same dimension as f.
        """
        N = tf.shape(f)
        zeros = tf.zeros(N)
        ones = tf.ones(N)
        minus_ones = tf.scalar_mul(-1, ones)
        condition = tf.greater_equal(f, zeros)
        pred = tf.where(condition, ones, minus_ones)
        return pred
