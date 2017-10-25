"""Logistic regression model implemented in TensorFlow.
"""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from models.linear_model_tf import LinearModelTf


class LogisticRegressionTf(LinearModelTf):
    def loss(self, f, y):
        """The average loss across batch examples.
        Computes the average log loss.

        Args:
            f: Tensor containing the output of the forward operation.
            y(tf.placeholder): Tensor containing the ground truth label.
        Returns:
            (1): Returns the loss function tensor.
        """
        f = (f + 1) / 2
        y = (y + 1) / 2
        labels = (tf.reshape(y,[tf.shape(y)[0], 1]))
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=f,labels=labels)
        loss = tf.reduce_mean(entropy)
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
