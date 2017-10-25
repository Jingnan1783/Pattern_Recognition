"""Main function for train, eval, and test.
"""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from models.linear_regression_tf import LinearRegressionTf
from models.logistic_regression_tf import LogisticRegressionTf
from models.support_vector_machine_tf import SupportVectorMachineTf

from train_eval_model_tf import train_model, eval_model
from utils.io_tools import read_dataset
from utils.data_tools import preprocess_data

#flags = tf.app.flags
#FLAGS = flags.FLAGS
#flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
#flags.DEFINE_string('feature_type', 'default', 'Feature type, supports ['']')
#flags.DEFINE_string('model_type', 'linear', 'Feature type, supports ['']')


def main(_):
    """High level pipeline.

    This script performs the training, evaling and testing state of the model.
    """
#    learning_rate = FLAGS.learning_rate
#    feature_type = FLAGS.feature_type
#    model_type = FLAGS.model_type

    feature_type = 'default'
    model_type = 'linear'
    # Load dataset.
    data = read_dataset('data/train_lab.txt', 'data/image_data')

    # Data Processing.
    data = preprocess_data(data, 'default')
    print("Finish preprocessing...")

    # Initialize model.
    ndim = data['image'].shape[1]
    if model_type == 'linear':
        model = LinearRegressionTf(ndim, 'gaussian')
    elif model_type == 'logistic':
        model = LogisticRegressionTf(ndim, 'uniform')
    elif model_type == 'svm':
        model = SupportVectorMachineTf(ndim, 'uniform')

    # Train Model.
    print("Start to train the model...")
    model = train_model(data, model)

    # Eval Model.
    print("Start to evaluate the model...")
    data_val = read_dataset('data/val_lab.txt', 'data/image_data')
    data_val = preprocess_data(data_val, feature_type)
    loss, acc = eval_model(data_val, model)
    print(loss, acc)
#    # Test Model.
#    data_test = read_dataset('data/test_lab.txt', 'data/image_data')
#    data_test = preprocess_data(data_test, feature_type)
    
    # Generate Kaggle output.


if __name__ == '__main__':
    tf.app.run()
