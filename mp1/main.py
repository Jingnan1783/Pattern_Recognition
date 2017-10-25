"""Main function for train, eval, and test.
"""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression
from models.support_vector_machine import SupportVectorMachine

from train_eval_model import train_model, eval_model
from utils.io_tools import read_dataset, write_dataset
from utils.data_tools import preprocess_data

#flags = tf.app.flags
#FLAGS = flags.FLAGS
#flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
#flags.DEFINE_integer('num_steps', 10000, 'Number of update steps to run.')
#flags.DEFINE_string('feature_type', 'default', 'Feature type, supports ['']')
#flags.DEFINE_string('model_type', 'linear', 'Feature type, supports ['']')


def main(_):
    """High level pipeline.
    This script performs the trainsing, evaling and testing state of the model.
    """
#    learning_rate = FLAGS.learning_rate
#    feature_type = FLAGS.feature_type
#    model_type = FLAGS.model_type
#    num_steps = FLAGS.num_steps

    feature_type = 'default'
    model_type = 'svm'
    # Load dataset.
    data = read_dataset('data/train_lab.txt', 'data/image_data')

    # Data Processing.
    data = preprocess_data(data, 'default')
    print("Finish preprocessing...")

    # Initialize model.
    ndim = data['image'].shape[1]
    if model_type == 'linear':
        model = LinearRegression(ndim, 'uniform')
    elif model_type == 'logistic':
        model = LogisticRegression(ndim, 'uniform')
    elif model_type == 'svm':
        model = SupportVectorMachine(ndim, 'uniform')

    # Train Model.
    print("Start to train the model...")
    model = train_model(data, model)

    # Eval Model.
    print("Start to evaluate the model...")
    data_val = read_dataset('data/val_lab.txt', 'data/image_data')
    data_val = preprocess_data(data_val, feature_type)
    loss, acc = eval_model(data_val, model)
    print(loss, acc)

    # Test Model.
    print("Start doing the test")
    data_test = read_dataset('data/test_lab.txt', 'data/image_data')
    print("Start preprocess testing data")
    data_test = preprocess_data(data_test, feature_type)
    print("Making predictions")
    data_test['label'] = model.predict(model.forward(data_test['image']))
    print("Output the results to csv file")
    write_dataset('data/test_lab.txt', data_test)
    # Generate Kaggle output.
    print("Finished!")

if __name__ == '__main__':
    tf.app.run()
