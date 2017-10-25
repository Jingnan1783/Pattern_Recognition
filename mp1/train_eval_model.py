"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression
from utils.io_tools import read_dataset
from utils.data_tools import preprocess_data
#Add by myself

def train_model(data, model, learning_rate=0.01, batch_size=64,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.
    batch_epoch_num = (data['label'].shape[0] + batch_size - 1) // batch_size
#    data_test = read_dataset('data/val_lab.txt', 'data/image_data/')
#    data_test = preprocess_data(data_test, 'default')
    # Second arg, 'default', 'raw'
    # data = d_tools.preprocess_data(data)
    # Perform training process
    for step in range(num_steps):
        # Shuffle
        if shuffle:
            shuffled_idx = np.arange(data['image'].shape[0])
            np.random.shuffle(shuffled_idx)
            data['image'] = data['image'][shuffled_idx]
            data['label'] = data['label'][shuffled_idx]
        for j in range(batch_epoch_num):
            if j != batch_epoch_num - 1:
                image_batch = data['image'][j*batch_size:(j+1)*batch_size]
                label_batch = data['label'][j*batch_size:(j+1)*batch_size]
            else:
                image_batch = data['image'][j*batch_size:]
                label_batch = data['label'][j*batch_size:]
            update_step(image_batch, label_batch, model, learning_rate)
#        if step % 100 == 0:
#            print("Evaluate the model for step:", step)
#            acc, loss = eval_model(data_test, model)
#            print(acc, loss)
    return model


def update_step(image_batch, label_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).
    Args:
        image_batch(numpy.ndarray): input data of dimension (N, ndims).
        label_batch(numpy.ndarray): label data of dimension (N,).
        model(LinearModel): Initialized linear model.
    """
    gradient = model.backward(model.forward(image_batch), label_batch)
    model.w = model.w - learning_rate * gradient

def eval_model(data, model):
    """Performs evaluation on a dataset.
    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    forward_step = model.forward(data['image'])
    predict_label = model.predict(forward_step)
    #print(data['label'].shape, predict_label.shape)
    correct_nums = np.count_nonzero(predict_label == data['label'])
    loss = model.loss(forward_step, data['label'])
    acc = correct_nums / data['label'].shape[0]
    return loss, acc
