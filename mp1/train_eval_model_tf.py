"""
Train model and eval model helpers for tensorflow implementation.
"""
from __future__ import print_function

import numpy as np
#from models.linear_regression import LinearRegression


def train_model(data, model, learning_rate=0.005, batch_size=16,
                num_steps=100, shuffle=True):
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
    batch_epoch_num = (data['label'].shape[0] + batch_size - 1) // batch_size
#    data_test = read_dataset('data/val_lab.txt', 'data/image_data')
#    data_test = preprocess_data(data_test, 'default')
#    
    for step in range(num_steps):
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
#        if step % 10 == 0:
##            print("Evaluate the model for step:", step)
#            loss, acc = eval_model(data_test, model)
#            print("Loss and Acc:", loss, acc)
    # Perform gradient descent.
    return model


def update_step(image_batch, label_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).
    Args:
        image_batch(numpy.ndarray): input data of dimension (N, ndims).
        label_batch(numpy.ndarray): label data of dimension (N,).
        model(LinearModel): Initialized linear model.
    """
    feed_dict = {model.x_placeholder: image_batch,model.y_placeholder: label_batch,\
                 model.learning_rate_placeholder: learning_rate}
    _, l = model.session.run([model.update_op_tensor, model.loss_tensor], feed_dict=feed_dict)
    return l


def eval_model(data, model):
    """Performs evaluation on a dataset.
    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    loss = model.session.run(model.loss_tensor, feed_dict={model.x_placeholder: data['image'],
                                                           model.y_placeholder: data['label']
                                                           })
    preds = model.session.run(model.predict_tensor, feed_dict={model.x_placeholder: data['image']})
#    print("preds", preds)
#    print("data labels", data['label'])
    correct_nums = np.count_nonzero(preds == (data['label'].reshape((data['label'].shape[0], 1))), axis=0)
#    print("Correct nums:", correct_nums)
    acc = correct_nums[0] / data['label'].shape[0]
    
    return loss, acc


