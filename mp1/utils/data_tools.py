"""
Implements feature extraction and other data processing helpers.
"""

import numpy as np
import skimage
from skimage import filters
from numpy import fft


def preprocess_data(data, process_method='default'):
    """
    Args:
        data(dict): Python dict loaded using io_tools.
        process_method(str): processing methods needs to support
          ['raw', 'default'].
        if process_method is 'raw'
          1. Convert the images to range of [0, 1]
          2. Remove mean.
          3. Flatten images, data['image'] is converted to dimension (N, 28*28)
        if process_method is 'default':
          1. Convert images to range [0,1]
          2. Apply laplacian filter with window size of 11x11. (use skimage)
          3. Remove mean.
          3. Flatten images, data['image'] is converted to dimension (N, 28*28)

    Returns:
        data(dict): Apply the described processing based on the process_method
        str to data['image'], then return data.
    """
    if process_method == 'default':
        data['image'] = data['image'] / 255 
        #apply laplacian filter with window size of 11
        data['image'] = filters.laplace(data['image'], 11)
        #remove mean
        data = remove_data_mean(data)
        #flatten images
        data['image'] = data['image'].reshape((data['image'].shape[0],+\
            data['image'].shape[1]*data['image'].shape[2]))
        
    elif process_method == 'raw':
        #convert to range[0,1]
        data['image'] = data['image'] / 255   
        #remove mean
        data = remove_data_mean(data)
        #flatten images
        data['image'] = data['image'].reshape((data['image'].shape[0],+\
            data['image'].shape[1]*data['image'].shape[2])) 
        
    elif process_method == 'custom':
        pass

    return data


def compute_image_mean(data):
    """ Computes mean image.
    Args:
        data(dict): Python dict loaded using io_tools.
    Returns:
        image_mean(numpy.ndarray): Avaerage across the example dimension.
    """
    image_mean = data['image'].mean(axis=0)
    return image_mean


def remove_data_mean(data):
    """
    Args:
        data(dict): Python dict loaded using io_tools.
    Returns:
        data(dict): Remove mean from data['image'] and return data.
    """
    image_mean = compute_image_mean(data)
    data['image'] -= image_mean
    return data
