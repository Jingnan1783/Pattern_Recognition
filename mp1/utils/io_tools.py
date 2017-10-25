"""Input and output helpers to load in data.
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.
    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.
    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,28,28)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """
    f = open(data_txt_file, 'r')
    image = []
    label = []
    for line in f:
        line.strip('\n')
        items = line.split('\t')
        image.append(image_data_path + '/' + items[0])
        label.append(int(items[1]))
    
    ic = io.imread_collection(image, True)
    images = io.concatenate_images(ic)
    data = {'image': images,
            'label': np.asarray(label)}
    
    return data



def write_dataset(data_txt_file, data):
    """Write python dictionary data into csv format for kaggle.
    Args:
        data_txt_file(str): path to the data txt file.
        data(dict): A Python dictionary with keys 'image' and 'label',
          (see descriptions above).
    """
    input_file = open(data_txt_file, 'r')
    with open('csvfile.csv','w') as file:
        file.write('Id,Prediction\n')
        i = 0
        for line in input_file:
            temp = line.split('\t')
            temp[1] = str(data['label'][i])
            file.write((','.join(temp) + '\n'))
            i += 1
    return
#data = read_dataset('../data/test_lab.txt', '../data/image_data/')
#write_dataset('../data/test_lab.txt', data)