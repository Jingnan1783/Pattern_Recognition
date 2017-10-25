################################################################################
# Author: Safa Messaoud                                                        #
# E-Mail: messaou2@illinois.edu                                                #
# Instituation: University of Illinois at Urbana-Champaign                     #
# Course: ECE 544_na Fall 2017                                                 #
# Date: July 2017                                                              #
#                                                                              #
# Description: the denoising convolutional autoencoder model                   #
#                                                                              #
#                                                                              #
################################################################################

import tensorflow as tf
import numpy as np
import utils


class DAE(object):
    """
    Denoising Convolutional Autoencoder
    """

    def __init__(self, config):
        """
        Basic setup.
        Args:
            config: Object containing configuration parameters.
        """

        # Model configuration.
        self.config = config    	

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.original_images = None

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.noisy_images = None

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.reconstructed_images = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # Global step Tensor.
        self.global_step = None

        # A boolean indicating whether the current mode is 'training'.
        self.phase_train = True



    
  
    def build_inputs(self):
        """ Input Placeholders.
        define place holders for feeding (1) noise-free images, (2) noisy images and (3) a boolean variable 
        indicating whether you are in the training or testing phase
        Outputs:
            self.original_images
            self.noisy_images
            self.phase_train
        """
        
        
        self.original_images = tf.placeholder(tf.float32, [None, self.config.n_input], 'images')
        self.noisy_images = tf.placeholder(tf.float32, [None, self.config.n_input], 'noisy_images')
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')



    def build_model(self):
        """Builds the model.
        # implements the denoising auto-encoder. Feel free to experiment with different architectures.
        Explore the effect of 1) deep networks (i.e., more layers), 2) interlayer batch normalization and
        3) dropout, 4) pooling layers, 5) convolution layers, 6) upsampling methods (upsampling vs deconvolution), 
        7) different optimization methods (e.g., stochastic gradient descent versus stochastic gradient descent
        with momentum versus RMSprop.  
        Do not forget to scale the final output between 0 and 1. 
        Inputs:
            self.noisy_images
            self.original_images
        Outputs:
            self.total_loss
            self.reconstructed_images 
        """  
        noisy_images = tf.reshape(self.noisy_images, [-1, self.config.image_height, self.config.image_width, 1])
        ### Encoder
        conv1 = tf.layers.conv2d(inputs=noisy_images, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        # Now 28x28x32
        maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
        # Batch Normalization
        batchnorm1 = utils.batch_norm(maxpool1, 32, self.phase_train)
        # Now 14x14x32
        conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        # Now 14x14x32
        maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
        # Batch Normalization
        batchnorm2 = utils.batch_norm(maxpool2, 32, self.phase_train)
        # Now 7x7x32

        ### Decoder
        conv4 = tf.layers.conv2d(inputs=batchnorm2, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        # Now 7x7x32
        upsample2 = tf.image.resize_images(conv4, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Batch Normalization
        batchnorm3 = utils.batch_norm(upsample2, 32, self.phase_train)
        # Now 14x14x32
        conv5 = tf.layers.conv2d(inputs=batchnorm3, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        # Now 14x14x32
        upsample3 = tf.image.resize_images(conv5, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Batch Normalization
        batchnorm4 = utils.batch_norm(upsample3, 32, self.phase_train)
        # Now 28x28x32
        conv6 = tf.layers.conv2d(inputs=batchnorm4, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        # Now 28x28x32
        logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3), padding='same', activation=None)
        #Now 28x28x1

        # Pass logits through sigmoid to get reconstructed image
        x_reconstructed = tf.nn.sigmoid(logits)
        x_reconstructed = tf.reshape(x_reconstructed, [-1, self.config.n_input])

        #self.reconstructed_images = tf.placeholder(tf.float32, [None, self.config.image_height, self.config.image_width, 1])
        self.reconstructed_images = x_reconstructed
        
        # Compute losses.
        self.total_loss = tf.sqrt(tf.reduce_mean(tf.square(x_reconstructed - self.original_images)))


       

    def setup_global_step(self):
	    """Sets up the global step Tensor."""
	    global_step = tf.Variable(
	    	initial_value=0,
	        name="global_step",
	        trainable=False,
	        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

	    self.global_step = global_step

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
        self.build_model()
        self.setup_global_step()


