
#############################################################
##  CreateGAN.py
##  Jan. 2020 - J. Hill
#############################################################

"""
CreateGenerator.py creates an adversarial generative neural network inspired by the work of
Ye et al., Yet Another Captcha Solver: An Adversarial Generative Neural Network Based Approach

The configuration herein is much simpler and less robust, but it follows the same method:
A relatively small sample of captcha images are presented to a network containing a generator,
attempting to reproduce such captcha images from their corresponding labels and a discriminator,
attempting to discern authentic and synthetic images. The two work toward opposing goals, and
training ceases when the discriminator is unable to correctly classify a certain fraction of the
inputs.

A solver is then trained with synthetic captcha images from the generator. Finally, the solver
is refined via training with the authentic captchas.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import time
import pathlib
import string
import re
import os
import string
import numpy             as np
import tensorflow        as tf
import tensorflow.keras
import matplotlib.pyplot as plt

from tensorflow.keras.models              import Sequential
from tensorflow.keras.layers              import Conv2D,Conv2DTranspose,MaxPooling2D,Dense,Flatten,Reshape,Dropout,Permute,InputLayer

AUTOTUNE    = tf.data.experimental.AUTOTUNE

N           =   5
D           =  36
IMG_HEIGHT  =  50
IMG_WIDTH   = 200

        
#############################################################
## create a pseudo (non-unitary) basis rotation layer
#############################################################

class BasisRotation(tf.keras.layers.Layer):
    def __init__(self, shape):
        super(BasisRotation,self).__init__()
        self.shape = shape

    def build(self, input_shape):
        if (input_shape[0] is not None and 
            input_shape != self.shape):
            print('Error: input shape {} must be identical to the kernel shape {}'.format(input_shape,self.shape))

        self.kernel = self.add_weight("kernel", shape=self.shape, initializer=tf.keras.initializers.Orthogonal)

    def call(self,input):
        temp = tf.linalg.matmul(input,self.kernel,transpose_a=False,transpose_b=False)
        return tf.linalg.matmul(self.kernel,temp,transpose_a=True,transpose_b=False)
        
## a layer that acts via pseudo projection operators P and PT
## on inputs, mapping them from some MxN space to num_channels M'xN' spaces

class Projection(tf.keras.layers.Layer):
    def __init__(self,target_shape, num_output_channels):
        super(Projection,self).__init__()
        self.target_shape = target_shape
        self.num_output_channels = num_output_channels

    def build(self, input_shape):
        self.num_input_channels = input_shape[1]
        self.P = self.add_weight("kernel0",
                                 shape=[self.num_output_channels,self.num_input_channels,self.target_shape[-2],input_shape[-2]],
                                 initializer=tf.keras.initializers.Orthogonal)
        
        self.PT = self.add_weight("kernel1",
                                  shape=[self.num_output_channels,self.num_input_channels,input_shape[-1],self.target_shape[-1]],
                                  initializer=tf.keras.initializers.Orthogonal)

    @tf.function
    def call(self,input):
        output = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        d = 0
        for mat in input:
            temp = tf.linalg.matmul(mat,self.PT,transpose_a=False,transpose_b=False)
            temp = tf.linalg.matmul(self.P,temp,transpose_a=False,transpose_b=False)

            ## fold the tensor along the input channel
            temp = tf.math.reduce_sum(temp, axis=1)
            output = output.write(d, temp)
            d += 1

        return output.stack()

        
#############################################################
## create the generator and discriminator
#############################################################
    
w = 0.001
l2_reg = tf.keras.regularizers.l2(l=w)

####################################
generator = Sequential([InputLayer(input_shape=(N,D,D)),
                        BasisRotation(shape=(N,D,D)),
                        Projection(target_shape=(50,200), num_output_channels=50),
                        Permute((2,3,1)),
                        Conv2D(20, kernel_size=2, padding='same', activation='relu'),
                        Conv2D(3, kernel_size=2, padding='same', activation='tanh')])
####################################

####################################
discriminator = Sequential([Conv2D(10, kernel_size=2, padding='same', activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,3)),
                            MaxPooling2D(2,2),
                            Conv2D(25, kernel_size=2, padding='same', activation='relu'),
                            MaxPooling2D(2,2),
                            Dropout(0.2),
                            Conv2D(50, kernel_size=2, padding='same', activation='relu'),
                            MaxPooling2D(2,2),
                            Flatten(),
                            Dense(360, activation=None, kernel_regularizer=l2_reg),
                            Dropout(0.2),
                            Dense(1, activation='tanh')])
####################################

generator.build(input_shape=(N,D,D))
print('Generator Summary:')
generator.summary()

discriminator.build(input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
print('Discriminator Summary:')
discriminator.summary()


## save the model
keras_gen_path = "/tmp/keras_save/gen"
keras_disc_path = "/tmp/keras_save/disc"
print('Saving generator to {}'.format(keras_gen_path))
generator.save(keras_gen_path)
print('Saving discriminator to {}'.format(keras_disc_path))
discriminator.save(keras_disc_path)


