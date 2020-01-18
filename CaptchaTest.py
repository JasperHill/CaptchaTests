from __future__ import absolute_import, division, print_function, unicode_literals

import string
import re
import os
import numpy             as np
import tensorflow        as tf
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

cwd = os.getcwd()
captcha_dir = os.path.join(cwd,'captcha_samples')

train_dir   = os.path.join(captcha_dir,'training_samples')
test_dir    = os.path.join(captcha_dir,'testing_samples')

train_size  = len(os.listdir(train_dir))
test_size   = len(os.listdir(test_dir))

#############################################################
## create labels from file names
#############################################################
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)
train_labels = []
test_labels = []


for file in train_files:
    if re.search('png',file) is not None:
        train_labels.append(file.replace('.png',''))

    elif re.search('jpg',file) is not None:
        train_labels.append(file.replace('.jpg',''))

for file in test_files:
    if re.search('png',file) is not None:
        test_labels.append(file.replace('.png',''))

    elif re.search('jpg',file) is not None:
        test_labels.append(file.replace('.jpg',''))

#############################################################
## map the captcha strings to vectors
#############################################################
alphanumeric = string.digits + string.ascii_lowercase
def map_char(char):
    match = re.search(char,alphanumeric)
    return match.span()[0]

def vectorize(string):
    D = len(string)
    v = np.zeros([D])
    
    d = 0
    for char in string:
        n = map_char(char)
        v[d] = n
        d++

return v/D

for label in train_labels:
    vectorize(label)

for label in test_labels:
    vectorize(label)

#############################################################
## preprocess captcha images
#############################################################

IMG_HEIGHT  =  50
IMG_WIDTH   = 200

EPOCHS      =  15
BATCH_SIZE  = 128

train_image_generator = ImageDataGenerator(rescale=1/255)
test_image_generator = ImageDataGenerator(rescale=1/255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           director=train_dir,
                                                           shuffle=False,
                                                           target_size=(IMG_HEIGHT,IMG_WIDTH)
                                                           )

test_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           director=test_dir,
                                                           shuffle=False,
                                                           target_size=(IMG_HEIGHT,IMG_WIDTH)
                                                           )
