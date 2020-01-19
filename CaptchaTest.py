from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import string
import re
import os
import string
import numpy             as np
import tensorflow        as tf
import tensorflow.keras
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

URL = 'https://www.researchgate.net/profile/Rodrigo_Wilhelmy/publication/248380891_captcha_dataset/data/00b4951ddc422dddad000000/captcha-dataset.zip'
data_dir = tf.keras.utils.get_file(fname='captcha_images.zip',origin=URL)
data_dir = pathlib.Path(data_dir)
data_dir = data_dir.parents[0]
#data_dir = os.path.join(data_dir,'samples')
data_dir = pathlib.Path(data_dir)

jpg_count = len(list(data_dir.glob('samples/*.jpg')))
png_count = len(list(data_dir.glob('samples/*.png')))

print('jpg_count: {}, png_count: {}'.format(jpg_count, png_count))



#############################################################
## create label, image pairs
#############################################################

IMG_WIDTH = 200
IMG_HEIGHT = 50


def generate_label(path):
    parts = tf.strings.split(path,os.path.sep)
    raw_label = parts[-1]
    parts = tf.strings.split(raw_label,'.')
    type = parts[-1]
    label = parts[0]
    print('label: {}, type: {}'.format(label,type))
    return label, type

def decode_img(img,type):
    img = tf.cond(tf.equal(tf.strings.regex_full_match(type,'png'),tf.constant(True)),
                  tf.image.decode_png(img,channels=3),
                  tf.image.decode_jpeg(img,channels=3)
                  )

    img = tf.image.convert_image_dtype(img,tf.float32)
    return tf.image.resize(img, [IMG_WIDTH,IMG_HEIGHT])

def process_path(path):
    label, type = generate_label(path)
    img = tf.io.read_file(path)
    img = decode_img(img,type)

    return label,img

##

list_ds = tf.data.Dataset.list_files([str(data_dir/'*/*.jpg'),str(data_dir/'*/*.png')])
labeled_ds = list_ds.map(process_path)

for label, raw_img in labeled_ds.take(1):
    print(repr(raw_img.numpy()[:100]))
    print('')
    print(label.numpy())

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
        d += 1

    return v/D


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
                                                           directory=data_dir,
                                                           shuffle=False,
                                                           target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                           )

test_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                          directory=data_dir,
                                                          shuffle=False,
                                                          target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                          )


