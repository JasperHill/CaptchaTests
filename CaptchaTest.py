
#############################################################
##  CaptchaTest.py
##  Jan. 2020 - J. Hill
#############################################################


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

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models              import Sequential
from tensorflow.keras.layers              import Conv2D,MaxPooling2D,Dense,Flatten

AUTOTUNE = tf.data.experimental.AUTOTUNE

## note: one of the png files from this source is improperly named; 3bnfnd.png should be 3bfnd.png
URL = 'https://www.researchgate.net/profile/Rodrigo_Wilhelmy/publication/248380891_captcha_dataset/data/00b4951ddc422dddad000000/captcha-dataset.zip'
data_dir = tf.keras.utils.get_file(fname='captcha_images.zip',origin=URL)
data_dir = pathlib.Path(data_dir)
data_dir = data_dir.parents[0]
#data_dir = os.path.join(data_dir,'samples')
data_dir = pathlib.Path(data_dir)

jpg_count = len(list(data_dir.glob('samples/*.jpg')))
png_count = len(list(data_dir.glob('samples/*.png')))
NUM_OF_IMAGES = jpg_count + png_count

print('jpg_count: {}, png_count: {}'.format(jpg_count, png_count))



#############################################################
## processing and mapping functions for the files
#############################################################

IMG_HEIGHT  =  50
IMG_WIDTH   = 200


## label generation

alphanumeric = string.digits + string.ascii_lowercase
alphanum_vec = tf.io.decode_raw(alphanumeric,tf.uint8).numpy()
D = len(alphanumeric)
N = 5

def char_to_vec(char):
    vec = np.zeros(D)
    match = re.search(char,alphanumeric)
    vec[match.span()[0]] = 1
    return match.span()[0]

def string_to_mat(string):
    N = len(string)
    mat = np.zeros([N,D])
    
    d = 0
    for char in string:
        mat[d] = char_to_vec(char)
        d += 1

    return mat

def uint8_to_mat(vec): ##transforms vectors from tf.io.decode_raw to sparse matrices    
    mat = np.zeros([N,D])

    for i in vec:
        for j in range(D):
            if (i == alphanum_vec[j]):
                mat[i][j] = 1
                break
    return mat

def mat_to_string(mat): ##transforms vector outputs from the network back to strings for visualization
     string = ''
     npmat = mat.numpy()
     print('npmat is ',npmat)
     for i in range(N):
         for j in range(D):
             if (npmat[i] == alphanum_vec[j]):
                 string += alphanumeric[j]
                 break

     return string

@tf.function
def generate_labels(path):    
    parts = tf.strings.split(path,os.path.sep)
    raw_label = parts[-1]
    parts = tf.strings.split(raw_label,'.')
    string_label = parts[0] ## string_label is a tensor of the form b'true_label'

    vectorized_label = tf.io.decode_raw(string_label,tf.uint8)
    #mat_label = uint8_to_mat(vectorized_label)
    return vectorized_label

## image generation

def decode_png(png):
    png = tf.image.decode_png(png,channels=3)
    png = tf.image.convert_image_dtype(png,tf.float32)
    return tf.image.resize(png, [IMG_HEIGHT,IMG_WIDTH])

def decode_jpg(jpg):
    jpg = tf.image.decode_jpeg(jpg,channels=3)
    jpg = tf.image.convert_image_dtype(jpg,tf.float32)
    return tf.image.resize(jpg, [IMG_HEIGHT,IMG_WIDTH])

def process_path_png(path):
    mat_label = generate_labels(path)
    img = tf.io.read_file(path)
    img = decode_png(img)

    return img,mat_label

def process_path_jpg(path):
    mat_label = generate_labels(path)
    img = tf.io.read_file(path)
    img = decode_jpg(img)
    
    return img,mat_label


#############################################################
## compile dataset
#############################################################

list_png_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.png'))
list_jpg_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))
labeled_png_ds = list_png_ds.map(process_path_png,num_parallel_calls=AUTOTUNE)
labeled_jpg_ds = list_jpg_ds.map(process_path_jpg,num_parallel_calls=AUTOTUNE)
full_ds = tf.data.Dataset.concatenate(labeled_png_ds,labeled_jpg_ds)
train_ds = full_ds.shard(num_shards=2, index=0)
test_ds = full_ds.shard(num_shards=2, index=1)

EPOCHS      = 150
BATCH_SIZE  = 128
STEPS_PER_EPOCH = np.ceil(0.5*NUM_OF_IMAGES/BATCH_SIZE) ## factor of 0.5 is for the validation split

def prep_for_training(ds,cache=True,shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache,str):
            ds.cache(cache)
        else:
            ds.cache()

    ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

## auxiliary function to visualize the data
def show_batch(img_batch,label_batch):
    plt.figure(figsize=(10,10))

    for n in range(14):
        str_label = mat_to_string(label_batch[n])

        ax = plt.subplot(7,2,n+1)
        plt.imshow(img_batch[n])
        plt.title(str(str_label))
        plt.axis('off')
        
        plt.savefig('Sample_Batch.pdf')

train_ds = prep_for_training(train_ds)
img_batch,label_batch = next(iter(train_ds))
show_batch(img_batch.numpy(),label_batch)

test_ds = prep_for_training(test_ds)
## auxiliary function to quantify dataset performance
default_timeit_steps = 1000
a = tf.io.decode_raw()
def timeit(ds, steps=default_timeit_steps):
    start = time.time()
    it = iter(ds)

    for i in range(steps):
        batch = next(it)    
        if (i%10 == 0): print('.',end='')

    print()
    end = time.time()
    duration = end - start
    print('{} batches in {:.3}s'.format(steps,duration))
    print(' {:.6} images/s'.format(BATCH_SIZE*steps/duration))

timeit(train_ds)
        
#############################################################
## build and train the model
#############################################################

model = Sequential([Conv2D(10, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,3)),
                    MaxPooling2D(2,2),
                    Conv2D(20, 3, padding='same', activation='relu'),
                    MaxPooling2D(2,2),
                    Conv2D(40, 3, padding='same', activation='relu'),
                    MaxPooling2D(2,2),
                    Flatten(),
                    Dense(1000, activation='sigmoid'),
                    Dense(5, activation=None)])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds,
                    validation_data=test_ds,
                    epochs=EPOCHS,
                    verbose=1,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_steps=STEPS_PER_EPOCH
                    )


labels = model.predict(test_ds,steps=1)

print(labels)
