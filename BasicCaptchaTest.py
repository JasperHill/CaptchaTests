
#############################################################
##  BasicCaptchaTest.py
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

from tensorflow.keras.models              import Sequential
from tensorflow.keras.layers              import Conv2D,MaxPooling2D,Dense,Flatten,Reshape,Dropout

AUTOTUNE = tf.data.experimental.AUTOTUNE

## note: one of the png files from this source is improperly named; mv 3bnfnd.png 3bfnd.png
URL = 'https://www.researchgate.net/profile/Rodrigo_Wilhelmy/publication/248380891_captcha_dataset/data/00b4951ddc422dddad000000/captcha-dataset.zip'
data_dir = tf.keras.utils.get_file(fname='captcha_images.zip',origin=URL)

data_dir      =                    pathlib.Path(data_dir)
data_dir      =                       data_dir.parents[0]
data_dir      =                    pathlib.Path(data_dir)

jpg_count     = len(list(data_dir.glob('samples/*.jpg')))
png_count     = len(list(data_dir.glob('samples/*.png')))
NUM_OF_IMAGES =                     jpg_count + png_count

print('jpg_count: {}, png_count: {}'.format(jpg_count, png_count))



#############################################################
## processing and mapping functions for the files
#############################################################

IMG_HEIGHT  =  50
IMG_WIDTH   = 200
CUBE        = False

## label generation and vectorization ##

alphanumeric = string.digits + string.ascii_lowercase
D = len(alphanumeric)
N = 5 ## number of characters in each captcha title

def char_to_vec(char):
    vec = np.zeros(D)
    match = re.search(char,alphanumeric)
    vec[match.span()[0]] = 1
    return vec

def string_to_mat(string):
    N = len(string)
    mat = np.zeros([N,D])
    
    d = 0
    for char in string:
        mat[d] = char_to_vec(char)
        d += 1

    return mat

def NN_mat_to_string(nnmat): ##transforms network outputs to strings
    string = ''

    for i in range(N):
        idx = tf.argmax(nnmat[i])
        string += alphanumeric[idx]

    return string

def mat_to_string(mat): ##transforms matrices from the dataset back to strings for visualization
     string = ''
     npmat = mat.numpy()

     for i in range(N):
         for j in range(D):
             if (npmat[i][j] == 1):
                 string += alphanumeric[j]
                 break

     return string

def generate_labels(filenames):
    mat_labels = []
    for file in filenames:
        parts = re.split('\.',file)
        string_label = parts[0]
        mat_label = string_to_mat(string_label)        
        mat_labels.append(mat_label)
    return mat_labels

## image generation ##

def decode_png(png):
    png = tf.image.decode_png(png,channels=3)
    png = tf.image.convert_image_dtype(png,tf.float32)
    return tf.image.resize(png, [IMG_HEIGHT,IMG_WIDTH])

def decode_jpg(jpg):
    jpg = tf.image.decode_jpeg(jpg,channels=3)
    jpg = tf.image.convert_image_dtype(jpg,tf.float32)
    return tf.image.resize(jpg, [IMG_HEIGHT,IMG_WIDTH])


def img_cubed(img): ## auxiliary function to cube the image matrices if CUBE=True
    img_cubed = []

    for channel in range(3):
        A = img[...,channel]
        prod = tf.linalg.matmul(A,A,transpose_b=True)
        prod = tf.linalg.matmul(prod,A,transpose_b=False)
        img_cubed.append(prod // (0.4*IMG_HEIGHT*IMG_WIDTH))

    img_cubed = tf.stack(img_cubed,axis=-1)
    return img_cubed

def process_path_png(path):
    img = tf.io.read_file(path)
    img = decode_png(img)

    if CUBE: img = img_cubed(img)

    return img

def process_path_jpg(path):
    img = tf.io.read_file(path)
    img = decode_jpg(img)
    
    if CUBE: img = img_cubed(img)

    return img


#############################################################
## compile dataset
#############################################################
## shuffling is disabled to ensure proper zipping with the labels, which are listed in alphanumeric order

list_png_ds       =   tf.data.Dataset.list_files(str(data_dir/'*/*.png'),shuffle=False)
list_jpg_ds       =   tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'),shuffle=False)
png_ds            =       list_png_ds.map(process_path_png,num_parallel_calls=AUTOTUNE)
jpg_ds            =       list_jpg_ds.map(process_path_jpg,num_parallel_calls=AUTOTUNE)
img_ds            =                          tf.data.Dataset.concatenate(png_ds,jpg_ds)

filenames         =                                      os.listdir(data_dir/'samples')
filenames.sort()
mat_labels        =                                          generate_labels(filenames)
mat_ds            =                      tf.data.Dataset.from_tensor_slices(mat_labels)

full_ds           =                                tf.data.Dataset.zip((img_ds,mat_ds))

## split the dataset into a training set (80%) and a testing set (20%)

train_ds0  =             full_ds.shard(num_shards=5, index=0)
train_ds1  =             full_ds.shard(num_shards=5, index=1)
train_ds01 = tf.data.Dataset.concatenate(train_ds0,train_ds1)

train_ds2  =             full_ds.shard(num_shards=5, index=2)
train_ds3  =             full_ds.shard(num_shards=5, index=3)
train_ds23 = tf.data.Dataset.concatenate(train_ds2,train_ds3)

train_ds = tf.data.Dataset.concatenate(train_ds01,train_ds23)
test_ds  =               full_ds.shard(num_shards=5, index=4)

EPOCHS      = 30
BATCH_SIZE  = 10
MAX_STEPS   = np.ceil(NUM_OF_IMAGES/BATCH_SIZE)

def prep_for_training(ds,cache=True,shuffle_buffer_size=10000):
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
def show_batch(img_batch,label_batch,guesses):
    plt.figure(figsize=(10,10))

    for n in range(10):
        str_label = mat_to_string(label_batch[n])
        if guesses is not None: guess = NN_mat_to_string(guesses[n])

        ax = plt.subplot(5,2,n+1)
        plt.imshow(img_batch[n])

        if guesses is not None: plt.title('guess: {}'.format(guess))
        else:                   plt.title(str(str_label))

        plt.axis('off')

        if guesses is not None: plt.savefig('Test_Results.pdf')
        else:                   plt.savefig('Sample_Batch.pdf')


train_ds = prep_for_training(train_ds)
test_ds = prep_for_training(test_ds)

img_batch,label_batch = next(iter(train_ds))
show_batch(img_batch.numpy(),label_batch,guesses=None)

## auxiliary function to quantify dataset performance
default_timeit_steps = 1000

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

timeit(full_ds)

        
#############################################################
## build and train the model
#############################################################

STEPS_PER_EPOCH = np.ceil(0.8*MAX_STEPS)
VALIDATION_STEPS = np.ceil(0.2*MAX_STEPS)



w = 0.01
l2_reg = tf.keras.regularizers.l2(l=w)
model = Sequential([Conv2D(10, kernel_size=2, padding='same', activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,3)),
                    MaxPooling2D(2,2),
                    Dropout(0.1),
                    Conv2D(20, kernel_size=2, padding='same', activation='relu'),
                    MaxPooling2D(2,2),
                    Flatten(),
                    Dropout(0.2),
                    Dense(10, activation=None, kernel_regularizer=l2_reg),
                    Dense(180, activation='softmax'),
                    Reshape((5,36),input_shape=(180,)),
                    ])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_ds,
                    validation_data=test_ds,
                    epochs=EPOCHS,
                    verbose=1,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_steps=VALIDATION_STEPS
                    )


acc = history.history['accuracy']
loss = history.history['loss']

val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

plt.figure(figsize=(10,10))

##
plt.subplot(1,2,1)

plt.plot(range(EPOCHS), acc, label='accuracy')
plt.plot(range(EPOCHS), val_acc, label='val_accuracy')

plt.legend(loc='lower right')
plt.title('Training and Validation Accuracies')

##
plt.subplot(1,2,2)

plt.plot(range(EPOCHS), loss, label='loss')
plt.plot(range(EPOCHS), val_loss, label='val_loss')

plt.legend(loc='upper right')
plt.title('Training and Validation Losses')

plt.savefig('History.pdf')

img, true_label = next(iter(test_ds))
guesses = model.predict(img,steps=1)
    
show_batch(img, true_label, guesses=guesses)    

