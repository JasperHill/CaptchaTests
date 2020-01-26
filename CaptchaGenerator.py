
#############################################################
##  CaptchaGenerator.py
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
from tensorflow.keras.layers              import Conv2D,Conv2DTranspose,MaxPooling2D,Dense,Flatten,Reshape,Dropout,Permute,Cropping2D

AUTOTUNE = tf.data.experimental.AUTOTUNE

## note: one of the png files from this source is improperly named; mv 3bnfnd.png 3bfnd.png
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


## label generation and vectorization ##

alphanumeric = string.digits + string.ascii_lowercase
D = len(alphanumeric)
N = 5 ## number of characters in each captcha title

def char_to_vec(char):
    vec = np.zeros(D)
    match = re.search(char,alphanumeric)
    vec[match.span()[0]] = 1
    return vec

def string_to_mat_and_sparse_mat(string):
    N = len(string)
    mat = np.zeros([N,D])
    sparse_mat = np.zeros([1,D,D])

    d = 0
    for char in string:
        mat[d] = char_to_vec(char)
        sparse_mat += np.tensordot(mat[d],mat[d],axes=0)
        d += 1

    return mat,sparse_mat

def mat_to_sparse(mat):
    npmat = mat
    print(mat[0][0])
    dim = mat.shape[-1]
    sparse = np.zeros([1,dim,dim])

    for i in range(N):
        for j in range(dim):
            if (npmat[i][j] == 1): sparse[0][j][j] = 1

    return sparse

def NN_mat_to_string(nnmat):
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
    sparse_labels = []
    for file in filenames:
        parts = re.split('\.',file)
        string_label = parts[0]

        mat_label,sparse_label = string_to_mat_and_sparse_mat(string_label)        
        mat_labels.append(mat_label)
        sparse_labels.append(sparse_label)

    return mat_labels,sparse_labels

## image generation ##

def decode_png(png):
    png = tf.image.decode_png(png,channels=3)
    png = tf.image.convert_image_dtype(png,tf.float32)
    return tf.image.resize(png, [IMG_HEIGHT,IMG_WIDTH])

def decode_jpg(jpg):
    jpg = tf.image.decode_jpeg(jpg,channels=3)
    jpg = tf.image.convert_image_dtype(jpg,tf.float32)
    return tf.image.resize(jpg, [IMG_HEIGHT,IMG_WIDTH])

def process_path_png(path):
    #mat_label = generate_labels(path)
    img = tf.io.read_file(path)
    img = decode_png(img)

    return img

def process_path_jpg(path):
    #mat_label = generate_labels(path)
    img = tf.io.read_file(path)
    img = decode_jpg(img)
    
    return img


#############################################################
## compile dataset
#############################################################
## shuffling is disabled to ensure proper zipping with the labels, which are listed in alphanumeric order

## import and preprocess images
list_png_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.png'),shuffle=False)
list_jpg_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'),shuffle=False)
png_ds = list_png_ds.map(process_path_png,num_parallel_calls=AUTOTUNE)
jpg_ds = list_jpg_ds.map(process_path_jpg,num_parallel_calls=AUTOTUNE)
img_ds = tf.data.Dataset.concatenate(png_ds,jpg_ds)

## generate label matrices for the solver and sparse matrices for the generator
filenames = os.listdir(data_dir/'samples')
filenames.sort()
mat_labels,sparse_labels = generate_labels(filenames)
mat_ds = tf.data.Dataset.from_tensor_slices(mat_labels)
sparse_ds = tf.data.Dataset.from_tensor_slices(sparse_labels)
#sparse_mat_ds = mat_ds.map(mat_to_sparse,num_parallel_calls=AUTOTUNE)

full_ds = tf.data.Dataset.zip((img_ds,mat_ds,sparse_ds))

## split the dataset into fifths ## 80% is for training; 20% for testing
train_ds0 = full_ds.shard(num_shards=5, index=0)
train_ds1 = full_ds.shard(num_shards=5, index=1)
train_ds01 = tf.data.Dataset.concatenate(train_ds0,train_ds1)

train_ds2 = full_ds.shard(num_shards=5, index=2)
train_ds3 = full_ds.shard(num_shards=5, index=3)
train_ds23 = tf.data.Dataset.concatenate(train_ds2,train_ds3)

train_ds = tf.data.Dataset.concatenate(train_ds01,train_ds23)
test_ds = full_ds.shard(num_shards=5, index=4)

EPOCHS      = 30
BATCH_SIZE  = 10
MAX_STEPS = np.ceil(NUM_OF_IMAGES/BATCH_SIZE) ## factor of 0.5 is for the validation split

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

img_batch,label_batch,sparse_batch = next(iter(train_ds))
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
## create a pseudo (non-unitary) basis rotation layer
#############################################################

class BasisRotation(tf.keras.layers.Layer):
    def __init__(self, shape):
        super(BasisRotation,self).__init__()
        self.shape = shape

    def build(self, input_shape):
        if (input_shape[0] is not None and 
            Noneinput_shape != self.shape):
            print('Error: input shape {} must be identical to the kernel shape {}'.format(input_shape,self.shape))

        self.kernel = self.add_weight("kernel", shape=self.shape)

    def call(self,input):
        temp = tf.linalg.matmul(input,self.kernel,transpose_a=False,transpose_b=False)
        return tf.linalg.matmul(self.kernel,temp,transpose_a=True,transpose_b=False)
        
## a layer that acts via pseudo projection operators P and PT
## on inputs, mapping them from some MxN space to num_channels M'xN' spaces

class Projection(tf.keras.layers.Layer):
    def __init__(self,target_shape,num_channels):
        super(Projection,self).__init__()
        self.target_shape = target_shape
        self.num_channels = num_channels

    def build(self, input_shape):
        self.P = self.add_weight("kernel", shape=[self.num_channels,self.target_shape[-2],input_shape[-1]])
        self.PT = self.add_weight("kernel", shape=[self.num_channels,input_shape[-2],self.target_shape[-1]])

    def call(self,input):
        temp = tf.linalg.matmul(input,self.PT,transpose_a=False,transpose_b=False)
        return tf.linalg.matmul(self.P,temp,transpose_a=False,transpose_b=False)

#############################################################
## define custom training steps
#############################################################
STEPS_PER_EPOCH = np.ceil(0.8*MAX_STEPS)
VALIDATION_STEPS = np.ceil(0.2*MAX_STEPS)

optimizer = tf.keras.optimizers.Adam()
generator_loss_object = tf.keras.losses.MeanSquaredError()
discriminator_loss_object = tf.keras.losses.BinaryCrossentropy()

generator_avg_loss = tf.keras.metrics.Mean()
discriminator_avg_loss = tf.keras.metrics.Mean()

generator_MSE = tf.keras.metrics.MeanSquaredError()
discriminator_acc = tf.keras.metrics.BinaryAccuracy()
discriminator_avg_acc = tf.keras.metrics.Mean()

generator_loss_hist     = []
generator_MSE_hist      = []

discriminator_loss_hist = []
discriminator_acc_hist  = []

## labels for discriminator receiving authentic images first then synthetic images
true_labels = [tf.constant([1]),tf.constant([0])]

@tf.function
def generator_train_step(ds,generator):
    print('gen_train_step')
    for imgs,labels,sparse_mats in ds:
        with tf.GradientTape() as tape:
            gen_imgs = generator(sparse_mats,training=True)
            loss = generator_loss_object(imgs,gen_imgs)

            grads = tape.gradient(loss,generator.trainable_variables)
            optimizer.apply_gradients(zip(grads,generator.trainable_variables))

        generator_avg_loss(loss)
        generator_MSE(imgs,gen_imgs)
    generator_loss_hist.append(generator_avg_loss.result())
    generator_MSE_hist.append(generator_MSE.result())

@tf.function
def discriminator_train_step(ds,generator,discriminator):
    print('disc_train_step')
    for imgs,labels,sparse_mats in ds:
        discriminator_acc.reset_states()

        gen_img = generator(sparse_mats,training=False)

        with tf.GradientTape() as tape:
            guesses = discriminator(imgs),discriminator(gen_imgs)
            loss = discriminator_loss_object(true_labels,guesses)

            grads = tape.gradient(loss,discriminator.trainable_variables)
            optimizer.apply_gradients(zip(grads,discriminator.trainable_variables))

        discriminator_avg_acc(discriminator_acc(true_labels,guesses))
        discriminator_avg_loss(loss)
    discriminator_loss_hist.append(discriminator_avg_loss.result())
    discriminator_acc_hist.append(discriminator_avg_acc.result())

@tf.function
def generator_test_step(ds,generator):
    for imgs,labels,sparse_mats in ds:
        gen_imgs = generator(sparse_mats,training=False)
        loss = generator_loss_object(imgs,gen_imgs)
        
        generator_avg_loss(loss)
        generator_metric(imgs,gen_imgs)
    generator_loss_hist.append(generator_avg_loss.result())
    generator_MSE_hist.append(generator_metric.result())    

@tf.function
def discriminator_test_step(ds,generator,discriminator):
    for imgs,labels,sparse_mats in ds:
        discriminator_acc.reset_states()

        gen_imgs = generator(sparse_mats,training=False)

        guesses = discriminator(imgs),discriminator(gen_imgs)
        loss = discriminator_loss_object(true_labels,guesses)

        discriminator_avg_loss(loss)
        discriminator_avg_acc(discriminator_acc(true_labels,guesses))
    discriminator_loss_hist.append(discriminator_avg_loss.result())
    discriminator_acc_hist.append(discriminator_avg_acc.result())

generator = Sequential([tf.keras.layers.InputLayer(input_shape=(1,D,D)),
                        BasisRotation(shape=(1,D,D)),
                        Projection(target_shape=(50,200),num_channels=20),
                        Permute((2,3,1)),
                        Conv2D(3, kernel_size=2, padding='same',activation='relu')
                        ])

generator.build(input_shape=(1,D,D))
print('Generator Summary:')
generator.summary()

w = 0.001
l2_reg = tf.keras.regularizers.l2(l=w)
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
                            Dense(2, activation='softmax')
                            ])

discriminator.build(input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
print('Discriminator Summary:')
discriminator.summary()

for epoch in range(EPOCHS):
    print('epoch: {}'.format(epoch))
    ## reset states
    generator_avg_loss.reset_states()
    generator_MSE.reset_states()

    discriminator_avg_loss.reset_states()
    discriminator_avg_acc.reset_states()

    ## train and test
    generator_train_step(train_ds,generator)
    discriminator_train_step(train_ds,generator,discriminator)

    generator_test_step(test_ds,generator)
    discriminator_test_step(test_ds,generator,discriminator)

    
