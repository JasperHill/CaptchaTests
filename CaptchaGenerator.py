
#############################################################
##  CaptchaGenerator.py
##  Jan. 2020 - J. Hill
#############################################################

"""
CaptchaGenerator.py is an adversarial generative neural network inspired by the work of
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
from tensorflow.keras.layers              import Conv2D,Conv2DTranspose,MaxPooling2D,Dense,Flatten,Reshape,Dropout,Permute,Cropping2D

AUTOTUNE = tf.data.experimental.AUTOTUNE

## note: one of the png files from this source is improperly named; mv 3bnfnd.png 3bfnd.png
URL = 'https://www.researchgate.net/profile/Rodrigo_Wilhelmy/publication/248380891_captcha_dataset/data/00b4951ddc422dddad000000/captcha-dataset.zip'
data_dir = tf.keras.utils.get_file(fname='captcha_images.zip',origin=URL)
data_dir = pathlib.Path(data_dir)
data_dir = data_dir.parents[0]
data_dir = pathlib.Path(data_dir)

tf.print(data_dir)
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
            if (npmat[i][j] == 1): sparse[0][j][j] += 1

    return sparse

def NN_mat_to_string(nnmat):
    string = ''

    for i in range(N):
        idx = tf.argmax(nnmat[i])
        string += alphanumeric[idx]

    return string

## transform matrices from the dataset back to strings for visualization
def mat_to_string(mat):
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
    img = tf.io.read_file(path)
    img = decode_png(img)

    return img

def process_path_jpg(path):
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

EPOCHS      = 2
BATCH_SIZE  = 10
MAX_STEPS = np.ceil(NUM_OF_IMAGES/BATCH_SIZE)

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
def show_batch(img_batch,label_batch,guesses,filename):
    plt.figure(figsize=(10,10))

    for n in range(10):
        str_label = mat_to_string(label_batch[n])
        if guesses is not None: guess = NN_mat_to_string(guesses[n])

        ax = plt.subplot(5,2,n+1)
        plt.imshow(img_batch[n])

        if guesses is not None: plt.title('guess: {}'.format(guess))
        else:                   plt.title(str(str_label))

        plt.axis('off')
        plt.savefig(filename+'.pdf')

train_ds = prep_for_training(train_ds)
test_ds = prep_for_training(test_ds)

img_batch,label_batch,sparse_batch = next(iter(train_ds))
show_batch(img_batch.numpy(),label_batch,guesses=None,filename='Sample_Batch')

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
            input_shape != self.shape):
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
STEPS_PER_EPOCH = int(np.ceil(0.8*MAX_STEPS))
VALIDATION_STEPS = int(np.ceil(0.2*MAX_STEPS))

generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()

generator_loss_object = tf.keras.losses.MeanSquaredError()
discriminator_loss_object = tf.keras.losses.BinaryCrossentropy()

generator_avg_loss = tf.keras.metrics.Mean()
discriminator_avg_loss = tf.keras.metrics.Mean()

generator_MSE = tf.keras.metrics.MeanSquaredError()
discriminator_acc = tf.keras.metrics.BinaryAccuracy()

generator_loss_hist     = []
generator_MSE_hist      = []

discriminator_loss_hist = []
discriminator_acc_hist  = []

## labels for discriminator receiving authentic images first then synthetic images
auth_labels = np.ones([BATCH_SIZE,1])
synth_labels = np.zeros([BATCH_SIZE,1])

    
@tf.function
def GAN_train_step(ds,generator,discriminator,steps):
    for step in range(steps):
        imgs,labels,sparse_mats = next(iter(ds))
        gen_imgs = generator(sparse_mats,training=False)

        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
            gen_imgs = generator(sparse_mats,training=True)            
            auth_guesses = discriminator(imgs,training=True)
            synth_guesses = discriminator(gen_imgs,training=True)

            gen_loss = generator_loss_object(imgs,gen_imgs)
            auth_loss = discriminator_loss_object(auth_labels,auth_guesses)
            synth_loss = discriminator_loss_object(synth_labels,synth_guesses)

            discriminator_total_loss = auth_loss + synth_loss
            
            gen_grads = gen_tape.gradient(gen_loss,generator.trainable_variables)
            disc_grads = disc_tape.gradient(discriminator_total_loss,discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gen_grads,generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(disc_grads,discriminator.trainable_variables))

        generator_avg_loss(gen_loss)
        generator_MSE(imgs,gen_imgs)
        discriminator_avg_loss(auth_loss+synth_loss)
            
        #calculate accuracy and loss for current epoch    
        discriminator_acc.update_state(auth_labels,auth_guesses)
        discriminator_acc.update_state(synth_labels,synth_guesses)
        discriminator_avg_loss(discriminator_total_loss)

    generator_loss_hist.append(generator_avg_loss.result())
    generator_MSE_hist.append(generator_MSE.result())
    discriminator_loss_hist.append(discriminator_avg_loss.result())
    discriminator_acc_hist.append(discriminator_acc.result())
    print('## training epoch complete ##')
    

@tf.function
def GAN_test_step(ds,generator,discriminator,steps):
    for step in range(steps):
        imgs,labels,sparse_mats = next(iter(ds))
        gen_imgs = generator(sparse_mats,training=False)

        auth_guesses = discriminator(imgs,training=False)
        synth_guesses = discriminator(gen_imgs,training=False)

        gen_loss = generator_loss_object(imgs,gen_imgs)
        auth_loss = discriminator_loss_object(auth_labels,auth_guesses)
        synth_loss = discriminator_loss_object(synth_labels,synth_guesses)

        total_disc_loss = auth_loss + synth_loss

        generator_avg_loss(gen_loss)
        generator_MSE(imgs,gen_imgs)
        
        discriminator_avg_loss(total_disc_loss)
        discriminator_acc.update_state(auth_labels,auth_guesses)
        discriminator_acc.update_state(synth_labels,synth_guesses)

    generator_loss_hist.append(generator_avg_loss.result())
    generator_MSE_hist.append(generator_MSE.result())
    discriminator_loss_hist.append(discriminator_avg_loss.result())
    discriminator_acc_hist.append(discriminator_acc.result())
    print('## testing epoch complete ##')

    
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
                            Dense(1, activation='softmax')
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
    discriminator_acc.reset_states()

    ## train and test
    print('beginning training step')
    GAN_train_step(train_ds,generator,discriminator,STEPS_PER_EPOCH)

    print('beginning testing step')
    GAN_test_step(test_ds,generator,discriminator,VALIDATION_STEPS)

    
img_batch,label_batch,sparse_batch = next(iter(test_ds))
gen_img_batch = generator(sparse_batch,training=False)
show_batch(gen_img_batch.numpy(),label_batch,guesses=None,filename='Generator_Results')


