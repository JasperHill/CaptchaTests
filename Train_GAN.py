
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

AUTOTUNE          = tf.data.experimental.AUTOTUNE

## note: one of the png files from this source is improperly named; mv 3bnfnd.png 3bfnd.png
URL = 'https://www.researchgate.net/profile/Rodrigo_Wilhelmy/publication/248380891_captcha_dataset/data/00b4951ddc422dddad000000/captcha-dataset.zip'
data_dir = tf.keras.utils.get_file(fname='captcha_images.zip',origin=URL)
data_dir = pathlib.Path(data_dir)
data_dir = data_dir.parents[0]
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
    vec = np.zeros(D, dtype=np.float32)
    match = re.search(char,alphanumeric)
    vec[match.span()[0]] = 1
    return vec

def string_to_mat_and_sparse_mat(string):
    N = len(string)
    mat = np.zeros([N,D], dtype=np.float32)
    sparse_mat = np.zeros([N,D,D], dtype=np.float32)

    d = 0
    for char in string:
        mat[d] = char_to_vec(char)
        sparse_mat[d] = np.tensordot(mat[d],mat[d],axes=0)
        d += 1

    return mat,sparse_mat

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
BATCH_SIZE  = 100
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
## define custom training steps
#############################################################
STEPS_PER_EPOCH = int(np.ceil(0.8*MAX_STEPS))
VALIDATION_STEPS = int(np.ceil(0.2*MAX_STEPS))

generator_optimizer = tf.keras.optimizers.Adam(1e-3)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

generator_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
discriminator_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_avg_loss = tf.keras.metrics.Mean()
discriminator_avg_loss = tf.keras.metrics.Mean()

discriminator_acc = tf.keras.metrics.BinaryAccuracy()

generator_loss_hist     = []

discriminator_loss_hist = []
discriminator_acc_hist  = []

## labels for discriminator receiving authentic images first then synthetic images
auth_labels = np.ones([BATCH_SIZE,1])
synth_labels = np.zeros([BATCH_SIZE,1])

    
@tf.function
def GAN_train_step(ds,generator,discriminator,steps):
    for step in range(steps):
        imgs,labels,sparse_mats = next(iter(ds))

        with tf.GradientTape(persistent=False) as gen_tape, tf.GradientTape(persistent=False) as disc_tape:
            gen_imgs = generator(sparse_mats,training=True)
            print('checkpoint')
            auth_guesses = discriminator(imgs,training=True)
            synth_guesses = discriminator(gen_imgs,training=True)

            ## generator loss is the binary crossentropy between the discriminator guessing it to be authentic and
            ## the true discriminator guess
            gen_loss = generator_loss_object(auth_labels,synth_guesses)
            auth_loss = discriminator_loss_object(auth_labels,auth_guesses)
            synth_loss = discriminator_loss_object(synth_labels,synth_guesses)

            discriminator_total_loss = auth_loss + synth_loss
            
            gen_grads = gen_tape.gradient(gen_loss,generator.trainable_variables)
            disc_grads = disc_tape.gradient(discriminator_total_loss,discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gen_grads,generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(disc_grads,discriminator.trainable_variables))

        generator_avg_loss(gen_loss)
        discriminator_avg_loss(discriminator_total_loss)
            
        #calculate accuracy and loss for current epoch    
        discriminator_acc.update_state(auth_labels,auth_guesses)
        discriminator_acc.update_state(synth_labels,synth_guesses)
        discriminator_avg_loss(discriminator_total_loss)

    generator_loss_hist.append(generator_avg_loss.result())
    discriminator_loss_hist.append(discriminator_avg_loss.result())
    discriminator_acc_hist.append(discriminator_acc.result())
    

@tf.function
def GAN_test_step(ds,generator,discriminator,steps):
    for step in range(steps):
        imgs,labels,sparse_mats = next(iter(ds))
        gen_imgs = generator(sparse_mats,training=False)

        auth_guesses = discriminator(imgs,training=False)
        synth_guesses = discriminator(gen_imgs,training=False)

        gen_loss = generator_loss_object(auth_labels,synth_guesses)
        auth_loss = discriminator_loss_object(auth_labels,auth_guesses)
        synth_loss = discriminator_loss_object(synth_labels,synth_guesses)

        total_disc_loss = auth_loss + synth_loss

        generator_avg_loss(gen_loss)
        
        discriminator_avg_loss(total_disc_loss)
        discriminator_acc.update_state(auth_labels,auth_guesses)
        discriminator_acc.update_state(synth_labels,synth_guesses)

    generator_loss_hist.append(generator_avg_loss.result())
    discriminator_loss_hist.append(discriminator_avg_loss.result())
    discriminator_acc_hist.append(discriminator_acc.result())


#############################################################
## load and train the GAN; save to same directory afterward
#############################################################
    
gen_path          = "/tmp/keras_save/gen"
disc_path         = "/tmp/keras_save/disc"
generator         = tf.keras.models.load_model(gen_path)
discriminator     = tf.keras.models.load_model(disc_path)

chkpt_dir         = "./training_checkpoints"
chkpt_prefix      = os.path.join(chkpt_dir, "chkpt")
checkpoint        = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                        discriminator_optimizer=discriminator_optimizer,
                                        generator=generator,
                                        discriminator=discriminator)

checkpoint.restore(tf.train.latest_checkpoint(chkpt_dir))

for epoch in range(EPOCHS):
    if (epoch <= 10 or epoch%10 == 0):
        print('epoch: {}'.format(epoch), end=' | ')
        
        ## reset states
        generator_avg_loss.reset_states()
        
        discriminator_avg_loss.reset_states()
        discriminator_acc.reset_states()
        
        ## train and test
        GAN_train_step(train_ds,generator,discriminator,STEPS_PER_EPOCH)
        GAN_test_step(test_ds,generator,discriminator,VALIDATION_STEPS)

        ## print model metrics and losses
    if (epoch <= 10 or epoch%10 == 0):
        print('avg generator loss: {} | discriminator accuracy: {}'.format(generator_avg_loss.result(), discriminator_acc.result()))
        

checkpoint.save(file_prefix=chkpt_prefix)

img_batch,label_batch,sparse_batch = next(iter(test_ds))
gen_img_batch = generator(sparse_batch,training=False)
show_batch(gen_img_batch.numpy(),label_batch,guesses=None,filename='Generator_Results')


