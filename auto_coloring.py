# -*- coding: utf-8 -*-

'''Coloring autoencoder network 
X - images in grey scale
y - images in normal scale (colored images)

Coloring autoencoder can be explained as autoencoder that instead of the removing the noise it is added (as the color)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import os

def rgb2gray(rgb):
    """Convert image RGB to G .
       grey scale = 0.299* R+ 0.587*G + 0.114*B
    Arguments:
        rgb (tensor): image rgb
    return:
        (tensor): image G
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


(x_train, _), (x_test, _) = cifar10.load_data()

# images number_of_images x 32 x 32 x 3 
img_rows = x_train.shape[1] 
img_cols = x_train.shape[2]
channels = x_train.shape[3]

imgs_dir = 'zapisane_obrazy'
save_dir = os.path.join(os.getcwd(), imgs_dir)
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

# display first 100 input images (colored)
imgs = x_test[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Kolorowe obrazy testowe \n(bezpośrednio ze zbioru CIFAR10)')
plt.imshow(imgs, interpolation='none')
plt.savefig('%s/test_color.png' % imgs_dir)
plt.savefig('%s/test_color.tif' % imgs_dir)
plt.show()

x_train_gray = rgb2gray(x_train)
x_test_gray = rgb2gray(x_test)

# display first 100 input images (in grey scale)

imgs = x_test_gray[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Obrazy testowe w skali szarości (Wejście)')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('%s/test_gray.png' % imgs_dir)
plt.savefig('%s/test_gray.tif' % imgs_dir)
plt.show()

# normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train_gray = x_train_gray.astype('float32') / 255
x_test_gray = x_test_gray.astype('float32') / 255

# reshaping the output images to row x column x chanel (3 because it is the color) 
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
print(x_train.shape)

# reshaping the input images to row x column x 1 (1 because it is the grey) 
x_train_gray = x_train_gray.reshape(x_train_gray.shape[0], img_rows, img_cols, 1)
x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)
batch_size = 32
kernel_size = 3
latent_dim = 256 # the size of the implicit/ hidden vector 
layer_filters = [64, 128, 256] # number of filters in the layers

# building autoencoder
# 1. encoder 
inputs = Input(shape=input_shape, name='koder_wejście')
x = inputs
# stack of layers Conv2D(32)-Conv2D(64)
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)
# now the outputs is 4 x 4 x 256 (32 x 32 x 1 -> 16 x 16 x 64 -> 8 x 8 x 128 -> 4 x 4 x 256)
# we are gonna use decoder to transform it back to 32 x 32 x 3
shape = K.int_shape(x)

# generating the latent vector
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# creating the MODEL of encoder (3 conv2 layers -> flatten -> dense to 256)
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# 2. Decoder
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# stack od layers Conv2DTranspose(256)-Conv2DTranspose(128)-Conv2DTranspose(64)
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)

outputs = Conv2DTranspose(filters=channels,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# creating the instance of the decoder MODEL
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# model Autoencoder = model encoder + model decoder
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

# catalog for saving images
save_dir = os.path.join(os.getcwd(), 'zapisane_modele')
model_name = 'colorized_ae_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# reduction of the learning rate by sqrt(0.1) if the cost function wont decrease in 3 epoch 
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               verbose=1,
                               min_lr=0.5e-6)

# saving weights 
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

# loss function : MSE
# optimization:  Adam
autoencoder.compile(loss='mse', optimizer='adam')

callbacks = [lr_reducer, checkpoint]

autoencoder.fit(x_train_gray,
                x_train,
                validation_data=(x_test_gray, x_test),
                epochs=30,
                batch_size=batch_size,
                callbacks=callbacks)

# przewidywanie wyjść sieci autokodującej na danych testowych
x_decoded = autoencoder.predict(x_test_gray)

# wyświetlenie pierwszych 100 pokolorowanych obrazów
imgs = x_decoded[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Pokolorowane obrazy testowe \n(przewidywane)')
plt.imshow(imgs, interpolation='none')
plt.savefig('%s/colorized.png' % imgs_dir)
plt.savefig('%s/colorized.tif' % imgs_dir)
plt.show()
