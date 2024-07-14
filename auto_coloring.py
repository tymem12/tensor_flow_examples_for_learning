# -*- coding: utf-8 -*-

'''Kolorująca sieć autokodująca

Do treningu sieci użyte są obrazy w skali szarości jako dane wejściowe 
oraz kolorowe jako dane wyjściowe.
Autokoder kolorujący obrazy może być traktowany jako działający 
w sposób przeciwny do odszumiającego. Zamiast usuwać zakłócenia,
kolorowanie powoduje dodanie zakłócenia (koloru) do obrazu w skali szarości.

Obrazy w skali szarości --> Kolorowanie --> Obrazy kolorowe
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
    """Konwersja obrazu w RGB do skali szarości.
       Źródło: opencv.org
       skala szarości = 0.299*czerwony + 0.587*zielony + 0.114*niebieski
    Argumenty:
        rgb (tensor): obraz rgb
    Zwraca:
        (tensor): obraz w skali szarości
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# Załadowanie danych CIFAR10
(x_train, _), (x_test, _) = cifar10.load_data()

# wprowadzenie wymiarów obrazu

# zakładamy że format obrazów jest w "channels_last"
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
channels = x_train.shape[3]

# utworzenie folderu zapisane_obrazy
imgs_dir = 'zapisane_obrazy'
save_dir = os.path.join(os.getcwd(), imgs_dir)
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

# wyświetlenie pierwszych 100 obrazów wejściowych (kolorowe i skala szarości)
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

# konwersja obrazów treningowych i testowych na skale szarości
x_train_gray = rgb2gray(x_train)
x_test_gray = rgb2gray(x_test)

# wyświetlenie obrazów testowych w wersji skali szarości
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

# normalizacja wyjściowych obrazów zbioru treningowego i testowego w kolorze
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# normalizacja wejściowych obrazów zbioru treningowego 
# i testowego w skali szarości 
x_train_gray = x_train_gray.astype('float32') / 255
x_test_gray = x_test_gray.astype('float32') / 255

# zmiana kształtu obrazów na wiersz x kolumna x kanał 
# dla wyjścia CNN/walidacji 
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)

# zmiana kształtu obrazów na wiersz x kolumna x kanał dla wejścia CNN
x_train_gray = x_train_gray.reshape(x_train_gray.shape[0], img_rows, img_cols, 1)
x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], img_rows, img_cols, 1)

# parametry sieci
input_shape = (img_rows, img_cols, 1)
batch_size = 32
kernel_size = 3
latent_dim = 256
# liczba warstw CNN i filtrów na warstwę kodera i dekodera
layer_filters = [64, 128, 256]

# budowanie modelu autokodera
# jako pierwszy budujemy koder (ang. encoder)
inputs = Input(shape=input_shape, name='koder_wejście')
x = inputs
# stos warstw Conv2D(32)-Conv2D(64)
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)

# W celu zbudowania sieci autokodującej potrzebne są informacje 
# o kształcie (ang. shape) danych. By nie wykonywać obliczeń ręcznie
# najpierw wejście dekodera Conv2DTranspose będzie miało kształt
# (4, 4, 256), zostanie przekształcony z powrotem przez dekoder do
# (32, 32, 3)
shape = K.int_shape(x)

# generowanie wektora niejawnego
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# tworzenie instancji modelu kodera
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# budowanie modelu dekodera
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# stos warstwConv2DTranspose(256)-Conv2DTranspose(128)-Conv2DTranspose(64)
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

# utworzenie instancji modelu dekodera
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# sieć autokodująca = enkoder+dekoder
# utworzenie instancji sieci autokodującej
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

# przygotowanie katalogu do zapisu modelu
save_dir = os.path.join(os.getcwd(), 'zapisane_modele')
model_name = 'colorized_ae_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# redukcja współczynnika uczenia o sqrt(0.1) 
# jeśli funkcja straty nie ulegnie poprawie w ciągu 3 epok
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               verbose=1,
                               min_lr=0.5e-6)

# zapisanie wag "na przyszłość" (np. zmiana parametrów
# bez konieczności ponownego trenowania sieci)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

# funkcja straty jako błąd średniokwadratowy (ang. Mean Square Error),
# optymalizator Adam
autoencoder.compile(loss='mse', optimizer='adam')

# wywołanie każdej epoki
callbacks = [lr_reducer, checkpoint]

# trenowanie sieci autokodującej
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
