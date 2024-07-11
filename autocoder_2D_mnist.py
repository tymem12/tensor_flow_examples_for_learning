# -*- coding: utf-8 -*-

'''
Przykład sieci autokodującej wykorzystującej zbiór MNIST 
i  dwuwymiarowy wektor niejawny

Sieć autokodująca wymusza na koderze odkrycie dwuwymiarowego 
niejawnego wektora na bazie którego dekoder będzie w stanie 
odtworzyć oryginalne dane. Dwuwymiarowy wektor niejawny jest 
rzutowany na na przestrzeń dwuwymiarową, co umożliwia analizę
rozkładu kodu w przestrzeni niejawnej. Można nawigować w przestrzeni
niejawnej zmieniając wartości wektora niejawnego by otrzymać nowe
cyfry MNIST.

Ta sieć autokodująca ma budowę modułową. Koder, dekoder 
i sieć autokodująca są trzema modelami współdzielącymi wagi.
Na przykład, po trenowaniu sieci autokodującej koder może być 
użyty to generowania wektora niejawnego dla danych wejściowych
w celu użycia ich do przygotowania wizualizacji o 
obniżonej wymiarowości podobnie do PCA lub TSNE.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_results(models,
                 data,
                 batch_size=32,
                 model_name="autoencoder_2dim"):
    """	Wykreśla dwuwymiarowe wartości niejawne w postaci wykresu rozrzutu
		a następnie ryzuje cyfry MNIST w funkcji dwuwymiarowego 
		wektora niejawnego.
	
	Argumenty:
		models(list): modele kodera i dekodera
		data(list): dane testowe i etykiety
		batch_size(int):rozmiar wsadowy predykcji
		model_name(string): który model używa tej funkcji
    """

    encoder, decoder = models
    x_test, y_test = data
    xmin = ymin = -4
    xmax = ymax = +4
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "latent_2dim.png")
	# wyświetla wykres 2D klas cyfr w przestrzeni niejawnej
    z = encoder.predict(x_test,
                        batch_size=batch_size)
    plt.figure(figsize=(12, 10))

	# zakresy osi x i y
    axes = plt.gca()
    axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])

	# podpróbkowanie by zredukować gęstość punktów na wykresie
    z = z[0::2]
    y_test = y_test[0::2]
    plt.scatter(z[:, 0], z[:, 1], marker="")
    for i, digit in enumerate(y_test):
        axes.annotate(digit, (z[i, 0], z[i, 1]))
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
	# wyświetla dwuwymiarową hiperpowierzchnię cyfr o rozmiarze 30x30
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # współrzędne związane z wykresem 2D w skali liniowej
	# klas cyfr w przestrzeni niejawnej
    grid_x = np.linspace(xmin, xmax, n)
    grid_y = np.linspace(ymin, ymax, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z = np.array([[xi, yi]])
            x_decoded = decoder.predict(z)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# ładujemy zbiór MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# zmiana kształtu na (28,28,2) i normalizacja obrazów wejściowych
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# parametry sieci
input_shape = (image_size, image_size, 1)
batch_size = 32
kernel_size = 3
latent_dim = 2
# Liczba warstw CNN kodera/dekodera i liczba filtrów na warstwę
layer_filters = [32, 64]

# budowanie modelu sieci autokodującej
# najpierw budujemy model kodera
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# stos Conv2D(32)-Conv2D(64)
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

# informacja o kształcie danych jest potrzebna by zbudować model dekodera
# i nie wykonywać ręcznie obliczeń. Wejście na pierwszą Conv2DTranspose 
# będzie miało taki kształt.
# kształt to (7,7,64), będzie przetworzony przez dekoder z powrotem do (28,28,1)


shape = K.int_shape(x)

# tworzymy wektor niejawny
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# tworzymy instancję modelu kodera
encoder = Model(inputs, latent, name='encoder')
encoder.summary()
plot_model(encoder, to_file='encoder.png', show_shapes=True)

# budujemy model dekodera
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
# używamy zapisanego wcześniej kształtu (7,7,64) 
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
# z wektora do odpowiedniego kształtu lub transponowanego conv
x = Reshape((shape[1], shape[2], shape[3]))(x)

# stos Conv2DTranspose(64)-Conv2DTranspose(32)

for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)

# rekonstrukcja wejścia
outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# tworzenie instancji modelu dekodera
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='decoder.png', show_shapes=True)

# sieć autokodująca =  koder + dekoder
# tworzenie instancji modelu sieci autokodującej
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()
plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)

# Funkcja straty MSE, optymalizator Adam
autoencoder.compile(loss='mse', optimizer='adam')

# trenowanie sieci autokodującej
autoencoder.fit(x_train,
                x_train,
                validation_data=(x_test, x_test),
                epochs=20,
                batch_size=batch_size)

# przewidywanie odpowiedzi sieci autokodującej na danych testowych
x_decoded = autoencoder.predict(x_test)

# wyświetlenie pierwszych ośmiu wejściowych i zrekonstruowanych obrazów
imgs = np.concatenate([x_test[:8], x_decoded[:8]])
imgs = imgs.reshape((4, 4, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Wejście: pierwsze dwa wiersze,/n Wyjście: ostatnie dwa wiersze')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('input_and_decoded.png')
plt.savefig('input_and_decoded.tif')
plt.show()

# projekcja dwuwymiarowego wektora niejawnego na przestrzeń 2D
models = (encoder, decoder)
data = (x_test, y_test)
plot_results(models, data,
             batch_size=batch_size,
             model_name="autoencoder-2dim")
