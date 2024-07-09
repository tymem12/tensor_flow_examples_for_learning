"""Trenowanie stuwarstwowej sieci DenseNet na zbiorze CIFAR10.

Z dogenerowaniem danych:
Dokładność wyższa niż 93.55% na zbiorze testowym w ciągu 200 epok
225 sekund na epokę na GTX 1080Ti

Densely Connected Convolutional Networks
https://arxiv.org/pdf/1608.06993.pdf
http://openaccess.thecvf.com/content_cvpr_2017/papers/
    Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf
Poniższa sieć jest podobna do stuwarstwowej DenseNet-BC (k=12)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Conv2D, BatchNormalization
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Input, Flatten, Dropout
from keras.layers import concatenate, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import plot_model
from keras.utils import to_categorical
import os
import numpy as np
import math

# parametry uczenia
batch_size = 32
epochs = 200
data_augmentation = True

# parametry sieci
num_classes = 10
num_dense_blocks = 3
use_max_pool = False

# DenseNet-BC z dogenerowaniem danych
# współczynnik wzrostu  | Głębokość |  Dokładność (artykuł) | Dokładność (tutaj)   |
# 12                    | 100       |  95.49%               | 93.74%               |
# 24                    | 250       |  96.38%               | requires big mem GPU |
# 40                    | 190       |  96.54%               | requires big mem GPU |
growth_rate = 12
depth = 100
num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)

num_filters_bef_dense_block = 2 * growth_rate
compression_factor = 0.5

# Załadowanie zbioru CIFAR10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# wymiary obrazów wejściowych
input_shape = x_train.shape[1:]

# normalizacja danych
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# konwersja wektorów klas na binarne macierze klas
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def lr_schedule(epoch):
    """Harmonogramowanie współczynnika uczenia

    Współczynnik uczenia jest redukowany po 80, 120, 160, 180 epoce.
    Automatyczne wywolywanie w każxej epoce jako część wywołania zwrotnego podczas uczenia.

    # Argumenty
        epoch (int): Liczba epok

    # Zwraca
        lr (float32): współczynnik uczenia
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Wspolczynnik uczenia: ', lr)
    return lr


# początek definicji modelu
# głęboka sieć CNN, kompleks złożony z BN-ReLU-Conv2D
inputs = Input(shape=input_shape)
x = BatchNormalization()(inputs)
x = Activation('relu')(x)
x = Conv2D(num_filters_bef_dense_block,
           kernel_size=3,
           padding='same',
           kernel_initializer='he_normal')(x)
x = concatenate([inputs, x])

# stos bloków gęstych połączonych warstwami przekształcającymi
for i in range(num_dense_blocks):
    # blok gęsty jest stosem warstw zwężających
    for j in range(num_bottleneck_layers):
        y = BatchNormalization()(x)
        y = Activation('relu')(y)
        y = Conv2D(4 * growth_rate,
                   kernel_size=1,
                   padding='same',
                   kernel_initializer='he_normal')(y)
        if not data_augmentation:
            y = Dropout(0.2)(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(growth_rate,
                   kernel_size=3,
                   padding='same',
                   kernel_initializer='he_normal')(y)
        if not data_augmentation:
            y = Dropout(0.2)(y)
        x = concatenate([x, y])

     # bez warstwy przekształcającej po ostatnim gęstym bloku
    if i == num_dense_blocks - 1:
        continue

    # warstwa przekształcająca kompresuje liczbę map cech i redukuje rozmiar
    num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
    num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
    y = BatchNormalization()(x)
    y = Conv2D(num_filters_bef_dense_block,
               kernel_size=1,
               padding='same',
               kernel_initializer='he_normal')(y)
    if not data_augmentation:
        y = Dropout(0.2)(y)
    x = AveragePooling2D()(y)

# dodanie klasyfikatora na szczycie
# łączenie-średnia, rozmiar mapy cech 1 x 1
x = AveragePooling2D(pool_size=8)(x)
y = Flatten()(x)
outputs = Dense(num_classes,
                kernel_initializer='he_normal',
                activation='softmax')(y)
# stworzenie instancji i kompilacja modelu
# W oryginalnym artykule użyto SGD, ale RMSprop działa lepiej dla DenseNet.
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(1e-3),
              metrics=['acc'])
model.summary()

# włącz to jeśli możesz zainstalować pydot
# pip install pydot
#plot_model(model, to_file="cifar10-densenet.png", show_shapes=True)

# przygotowanie katalogu do zapisu modelu
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_densenet_model.{epoch:02d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# przygotowanie wywołań zwrotnych do zapisywania modelu i redukcji współczynnika uczenia
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# uruchom trening z, lub bez dogenerowywania danych
if not data_augmentation:
    print('Bez dogenerowania danych.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Dane dogenerowywane w czasie rzeczywistym.')
    # wstępne przygotowanie i dogenerowanie danych w czasie rzeczywistym
    datagen = ImageDataGenerator(
        featurewise_center=False,  # ustaw średnia wejścia na 0 dla całego zbioru
        samplewise_center=False,  # ustaw średnią dla każdej próbki na 0
        featurewise_std_normalization=False,  # podziel wejścia przez odchylenie standardowe  zbioru
        samplewise_std_normalization=False,  # podziel każde wejście przez jego odchylenie standardowe
        zca_whitening=False,  # zastosuj wybielanie ZCA
        rotation_range=0,  # losowo obróć obrazy w zakresie od 0 do 180 stopni
        width_shift_range=0.1,  # losowo przesuń obrazy w poziomie
        height_shift_range=0.1,  # losowo przesuń obrazy w pionie
        horizontal_flip=True,  # losowo odbij obrazy w poziomie
        vertical_flip=False)  # losowo odbij obrazy w pionie

    # obliczenie wartości liczbowych potrzebnych do normalizacji cech
    # (średnia, odchylenie standardowe i składowe główne jeśli użyto wybielania)
    datagen.fit(x_train)

    steps_per_epoch = math.ceil(len(x_train) / batch_size)
    # dopasowywanie modelu partiami generowanymi przez datagen.flow().
    model.fit(x=datagen.flow(x_train, y_train, batch_size=batch_size),
              verbose=1,
              epochs=epochs,
              validation_data=(x_test, y_test),
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks)


    # dopasowanie modelu na partiach generowanych przez datagen.flow()
    #model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
    ##                    steps_per_epoch=x_train.shape[0] // batch_size,
    #                    validation_data=(x_test, y_test),
    #                    epochs=epochs, verbose=1,
    #                    callbacks=callbacks)

# ocena wytrenowanego modelu
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
