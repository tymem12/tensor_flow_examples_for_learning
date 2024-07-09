"""Training ResNet on CIFAR10.

ResNet v1
[a] Deep Residual Learning for Image Recognition

If we have the RestNet 20, it means that the depth = 3, because 3 * 6 + 2 = 20
it is like 3 (we do it always 3 times) * 2 (every rest block contains 2 layers) * n (number of rest blocks ) + 2

x_l = ReLU(F(x_l_minus_1) + x_l_minus_2)

F(x_l_minus1)  -> Conv2D-BN  called residual mapping

https://arxiv.org/pdf/1512.03385.pdf

ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Conv2D
from keras.layers import BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input
from keras.layers import Flatten, add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
import os
import math

batch_size = 32 # in the original article the batch size was 128 
epochs = 200
data_augmentation = True
num_classes = 10

# setting subtract_pixel_mean improves the accuracy 
subtract_pixel_mean = True

# Parametry modelu
# ----------------------------------------------------------------------------
#           |      | 200-epok    | Oryg Artyk  | 200-epok    | Oryg Artyk| sek/epokę
# Model     |  n   | ResNet v1   | ResNet v1    | ResNet v2   | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Dokładność | %Dokładność | %Dokładność | %Dokładność| v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 3

# versions of models:
# original article: version = 1 (ResNet v1), 
# better ResNet: version = 2 (ResNet v2)
version = 1

# calculating the depth 
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

model_type = 'ResNet%dv%d' % (depth, version)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
input_shape = x_train.shape[1:]

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# jeśli subtract_pixel_mean jest True
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print(' x_train:', x_train.shape)
print(x_train.shape[0], 'training observations')
print(x_test.shape[0], 'testing observations')
print(' y_train:', y_train.shape)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    """
    Schedule of the learning rate:
    lr need to change after certain epochs. 
    Needs to be reduced after 80, 120, 160 and 180 epoch
    It is called in each epoch

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
    print('learning rate ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """Constructing the stack of the layers 2D Convolution-Batch Normalization-Activation

    Arguments:
        inputs (tensor): tensor from the input image or from the previous layer 
        num_filters (int): number of filters Conv2D
        kernel_size (int): shape od the squared kernel
        strides (int): size of the stride Conv2D
        activation (string): activation function
        batch_normalization (bool): perform batch normalization
        conv_first (bool): conv-bn-activation (True) or bn-activation-conv (False)

    Return:
        x (tensor): tensor as the input for the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',  # improve the convergence
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """
    Constructor of thr Restnet v1

    Stack- 2 x (3x3) Conv2D-BN-ReLU ()
    Last layer ReLU is the shortcut connection.
    At the beginning of each step the map of features size is reduced by half. (ang. downsampled) by the conv2D layer with the stride 2, as long as the condition is met.
    in each step the number of filters is the same (it double after every step)
    
    Size of features map:
    step 0: 32x32, 16
    step 1: 16x16, 32
    step 2: 8x8, 64
  

    Arguments:
        input_shape (tensor): tensor of the input image
        depth (int): number of depth
        num_classes (int): number of classes (CIFAR10 has 10)

    Return:
        model (Model): Keras model
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('Depth is not 6n+2 (tzn. 20, 32, w [a])')

    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # stack of the rest layers
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            # pierwsza warstwa (ale nie stos)
            if stack > 0 and res_block == 0:
                strides = 2  # (ang. downsample)
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides) # Conv2D - BN- ReLU
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None) # Conv2D - BN
            
            # pierwsza warstwa (ale nie stos)
            if stack > 0 and res_block == 0:
                # linear projection of the rest shortcut so we could add it (for the first layer)
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # dodanie klasyfikatora na górze
    # v1 nie używa BN po ostatnim połączeniu skrótowym-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # stworzenie instancji modelu
    model = Model(inputs=inputs, outputs=outputs)
    return model




def resnet_v2(input_shape, depth, num_classes=10):
    """Konstruktor sieci ResNet w wersji 2 [b]

    Stos warstw BN-ReLU-Conv2D (1x1)-(3x3)-(1x1) 
    również znanych jako warstwa ograniczająca.
    Pierwsze połączenie skrótowe na warstwę to 1x1 Conv2D.
    Drugie i następne połączenia są identyczne.
    Na początku każdego etapu rozmiar mapy cech jest zmniejszany o połowę
    (ang. downsampled) przez warstwę splotową z krokiem 2 tak długo, jak długo 
    jest spełniony warunek. Na każdym etapie warstwy mają taką samą liczbę filtrów  taki sam rozmiar filtrów map.
    Rozmiary map cech:
    conv1  : 32x32,   16
    stage 0: 32x32,   64
    stage 1: 16x16, 128
    stage 2: 8x8,    256

    Argumenty:
        input_shape (tensor): postać tensora obrazu wejściowego
        depth (int): liczba warstw splotowego jądra bazowego
        num_classes (int): liczba klas (CIFAR10 ma 10 klas)

    Zwraca:
        model (Model): instancję modelu Keras
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('glebokosc powinna byc 9n+2 (tzn. 110 w [b])')
    # początek definicji modelu
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # w v2 występuje Conv2D z BN-ReLU na wejściu przed podziałem na dwie ścieżki
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # stworzenie instancji stosu jednostek resztkowych
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                # pierwsza warstwa i pierwszy stos
                if res_block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                # pierwsza warstwa (ale nie stos)
                if res_block == 0:
                    # zmniejszanie rozmiaru/próbkowanie w dół (ang. downsample)
                    strides = 2

            # resztkowa jednostka ograniczająca
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # projekcja liniowa skrótu resztkowego, połączenie — dopasowanie zmienionych wymiarów
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])

        num_filters_in = num_filters_out

    # dodanie klasyfikatora na górze, v2 ma BN-ReLU przed warstwą łączącą
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # stworzenie instancji modelu
    model = Model(inputs=inputs, outputs=outputs)
    return model



if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['acc'])
model.summary()

# włącz jesli możesz uzyć pydot
# pip install pydot
#plot_model(model, to_file="%s.png" % model_type, show_shapes=True)
print(model_type)

# przygotowanie katalogu do zapisania modeli
save_dir = os.path.join(os.getcwd(), 'zapisane_modele')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# przygotowanie wywołań zwrotnych 
# do zapisywania modeli i dopasowań współczynnika uczenia
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

# uruchomienie uczenia z lub bez dogenerowania danych
if not data_augmentation:
    print('Bez dogenerowania danych.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Dogenerowanie danych rzeczywistych.')
    # wykonanie wstępnego przetwarzania i dogenerowania danych rzeczywistych
    datagen = ImageDataGenerator(
        # ustawienie średniej dla zbioru wejściowego na 0
        featurewise_center=False,
        # ustawienie średniej dla każdej próbki na 0
        samplewise_center=False,
        # podział wejść wg odchylenia standardowego zbioru
        featurewise_std_normalization=False,
        # podziałkazdego wejścia według jego odchylenia standardowego
        samplewise_std_normalization=False,
        # zastosowanie wybielania ZCA
        zca_whitening=False,
        # obrót obrazów o kąt z losowego zakresu (0, 180) stopni
        rotation_range=0,
        # przypadkowe przesunięcie obrazu w poziomie
        width_shift_range=0.1,
        # przypadkowe przesuniecie obrazu w pionie
        height_shift_range=0.1,
        # przypadkowe odbicie w poziomie
        horizontal_flip=True,
        # przypadkowe odbicie w pionie
        vertical_flip=False)

    # obliczenia wartości wymaganych do normalizacji w obrebie cech
    # (odchylenie standardowe, średnia 
    # i składowe główne jeśli stosujemy wybielanie ZAC).
    datagen.fit(x_train)

    steps_per_epoch =  math.ceil(len(x_train) / batch_size)
    # dopasowanie modelu na próbkach generowanych przez datagen.flow().
    model.fit(x=datagen.flow(x_train, y_train, batch_size=batch_size),
              verbose=1,
              epochs=epochs,
              validation_data=(x_test, y_test),
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks)


# punktacja wyuczonego modelu
scores = model.evaluate(x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose=0)
print('Strata na zbiorze testowym:', scores[0])
print('Dokładność na zbiorze testowym:', scores[1])
