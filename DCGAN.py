'''
DCGAN- deep convulsion generative adversary network


Generator próbuje oszukać dyskryminator generując fałszywe obrazy.
Dyskryminator uczy się odróżniać obrazy fałszywe od prawdziwych.
Razem generator i dyskryminator tworzą sieć współzawodniczącą.
W DCGAN generator i dyskryminator są uczone naprzemiennie. Podczas treningu
dyskryminator nie tylko uczy się odróżniania obrazów prawdziwych 
od fałszywych, ale również 'podpowiada' generatorowi jak ulepszyć 
tworzone fałszywe obrazy.

[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
from keras.models import load_model

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse


def build_generator(inputs, image_size):
    """Konstruowanie modelu generatora

    Stos warstw BN-ReLU-Conv2DTranpose do generowania fałszywych obrazów
    Wyjściową funkcją aktywacji jest sigmoida zamiast tanh (jak w [1]).
    Sigmoida łatwiej osiąga zbieżność.

    Argumenty:
        inputs (Layer): warstwa wejściowa generatora — wektor z
        image_size (tensor): rozmiar jednego boku docelowego obrazu
        (zakładamy kształt kwadratu)

    Zwraca:
        generator (Model): model generatora
    """

    image_resize = image_size // 4
    # parametry sieci
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]

    x = Dense(image_resize * image_resize * layer_filters[0])(inputs)
    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

    for filters in layer_filters:
        # pierwsze dwie warstwy splotowe używają kroku 2 (strides = 2)
        # ostatnie dwie używają kroku 1 (strides = 1)
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same')(x)

    x = Activation('sigmoid')(x)
    generator = Model(inputs, x, name='generator')
    return generator


def build_discriminator(inputs):
    """Konstruowanie modelu dyskryminatora

    Stos warstw LeakyReLU-Conv2D do odróżniania obrazów prawdziwych 
    od fałszywych. Sieć nie osiąga zbieżności, gdy użyto BN, 
    więc go tu nie stosujemy w przeciwieństwie do oryginalnego artykułu [1].

    Argumenty:
        inputs (Layer): wejściowa warstwa dyskryminatora (obraz)

    Zwraca:
        discriminator (Model): model dyskryminatora
    """
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]

    x = inputs
    for filters in layer_filters:
        # pierwsze trzy warstwy splotowe używają kroku 2 (strides = 2)
        # ostatnia używa kroku 1 (strides = 1)
        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides = 2
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same')(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    discriminator = Model(inputs, x, name='discriminator')
    return discriminator


def train(models, x_train, params):
    """Uczenie dyskryminatora i sieci współzawodniczącej

    Naprzemienne uczenie dyskryminatora i sieci współzawodniczącej na próbkach.
    Najpierw uczony jest dyskryminator na prawdziwych i fałszywych obrazach.
    Następnie uczony jest oponent z obrazami fałszywymi mającymi udawać prawdziwe.
    Generowanie przykładowych obrazów co save_interval.

    Argumenty:
        models (list): generator, dyskryminator, model sieci
        x_train (tensor): obrazy uczące
        params (list): parametry sieci

    """
    # komponenty modelu GAN
    generator, discriminator, adversarial = models
    # parametry sieci
    batch_size, latent_size, train_steps, model_name = params
    # obraz generatora jest zapisywany co 500 kroków
    save_interval = 500
    # wektor szumu pozwalający widzieć ewolucję wyjść generatora podczas uczenia
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    # liczba elementów w zbiorze uczącym
    train_size = x_train.shape[0]
    for i in range(train_steps):
        # uczenie dyskryminatora dla 1 próbki
        # 1 próbka prawdziwych (etykieta=1.0) i podrobionych obrazów (etykieta=0.0)
        # losowy wybór prawdziwego obrazu ze zbioru
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_images = x_train[rand_indexes]
        # generowanie fałszywych obrazów z szumu przy użyciu generatora
        # generowanie szumu z użyciem rozkładu jednostajnego
        noise = np.random.uniform(-1.0,
                                  1.0,
                                  size=[batch_size, latent_size])
        # generowanie fałszywych obrazów
        fake_images = generator.predict(noise)
        # prawdziwy + fałszywy obraz = 1 próbka danych uczących
        x = np.concatenate((real_images, fake_images))
        # etykiety prawdziwego i fałszywego obrazu
        # etykieta prawdziwego to 1.0
        y = np.ones([2 * batch_size, 1])
        # etykieta fałszywego to 0.0
        y[batch_size:, :] = 0.0
        # uczenie sieci dyskryminatora, zapis funkcji straty i dokładności
        loss, acc = discriminator.train_on_batch(x, y)
        log = "%d: [Funkcja straty dyskryminatora: %f, dokładność: %f]" % (i, loss, acc)

        # uczenie sieci współzawodniczącej dla 1 próbki
        # 1 próbka fałszywych obrazów z etykietą równą 1.0
        # ponieważ wagi dyskryminatora są „zamrożone” w sieci współzawodniczącej 
        # tylko generator podlega uczeniu
        # generowanie szumu z wykorzystaniem rozkładu jednostajnego
        noise = np.random.uniform(-1.0,
                                  1.0,
                                  size=[batch_size, latent_size])
        # etykietuj fałszywe obrazy jako prawdziwe lub 1.0
        y = np.ones([batch_size, 1])
        # uczenie sieci współzawodniczącej
        # zauważ, że w przeciwieństwie do uczenia dyskryminatora
        # nie zapisujemy fałszywych obrazów w zmiennej
        # fałszywe obrazy są przesyłane na wejście dyskryminatora sieci współzawodniczącej
        # do klasyfikacji
        # zapis funkcji straty i dokładności
        loss, acc = adversarial.train_on_batch(noise, y)
        log = "%s [Funkcja straty sieci: %f, dokładność: %f]" % (log, loss, acc)
        print(log)
        if (i + 1) % save_interval == 0:
            # okresowy wydruk obrazów z generatora
            plot_images(generator,
                        noise_input=noise_input,
                        show=False,
                        step=(i + 1),
                        model_name=model_name)

    # zapisanie modelu wytrenowanego generatora
    # wytrenowany generator może być potem ponownie użyty
    # do ponownego generowania cyfr MNIST
    generator.save(model_name + ".h5")


def plot_images(generator,
                noise_input,
                show=False,
                step=0,
                model_name="gan"):
    """Generowanie i wyświetlanie fałszywych obrazów

    Aby pokazać działanie generujemy fałszywe obrazy
    i wyświetlamy je na kwadratowej siatce.

    Argumenty:
        generator (Model): Model generatora tworzącego fałszywe obrazy
        noise_input (ndarray): Tablica wektorów z
        show (bool): pokazywać czy nie?
        step (int): Dodanie do nazwy pliku zapisywanego obrazu
        model_name (string): Nazwa modelu

    """
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    images = generator.predict(noise_input)
    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')


def build_and_train_models():
    # załaduj zbiór MNIST
    (x_train, _), (_, _) = mnist.load_data()

    # Przekształcenie danych dla CNN do postaci (28, 28, 1) i normalizacja
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255

    model_name = "dcgan_mnist"
    # parametry sieci
    # niejawny wektor z ma 100 wymiarów
    latent_size = 100
    batch_size = 64
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, 1)

    # konstruowanie modelu dyskryminatora
    inputs = Input(shape=input_shape, name='discriminator_input')
    discriminator = build_discriminator(inputs)
    # [1] w oryginalnym artykule użyto adam, 
    # ale dyskryminator osiąga łatwiej zbieżność z RMSprop
    optimizer = RMSprop(lr=lr, decay=decay)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])
    discriminator.summary()

    # Konstruowanie modelu generatora
    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    generator = build_generator(inputs, image_size)
    generator.summary()

    # Budowanie modelu sieci
    optimizer = RMSprop(lr=lr * 0.5, decay=decay * 0.5)
    # zamrożenie wag dyskryminatora przy trenowaniu sieci GAN
    discriminator.trainable = False
    # sieć współzawodnicząca (GAN) = generator + dyskryminator
    adversarial = Model(inputs, 
                        discriminator(generator(inputs)),
                        name=model_name)
    adversarial.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()

    # uczenie dyskryminatora i sieci współzawodniczącej
    models = (generator, discriminator, adversarial)
    params = (batch_size, latent_size, train_steps, model_name)
    train(models, x_train, params)



def test_generator(generator):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    plot_images(generator,
                noise_input=noise_input,
                show=True,
                model_name="test_outputs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Ładowanie modelu generatora z wytrenowanymi wagami z pliku h5"
    parser.add_argument("-g", "--generator", help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        test_generator(generator)
    else:
        build_and_train_models()
