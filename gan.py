'''Konstruktor modelu sieci GAN i funkcje pomocnicze

[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

'''


from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model

import numpy as np
import math
import matplotlib.pyplot as plt
import os

def generator(inputs,
              image_size,
              activation='sigmoid',
              labels=None,
              codes=None):
    """Budowa modelu generatora

    Stos warstw BN-ReLU-Conv2DTranpose do generowania fałszywych obrazów.
    Funkcją aktywacji warstwy wyjściowej jest sigmoida zamiast tanh jak w [1].
    Przy sigmoidzie łatwo osiągnąć zbieżność.

    Argumenty:
        inputs (Layer): wejściowa warstwa generatora (wektor z)
        image_size (int): docelowy rozmiar jednego boku 
            (przy założeniu obrazu o kształcie kwadratu)
        activation (string): nazwa wyjściowej warstwy aktywacji
        labels (tensor): etykiety wejściowe
        codes (list): dwuwymiarowe rozplątane kodowania dla InfoGAN

    Zwraca:
        Model: model generatora
    """

    image_resize = image_size // 4
    # parametry sieci
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]

    if labels is not None:
        if codes is None:
            # etykiety ACGAN
            # połączenie etykiet OH oraz wektora szumu z
            inputs = [inputs, labels]
        else:
            # kody infoGAN 
            # połącz wektor szumu z, etykiety OH oraz kody 1 i 2
            inputs = [inputs, labels] + codes
        x = concatenate(inputs, axis=1)
    elif codes is not None:
        # generator 0 — StackedGAN
        inputs = [inputs, codes]
        x = concatenate(inputs, axis=1)
    else:
        # domyślne wejście to stuwymiarowy szum (z-code)
        x = inputs

    x = Dense(image_resize * image_resize * layer_filters[0])(x)
    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

    for filters in layer_filters:
        # dla pierwszych dwóch warstw splotowych krok = 2 (strides = 2)
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

    if activation is not None:
        x = Activation(activation)(x)

    # wyjście generatora jest sztucznie wygenerowanym obrazem x 
    return Model(inputs, x, name='generator')


def discriminator(inputs,
                  activation='sigmoid',
                  num_labels=None,
                  num_codes=None):
    """ Budowanie modelu dyskryminatora

    Stos warstw LeakyReLU-Conv2D do odróżniania prawdziwych od fałszywych
    Sieć nie osiąga zbieżności z BN, więc nie jest tu użyty w przeciwieństwie do [1]

    Argumenty:
        inputs (Layer): warstwa wejściowa dyskryminatora (obraz)
        activation (string): nazwa wyjściowej warstwy aktywacji
        num_labels (int): wymiary etykiety OH dla ACGAN i InfoGAN
        num_codes (int): num_codes-dim sieci Q jako wyjście, 
                jeśli StackedGAN lub 2 sieci Q InfoGAN

    Zwraca:
        Model: model dyskryminatora
    """

    kernel_size = 5
    layer_filters = [32, 64, 128, 256]

    x = inputs
    for filters in layer_filters:
        # pierwsze 3 warstwy splotowe używają kroku 2 (strides = 2)
        # ostatnia używa strides = 1
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
    # zdefiniowanie domyślnego wyjścia jako prawdopodobieństwa, że obraz jest prawdziwy
    outputs = Dense(1)(x)
    if activation is not None:
        print(activation)
        outputs = Activation(activation)(outputs)

    if num_labels:
        # ACGAN i InfoGAN mają drugie wyjście
        # drugie wyjście jest 10-wymiarowym wektorem OH etykiet
        layer = Dense(layer_filters[-2])(x)
        labels = Dense(num_labels)(layer)
        labels = Activation('softmax', name='label')(labels)
        if num_codes is None:
            outputs = [outputs, labels]
        else:
            # InfoGAN ma trzecie i czwarte wyjście
            # trzecie wyjście jest jednowymiarową ciągłą siecią Q pierwszego c przy zadanym x
            code1 = Dense(1)(layer)
            code1 = Activation('sigmoid', name='code1')(code1)

            # czwarte wyjście jest jednowymiarowym(ą) ciągłą Q dla drugiego c przy zadanym x
            code2 = Dense(1)(layer)
            code2 = Activation('sigmoid', name='code2')(code2)

            outputs = [outputs, labels, code1, code2]
    elif num_codes is not None:
        # Wyjście Q0 ze StackedGAN
        # z0_recon jest rekonstrukcją normalnego rozkładu z0
        z0_recon =  Dense(num_codes)(x)
        z0_recon = Activation('tanh', name='z0')(z0_recon)
        outputs = [outputs, z0_recon]

    return Model(inputs, outputs, name='discriminator')


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
                noise_label=None,
                noise_codes=None,
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
    for 

    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    rows = int(math.sqrt(noise_input.shape[0]))
    if noise_label is not None:
        noise_input = [noise_input, noise_label]
        if noise_codes is not None:
            noise_input += noise_codes

    images = generator.predict(noise_input)
    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
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


def test_generator(generator):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    plot_images(generator,
                noise_input=noise_input,
                show=True,
                model_name="test_outputs")

