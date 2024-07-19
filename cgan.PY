'''Uczenie sieci CGAN na zbiorze MNIST z użyciem Keras

CGAN to warunkowa generująca sieć współzawodnicząca.
Ta wersja CGAN jest podobna do DCGAN. Główną różnicą jest to,
że wektor 'z' generatora jest warunkowany przez etykietę OH
(by generować określone fałszywe obrazy). Dyskryminator jest uczony 
aby rozróżniać obrazy prawdziwe od fałszywych (warunkowanych
specyficzną etykietą OH).

[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

[2] Mirza, Mehdi, and Simon Osindero. "Conditional generative
adversarial nets." arXiv preprint arXiv:1411.1784 (2014).
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse


def build_generator(inputs, labels, image_size):
    """Konstruowanie modelu generatora 

    Wejścia są łączone przed warstwą gęstą 
    Stos warstw BN-ReLU-Conv2DTranpose do generowania fałszywych obrazów.
    Funkcją aktywacji warstwy wyjściowej jest sigmoida (zamiast tanh jak w oryginalnym DCGAN).
    Sigmoida zapewnia łatwiejsze osiągnięcie zbieżności

    Argumenty:
        inputs (Layer): warstwa wejściowa generatora (wektor z)
        labels (Layer): warstwa wejściowa dla wektora OH, by nałożyć warunki na wejścia
        image_size: docelowy rozmiar jednego boku (zakładamy, że obrazy są kwadratowe)

    Zwraca:
        generator (Model): model generatora
    """

    image_resize = image_size // 4
    # parametry sieci
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]

    x = concatenate([inputs, labels], axis=1)
    x = Dense(image_resize * image_resize * layer_filters[0])(x)
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
    # wejście jest warunkowane etykietami
    generator = Model([inputs, labels], x, name='generator')
    return generator


def build_discriminator(inputs, labels, image_size):
    """Konstruowanie modelu dyskryminatora

    Wejścia są łączone po warstwie gęstej.
    Stos warstw LeakyReLU-Conv2D do odróżniania prawdziwych i fałszywych
    Sieć nie uzyskuje zbieżności z BN, więc go tu nie używamy
    w przeciwieństwie do artykułu o DCGAN.

    Argumenty:
        inputs (Layer): warstwa wejściowa dyskryminatora (obraz)
        labels (Layer): warstwa wejściowa dla wektora OH do nałożenia warunku na wejścia
        image_size: docelowy rozmiar jednego boku (zakładamy kwadratowy kształt obrazu)

    Zwraca:
        discriminator (Model): model dyskryminatora
    """

    kernel_size = 5
    layer_filters = [32, 64, 128, 256]

    x = inputs

    y = Dense(image_size * image_size)(labels)
    y = Reshape((image_size, image_size, 1))(y)
    x = concatenate([x, y])

    for filters in layer_filters:
        # pierwsze trzy warstwy splotowe używają strides = 2
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
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    # Etykiety nakładają warunki na wejścia 
    discriminator = Model([inputs, labels], x, name='discriminator')
    return discriminator


def train(models, data, params):
    """Trenowanie dyskryminatora i sieci współzawodniczącej

    Naprzemienne trenowanie dyskryminatora i sieci współzawodniczącej przez próbki danych.
    Najpierw trenujemy dyskryminator z poprawnie zaetykietowanymi obrazami rzeczywistymi i sztucznymi.
    Następnie obrazami sztucznymi oznaczonymi jako prawdziwe trenowana jest sieć współzawodnicząca.
    Wejścia dyskryminatora są warunkowane przez etykiety ze zbioru treningowego dla obrazów prawdziwych
    i etykietami losowymi dla fałszywych.
    Wejścia sieci współzawodniczącej są warunkowane etykietami losowymi.
    Generowanie przykładowych obrazów co save_interval.

    Argumenty:
        models (list): generator, dyskryminator, model sieci współzawodniczącej
        data (list): x_train, y_train dane uczące
        params (list): parametry sieci

    """
    # modele GAN

    generator, discriminator, adversarial = models
    # obrazy i etykiety
    x_train, y_train = data
    # parametry sieci
    batch_size, latent_size, train_steps, num_labels, model_name = params
    # obraz generatora jest zapisywany co 500 kroków
    save_interval = 500
    # wektor szumu do oceny postępów generatora podczas trenowania
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    # etykieta dla warunkowania szumu
    noise_class = np.eye(num_labels)[np.arange(0, 16) % num_labels]
    # liczba elementów zbioru uczącego
    train_size = x_train.shape[0]

    print(model_name,
          "Etykiety dla generowanych obrazów",
          np.argmax(noise_class, axis=1))

    for i in range(train_steps):
        # uczenie dyskryminatora dla jednej próbki
        # 1 próbka rzeczywistych (etykieta=1.0) i wygenerowanych (etykieta=0.0) obrazów
        # wybór obrazu rzeczywistego ze zbioru — przypadkowy 
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_images = x_train[rand_indexes]
        # etykiety OH dla obrazów rzeczywistych
        real_labels = y_train[rand_indexes]
        # generowanie sztucznych obrazów przez generator z szumu
        # generowanie szumu według rozkładu jednostajnego
        noise = np.random.uniform(-1.0,
                                  1.0,
                                  size=[batch_size, latent_size])
        # przypisanie przypadkowej etykiety OH
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]

        # generowanie sztucznych obrazów warunkowanych sztucznymi etykietami
        fake_images = generator.predict([noise, fake_labels])
        # 1 partia danych uczących = 1 prawdziwy + 1 generowany obraz
        x = np.concatenate((real_images, fake_images))
        # 1 partia etykiet w czasie uczenia = OH dla prawdziwego obrazu + OH dla generowanego
        labels = np.concatenate((real_labels, fake_labels))
        
        # etykietowanie obrazów prawdziwych i generowanych
        # etykieta prawdziwych to 1.0
        y = np.ones([2 * batch_size, 1])
        # etykieta generowanych sztucznie to 0.0
        y[batch_size:, :] = 0.0
        # uczenie sieci dyskryminatora, zapisanie funkcji straty i dokładności
        loss, acc = discriminator.train_on_batch([x, labels], y)
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

        # uczenie sieci współzawodniczącej dla jednej partii danych
        # jedna partia sztucznych obrazów warunkowana sztucznymi etykietami OH z etykietą =1.0
        # Ponieważ wagi dyskryminatora są zamrożone w sieci współzawodniczącej,
        # uczony jest tylko generator.
        # generowanie szumu według rozkładu jednostajnego
      
        noise = np.random.uniform(-1.0,
                                  1.0,
                                  size=[batch_size, latent_size])
        # przypisanie losowych etykiet OH
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        # etykietowanie sztucznych obrazów jako prawdziwe (lub 1.0)
        y = np.ones([batch_size, 1])
        # uczenie sieci współzawodniczącej 
        # Zauważ, że w odróżnieniu od uczenia dyskryminatora nie zapisujemy sztucznych obrazów do zmiennej; 
        # sztuczne obrazy są przekazywane na wejście dyskryminatora sieci współzawodniczącej do sklasyfikowania.
        # zapis funkcji straty i dokładności
        loss, acc = adversarial.train_on_batch([noise, fake_labels], y)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        print(log)
        if (i + 1) % save_interval == 0:
            # okresowe wyświetlanie obrazów generatora
            plot_images(generator,
                        noise_input=noise_input,
                        noise_class=noise_class,
                        show=False,
                        step=(i + 1),
                        model_name=model_name)
    
    # zapisanie modelu po wytrenowaniu generatora
    # wytrenowany generator może być użyty w przyszłości 
    # do generowania cyfr wzorowanych na MNIST
    generator.save(model_name + ".h5")


def plot_images(generator,
                noise_input,
                noise_class,
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
    images = generator.predict([noise_input, noise_class])
    print(model_name , " labels for generated images: ", np.argmax(noise_class, axis=1))
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
    # załadowanie zbioru MNIST
    (x_train, y_train), (_, _) = mnist.load_data()

    # Przekształcenie danych na (28, 28, 1) oraz normalizacja
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255

    num_labels = np.amax(y_train) + 1
    y_train = to_categorical(y_train)

    model_name = "cgan_mnist"
    # parametry sieci
    # wektor niejawny ma 100 wymiarów
    latent_size = 100
    batch_size = 64
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    label_shape = (num_labels, )

    # utworzenie modelu dyskryminatora
    inputs = Input(shape=input_shape, name='discriminator_input')
    labels = Input(shape=label_shape, name='class_labels')

    discriminator = build_discriminator(inputs, labels, image_size)
    # W [1] lub oryginalnym artykule użyto Adam, 
    # ale dyskryminator łatwiej osiąga zbieżność z RMSprop
    optimizer = RMSprop(lr=lr, decay=decay)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])
    discriminator.summary()

    # utworzenie modelu generatora
    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    generator = build_generator(inputs, labels, image_size)
    generator.summary()

    # utworzenie sieci współzawodniczącej: generator + dyskryminator
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    # zamrożenie wag dyskryminatora podczas uczenia sieci współzawodniczącej
    discriminator.trainable = False
    outputs = discriminator([generator([inputs, labels]), labels])
    adversarial = Model([inputs, labels],
                        outputs,
                        name=model_name)
    adversarial.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()

    # uczenie dyskryminatora i sieci współzawodniczącej
    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)
    params = (batch_size, latent_size, train_steps, num_labels, model_name)
    train(models, data, params)


def test_generator(generator, class_label=None):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    step = 0
    if class_label is None:
        num_labels = 10
        noise_class = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_class = np.zeros((16, 10))
        noise_class[:,class_label] = 1
        step = class_label

    plot_images(generator,
                noise_input=noise_input,
                noise_class=noise_class,
                show=True,
                step=step,
                model_name="test_outputs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Ładowanie wytrenowanego modelu generatora z pliku h5"
    parser.add_argument("-g", "--generator", help=help_)
    help_ = "Podaj cyfrę którą chcesz generować"
    parser.add_argument("-d", "--digit", type=int, help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        class_label = None
        if args.digit is not None:
            class_label = args.digit
        test_generator(generator, class_label)
    else:
        build_and_train_models()
