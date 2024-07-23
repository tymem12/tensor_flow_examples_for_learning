''' ACGAN (Auxiliary classifier GAN) on dataset MNIST with Keras

ACGAN is similar to DCGAN, but main differences are:  
- vector 'z' og generator is conditioned by the OH because (all 0 and one 1 for the accurate label) to generate the specific image.
- discriminator is train to differentiate between real and fake classes and additionally to predict the correct label (correct number) .

[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

[2] Odena, Augustus, Christopher Olah, and Jonathon Shlens. 
"Conditional image synthesis with auxiliary classifier gans." 
arXiv preprint arXiv:1610.09585 (2016).
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model

import numpy as np
import argparse

import sys
# sys.path.append("..")
import gan

def train(models, data, params):
    """Uczenie dyskryminatora i sieci współzawodniczącej
    Naprzemienne uczenie dyskryminatora i sieci oponenta na partiach danych.
    Najpierw jest uczony dyskryminator z prawdziwymi i fałszywymi obrazami 
    oraz odpowiednimi etykietami OH
    Następnie uczona jest sieć oponenta z fałszywymi obrazami opisanymi jako prawdziwe 
    i odpowiednimi etykietami OH.
    Generowanie przykładowych obrazów co save_interval.

    Argumenty:
        models (list): generator, dyskryminator, modele współzawodniczące
        data (list): dane x_train, y_train 
        params (list): parametry sieci
    """
    # modele GAN

    generator, discriminator, adversarial = models
    # obrazy i ich etykiety OH
    x_train, y_train = data
    batch_size, latent_size, train_steps, num_labels, model_name = params
    # obrazy z generatora są zapisywane co każde 500 kroków
    save_interval = 500
    # wektor szumu, by obserwować, jak dane wyjściowe z generatora
    # zmieniają się w czasie treningu
    noise_input = np.random.uniform(-1.0,
                                    1.0, 
                                    size=[16, latent_size])
     # etykiety klas to: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5
    # generator musi wygenerować te konkretne cyfry MNIST
    noise_label = np.eye(num_labels)[np.arange(0, 16) % num_labels]
    # liczba elementów w zbiorze uczącym
    train_size = x_train.shape[0]
    print(model_name,
          "Etykiety dla wygenerowanych obrazów: ",
          np.argmax(noise_label, axis=1))

    for i in range(train_steps):
        # uczenie dyskryminatora dla 1 partii danych
        # 1 partia prawdziwych (etykieta = 1.0) i fałszywych obrazów (etykieta = 0.0)
        # losowy wybór prawdziwych obrazów oraz odpowiednich etykiet ze zbioru danych 
        rand_indexes = np.random.randint(0,
                                         train_size,
                                         size=batch_size)
        real_images = x_train[rand_indexes]
        real_labels = y_train[rand_indexes]
        # generowanie przez generator fałszywych obrazów z szumu
        # generowanie szumu z użyciem rozkładu jednostajnego
        noise = np.random.uniform(-1.0,
                                  1.0,
                                  size=[batch_size, latent_size])
        # losowy wybór etykiet OH
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        #generowanie fałszywych obrazów
        fake_images = generator.predict([noise, fake_labels])
        # prawdziwe+fałszywe obrazy = 1 partia danych uczących
        x = np.concatenate((real_images, fake_images))
        # prawdziwe+fałszywe etykiety = 1 partia etykiet zbioru uczącego
        labels = np.concatenate((real_labels, fake_labels))

        # etykiety prawdziwych i fałszywych obrazów
        # etykieta prawdziwych to 1.0
        y = np.ones([2 * batch_size, 1])
        # etykieta fałszywych to 0.0
        y[batch_size:, :] = 0
        # uczenie sieci dyskryminatora, zapis straty i dokładności
        # ['strata', 'strata_aktywacja_1', 
        # 'strata_etykieta', 'acc_aktywacja _1', 'acc_ etykieta']
        metrics  = discriminator.train_on_batch(x, [y, labels])
        fmt = "%d: [disc loss: %f, srcloss: %f," 
        fmt += "lblloss: %f, srcacc: %f, lblacc: %f]" 
        log = fmt % (i, metrics[0], metrics[1], \
                metrics[2], metrics[3], metrics[4])

        # uczenie sieci współzawodniczącej na jednej partii danych
        # 1 partia fałszywych obrazów z etykietą = 1.0 oraz
        # odpowiednią etykietą OH lub klasą 
        # ponieważ wagi dyskryminatora są zamrożone w sieci współzawodniczącej,
        # tylko generator podlega treningowi
        # generowanie szumu z użyciem rozkładu jednostajnego
        noise = np.random.uniform(-1.0,
                                  1.0, 
                                  size=[batch_size, latent_size])
        # losowy wybór etykiety OH
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        # zaetykietowanie fałszywego obrazu jako prawdziwy
        y = np.ones([batch_size, 1])
        # uczenie sieci współzawodniczącej 
        # zauważ, że w przeciwieństwie do uczenia dyskryminatora 
        # nie zapisujemy fałszywych obrazów do zmiennej
        # fałszywe obrazy są przekazywane na wejście dyskryminatora 
        # sieci współzawodniczącej do sklasyfikowania
        # zapis straty i dokładności
        metrics  = adversarial.train_on_batch([noise, fake_labels], # input of the generator so the noise and the random class- generator would create it
                                              [y, fake_labels])     # output to lie dicriminator ( in second list are the "correct values" based on which discriminator would be lied that that the images are real)
        fmt = "%s [advr loss: %f, srcloss: %f,"
        fmt += "lblloss: %f, srcacc: %f, lblacc: %f]" 
        log = fmt % (log, metrics[0], metrics[1],\
                metrics[2], metrics[3], metrics[4])
        print(log)
        if (i + 1) % save_interval == 0:
            # okresowe wydruki wygenerowanych obrazów
            gan.plot_images(generator,
                        noise_input=noise_input,
                        noise_label=noise_label,
                        show=False,
                        step=(i + 1),
                        model_name=model_name)

    # Zapisanie wyuczonego modelu generatora 
    # wytrenowany generator może zostać ponownie załadowany i użyty 
    # w przyszłości do generowanie cyfr MNIST
    generator.save(model_name + ".h5")


def build_and_train_models():
    """Ładowanie zbioru, budowanie dyskryminatora ACGAN,
    generatora i modelu sieci współzawodniczącej
    Wywołanie procedury uczenia ACGAN
    """
    # załadowanie zbioru MNIST
    (x_train, y_train), (_, _) = mnist.load_data()

    # zmiana rozmiaru dla sieci CNN do (28, 28, 1) i normalizacja
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, 
                         [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255

    # etykiety uczące
    num_labels = len(np.unique(y_train))
    y_train = to_categorical(y_train)

    model_name = "acgan_mnist"
    # parametry sieci
    latent_size = 100
    batch_size = 64
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    label_shape = (num_labels, )

    # budowanie modelu dyskryminatora
    inputs = Input(shape=input_shape,
                   name='discriminator_input')
    # wywołanie konstruktora dyskryminatora 
    # predykcja wyjścia z dwoma wyjściami i z etykietami
    discriminator = gan.discriminator(inputs, 
                                      num_labels=num_labels)
    # W [1] użyto Adam, ale dyskryminator łatwiej osiąga zbieżność z RMSprop
    optimizer = RMSprop(lr=lr, decay=decay)
    # 2 funkcje straty: 
    # - 1) prawdopodobieństwo, że obraz jest prawdziwy
    # - 2) etykieta klasy dla obrazu
    loss = ['binary_crossentropy', 'categorical_crossentropy']
    discriminator.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=['accuracy'])
    discriminator.summary()

    # budowanie modelu generatora
    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    labels = Input(shape=label_shape, name='labels')
    # wywołanie konstruktora generatora z etykietami wejściowymi
    generator = gan.generator(inputs,
                              image_size,
                              labels=labels)
    generator.summary()

    # budowanie modelu sieci współzawodniczącej = generator+dyskryminator
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    # zamrożenie wag dyskryminatora podczas uczenia sieci współzawodniczącej
    discriminator.trainable = False
    adversarial = Model([inputs, labels],
                        discriminator(generator([inputs, labels])),
                        name=model_name)
    # te same dwie funkcje straty: 1) prawdopodobieństwo, że obraz jest prawdziwy
    # 2) klasa etykiety obrazu
    adversarial.compile(loss=loss,
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()

    # uczenie dyskryminatora i sieci współzawodniczącej
    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)
    params = (batch_size, latent_size, \
             train_steps, num_labels, model_name)
    train(models, data, params)


def test_generator(generator, class_label=None):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    step = 0
    if class_label is None:
        num_labels = 10
        noise_label = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_label = np.zeros((16, 10))
        noise_label[:,class_label] = 1
        step = class_label

    gan.plot_images(generator,
                    noise_input=noise_input,
                    noise_label=noise_label,
                    show=True,
                    step=step,
                    model_name="test_outputs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Ładowanie wytrenowanego modelu z pliku h5"
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
