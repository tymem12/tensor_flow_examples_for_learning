'''Learning infoGAN on  MNIST
Ta wersja infoGAN jest podobna do DCGAN. Główną różnicą jest to, 
że wektor 'z' generatora jest warunkowany przez etykietę OH by wygenerować specyficzny, fałszywy obraz. Dyskryminator jest uczony odróżniania obrazów prawdziwych od fałszywych i przewidywania odpowiadających im etykiet OH.


[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).


[2] Chen, Xi, et al. "Infogan: Interpretable representation learning by
information maximizing generative adversarial nets." 
Advances in Neural Information Processing Systems. 2016.
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
from keras import backend as K

import numpy as np
import argparse

import sys
sys.path.append("..")
import gan

# from ..lib import gan

def train(models, data, params):
    """Uczenie dyskryminatora i sieci współzawodniczącej

    Naprzemienne uczenie dyskryminatora i sieci współzawodniczącej na partii danych.
    Najpierw uczony jest dyskryminator — na obrazach fałszywych i prawdziwych,
        odpowiadających im etykietach OH i ciągłych kodach.
    Następnie uczony jest oponent — z obrazami fałszywymi udającymi prawdziwe,
        odpowiadającymi im etykietami OH i ciągłymi kodami.
    Generowanie próbek obrazów co save_interval.

    # Argumenty
        models (Models): generator, dyskryminator, modele współzawodniczące
        data (tuple): dane x_train, y_train 
        params (tuple): parametry sieci
    """
    # modele GAN
    generator, discriminator, adversarial = models
    # obrazy i ich etykiety OH
    x_train, y_train = data
    # parametry sieci
    batch_size, latent_size, train_steps, num_labels, model_name = \
            params
    # obrazy generatora są zapisywane co 500 kroków
    save_interval = 500
    # odchylenie standardowe kodu
    code_std = 0.5
    # wektor szumu, by obserwować 
    # zmiany w wyjściach generatora podczas uczenia
    noise_input = np.random.uniform(-1.0,
                                    1.0,
                                    size=[16, latent_size])
    # losowe etykiety klas i kody
    noise_label = np.eye(num_labels)[np.arange(0, 16) % num_labels]
    noise_code1 = np.random.normal(scale=code_std, size=[16, 1])
    noise_code2 = np.random.normal(scale=code_std, size=[16, 1])
    # liczba elementów w zbiorze uczącym
    train_size = x_train.shape[0]
    print(model_name,
          "Labels for generated images: ",
          np.argmax(noise_label, axis=1))
    for i in range(train_steps):
        # uczenie dyskryminatora na 1 partii
        # 1 partia prawdziwych (etykieta = 1.0) 
        # i fałszywych obrazów (etykieta = 0.0)
        # losowy wybór prawdziwego obrazu 
        # i odpowiadających mu etykiet ze zbioru danych 
        rand_indexes = np.random.randint(0,
                                         train_size,
                                         size=batch_size)
        real_images = x_train[rand_indexes]
        real_labels = y_train[rand_indexes]
        # losowe kody dla prawdziwych obrazów
        real_code1 = np.random.normal(scale=code_std,
                                      size=[batch_size, 1])
        real_code2 = np.random.normal(scale=code_std, 
                                      size=[batch_size, 1])
        # generowanie fałszywych obrazów, etykiet i kodów
        noise = np.random.uniform(-1.0,
                                  1.0, 
                                  size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        fake_code1 = np.random.normal(scale=code_std,
                                      size=[batch_size, 1])
        fake_code2 = np.random.normal(scale=code_std, 
                                      size=[batch_size, 1])
        inputs = [noise, fake_labels, fake_code1, fake_code2]
        fake_images = generator.predict(inputs)

        # obraz prawdziwy+fałszywy = 1 partia danych uczących
        x = np.concatenate((real_images, fake_images))
        labels = np.concatenate((real_labels, fake_labels))
        codes1 = np.concatenate((real_code1, fake_code1))
        codes2 = np.concatenate((real_code2, fake_code2))

        # etykiety prawdziwych i fałszywych obrazów
        # etykieta prawdziwych to 1.0
        y = np.ones([2 * batch_size, 1])
        # etykieta prawdziwych to 0.0
        y[batch_size:, :] = 0

        # uczenie sieci dyskryminatora 
        # zapisanie straty i dokładności etykiety
        outputs = [y, labels, codes1, codes2]
        # metrics = ['loss', 'activation_1_loss', 'label_loss',
        # 'code1_loss', 'code2_loss', 'activation_1_acc',
        # 'label_acc', 'code1_acc', 'code2_acc']
        # z discriminator.metrics_names
        metrics = discriminator.train_on_batch(x, outputs)
        fmt = "%d: [dis: %f, bce: %f, ce: %f, mi: %f, mi:%f, acc: %f]"
        log = fmt % (i, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[6])

        # uczenie sieci współzawodniczącej na 1 partii
        # 1 partia fałszywych obrazów z etykietą = 1.0 oraz
        # odpowiadającymi im etykietami OH lub klasą + losowe kody;
        # ponieważ wagi dyskryminatora są zamrożone,
        # w sieci współzawodniczącej tylko generator jest uczony
        # generowanie fałszywych obrazów, etykiet i kodów
        noise = np.random.uniform(-1.0,
                                  1.0,
                                  size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        fake_code1 = np.random.normal(scale=code_std,
                                      size=[batch_size, 1])
        fake_code2 = np.random.normal(scale=code_std, 
                                      size=[batch_size, 1])
        # etykietowanie fałszywych obrazów jako prawdziwe
        y = np.ones([batch_size, 1])

        # uczenie sieci współzawodniczącej 
        # Zauważ, że w przeciwieństwie do uczenia dyskryminatora
        # nie zapisujemy fałszywych obrazów do zmiennej;
        # są one przekazywane do dyskryminatora.
        # wejście oponenta do klasyfikacji
        # zapis straty i dokładności etykiety
        inputs = [noise, fake_labels, fake_code1, fake_code2]
        outputs = [y, fake_labels, fake_code1, fake_code2]
        metrics  = adversarial.train_on_batch(inputs, outputs)
        fmt = "%s [adv: %f, bce: %f, ce: %f, mi: %f, mi:%f, acc: %f]"
        log = fmt % (log, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[6])

        print(log)
        if (i + 1) % save_interval == 0:
            # okresowy wydruk obrazów z generatora 
            gan.plot_images(generator,
                            noise_input=noise_input,
                            noise_label=noise_label,
                            noise_codes=[noise_code1, noise_code2],
                            show=False,
                            step=(i + 1),
                            model_name=model_name)
   
        # zapisanie modelu po wytrenowaniu generatora
        # Wytrenowany generator może być ponownie załadowany
        # w przyszłości do generowania cyfr MNIST.
        if (i + 1) % (2 * save_interval) == 0:
            generator.save(model_name + ".h5")


def mi_loss(c, q_of_c_given_x):
    """ Informacja wzajemna, równanie 5 w [2];
        zakładamy, że H(c) jest stałe
    """
    # mi_loss = –c * log(Q(c|x))
    return -K.mean(K.sum(c * K.log(q_of_c_given_x + K.epsilon()), 
                                   axis=1))


def build_and_train_models(latent_size=100):
    """Załadowanie zbioru danych, konstruowanie dyskryminatora InfoGAN,
    generatora i modelu współzawodniczącego.
    Wywołanie procedury uczącej InfoGAN.
    """
    # Załadowanie zbioru MNIST
    (x_train, y_train), (_, _) = mnist.load_data()

    # zmiana kształtu danych dla CNN do (28, 28, 1) i normalizacja
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255

    # etykiety treningowe
    num_labels = len(np.unique(y_train))
    y_train = to_categorical(y_train)

    model_name = "infogan_mnist"
    # parametry sieci
    batch_size = 64
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    label_shape = (num_labels, )
    code_shape = (1, )

    # konstruowanie modelu dyskryminatora
    inputs = Input(shape=input_shape, name='discriminator_input')
    # wywołanie konstruktora dyskryminatora z czterema wyjściami: 
    # źródło, etykieta oraz 2 kody
    discriminator = gan.discriminator(inputs,
                                      num_labels=num_labels,
                                      num_codes=2)
    # W [1] użyto Adam, ale dyskryminator łatwiej osiąga zbieżność z RMSprop
    optimizer = RMSprop(lr=lr, decay=decay)
    # funkcja straty: 1) prawdopodobieństwo, że obraz jest prawdziwy
    # (binarna entropia krzyżowa)
    # 2) skategoryzowana entropia krzyżowa etykiety obrazu
    # 3) oraz 4) strata informacji wzajemnej
    loss = ['binary_crossentropy', 
            'categorical_crossentropy', 
            mi_loss, 
            mi_loss]
    # waga lambda lub mi_loss wynosi 0,5
    loss_weights = [1.0, 1.0, 0.5, 0.5]
    discriminator.compile(loss=loss,
                          loss_weights=loss_weights,
                          optimizer=optimizer,
                          metrics=['accuracy'])
    discriminator.summary()

    # konstruowanie modelu generatora
    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    labels = Input(shape=label_shape, name='labels')
    code1 = Input(shape=code_shape, name="code1")
    code2 = Input(shape=code_shape, name="code2")
    # wywołanie generatora z wejściami: 
    # etykiety i kody jak kompletne wejście generatora
    generator = gan.generator(inputs,
                              image_size,
                              labels=labels,
                              codes=[code1, code2])
    generator.summary()

    # konstruowanie modelu sieci współzawodniczącej = generator+dyskryminator
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    discriminator.trainable = False
    # komplet na wejścia = kod szumu, etykiety, kody
    inputs = [inputs, labels, code1, code2]
    adversarial = Model(inputs,
                        discriminator(generator(inputs)),
                        name=model_name)
    # ta sama strata co w dyskryminatorze
    adversarial.compile(loss=loss,
                        loss_weights=loss_weights,
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()

    # uczenie dyskryminatora i sieci współzawodniczącej
    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)
    params = (batch_size, 
              latent_size, 
              train_steps, 
              num_labels, 
              model_name)
    train(models, data, params)


def test_generator(generator, params, latent_size=100):
    label, code1, code2, p1, p2 = params
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    step = 0
    if label is None:
        num_labels = 10
        noise_label = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_label = np.zeros((16, 10))
        noise_label[:,label] = 1
        step = label

    code_std = 2
    if code1 is None:
        noise_code1 = np.random.normal(scale=0.5, size=[16, 1])
    else:
        if p1:
            a = np.linspace(-code_std, code_std, 16)
            a = np.reshape(a, [16, 1])
            noise_code1 = np.ones((16, 1)) * a
        else:
            noise_code1 = np.ones((16, 1)) * code1
        print(noise_code1)

    if code2 is None:
        noise_code2 = np.random.normal(scale=0.5, size=[16, 1])
    else:
        if p2:
            a = np.linspace(-code_std, code_std, 16)
            a = np.reshape(a, [16, 1])
            noise_code2 = np.ones((16, 1)) * a
        else:
            noise_code2 = np.ones((16, 1)) * code2
        print(noise_code2)

    gan.plot_images(generator,
                    noise_input=noise_input,
                    noise_label=noise_label,
                    noise_codes=[noise_code1, noise_code2],
                    show=True,
                    step=step,
                    model_name="test_outputs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Ładowanie modelu generatora z wytrenowanymi wagami z pliku h5"
    parser.add_argument("-g", "--generator", help=help_)
    help_ = "Podaj cyfrę którą chcesz generować"
    parser.add_argument("-d", "--digit", type=int, help=help_)
    help_ = "Podaj pierwszy kod niejawny"
    parser.add_argument("-a", "--code1", type=float, help=help_)
    help_ = "Podaj drugi kod niejawny"
    parser.add_argument("-b", "--code2", type=float, help=help_)
    help_ = "Wydruk cyfr przy kodzie pierwszym w zakresie od -n1 do +n2"
    parser.add_argument("--p1", action='store_true', help=help_)
    help_ = "Wydruk cyfr przy kodzie drugim w zakresie od -n1 do +n2"
    parser.add_argument("--p2", action='store_true', help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        label = args.digit
        code1 = args.code1
        code2 = args.code2
        p1 = args.p1
        p2 = args.p2
        params = (label, code1, code2, p1, p2)
        test_generator(generator, params, latent_size=62)
    else:
        build_and_train_models(latent_size=62)
