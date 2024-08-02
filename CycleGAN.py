"""Konstruowanie i uczenie sieci CycleGAN

Sieć CycleGAN jest siecią międzydomenową. Podobnie jak ma to miejsce w przypadku innych sieci GAN może być uczona w sposób nienadzorowany.

CycleGAN jest złożona z dwóch generatorów (G oraz F) i dwóch dyskryminatorów.
Każdy z generatorów jest siecią typu U. Dyskryminator jest typową siecią dekodera z opcją używania struktury PatchGAN.

Mamy dwa zbiory danych: x = źródłowy, y = docelowy. 
Podczas cyklu w przód realizowane jest odwzorowanie x'= F(y') = F(G(x)), gdzie y' jest przewidywanym wyjściem w domenie y, oraz x' jest zrekonstruowanym wejściem.
Docelowy dyskryminator decyduje czy y' jest prawdziwy czy fałszywy.
Celem generatora G cyklu w przód jest nauczenie się jak zmylić docelowy generator tak, by uznał że y' jest prawdziwy. 

Cykl wstecz poprawia wydajność sieci CycleGAN wykonując proces odwrotny do cyklu w przód. Realizuje odwzorowanie y' = G(x') = G(F(y)) gdzie x' jest przewidywanym wyjściem w domenie x. Źródłowy dyskryminator decyduje czy x' jest prawdziwy czy fałszywy.
Celem generatora F w cyklu wstecz jest nauczenie się jak zmylić docelowy dyskryminator by uwierzył, że x' jest prawdziwy.

References:
[1]Zhu, Jun-Yan, et al. "Unpaired Image-to-Image Translation Using
Cycle-Consistent Adversarial Networks." 2017 IEEE International
Conference on Computer Vision (ICCV). IEEE, 2017.

[2]Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net:
Convolutional networks for biomedical image segmentation."
International Conference on Medical image computing and
computer-assisted intervention. Springer, Cham, 2015.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import concatenate
from keras.optimizers import RMSprop
from keras.models import Model
from keras.models import load_model

# from keras_contrib.layers.normalization import InstanceNormalization
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# install: pip install tensorflow-addons
# from tensorflow_addons.layers import InstanceNormalization
from keras.layers import InstanceNormalization  # wrong
# import InstanceNormalization
import numpy as np
import argparse
import cifar10_utils
import mnist_svhn_utils
import other_utils
import datetime


def encoder_layer(inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2,
                  activation='relu',
                  instance_norm=True):
    """Konstruowanie warstwy ogólnego modelu kodera z Conv2D-IN-LeakyReLU
    IN jest opcjonalne, LeakyReLU można zastąpić ReLU
"""


    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same')

    x = inputs
    if instance_norm:
        x = InstanceNormalization()(x)
    if activation == 'relu':
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    return x


def decoder_layer(inputs,
                  paired_inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2,
                  activation='relu',
                  instance_norm=True):
    """Konstruowanie warstwy ogólnego dekodera z Conv2D-IN-LeakyReLU
    IN jest opcjonalne, LeakyReLU można zastąpić ReLU
    
    Argumenty: (część)
    inputs (tensor): wejście warstwy dekodera
    paired_inputs (tensor): wejście warstwy kodera zapewniane przez połączenia skrótowe sieci U i połączone na wejście

    """


    conv = Conv2DTranspose(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding='same')

    x = inputs
    if instance_norm:
        x = InstanceNormalization()(x)
    if activation == 'relu':
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    x = concatenate([x, paired_inputs])
    return x


def build_generator(input_shape,
                    output_shape=None,
                    kernel_size=3,
                    name=None):
    """Generator jest siecią U złożoną z 4-warstwowego kodera i 4-warstwowego dekodera. 
        Warstwa n-i jest połączona z warstwą i.

    Argumenty:
    input_shape (tuple): kształt danych wejściowych 
    output_shape (tuple): kształt danych wyjściowych 
    kernel_size (int): rozmiar jądra warstw kodera i dekodera 
    name (string): nazwa przypisana do modelu generatora

    Zwraca:
    generator (Model):

    """


    inputs = Input(shape=input_shape)
    channels = int(output_shape[-1])
    e1 = encoder_layer(inputs,
                       32,
                       kernel_size=kernel_size,
                       activation='leaky_relu',
                       strides=1)
    e2 = encoder_layer(e1,
                       64,
                       activation='leaky_relu',
                       kernel_size=kernel_size)
    e3 = encoder_layer(e2,
                       128,
                       activation='leaky_relu',
                       kernel_size=kernel_size)
    e4 = encoder_layer(e3,
                       256,
                       activation='leaky_relu',
                       kernel_size=kernel_size)

    d1 = decoder_layer(e4,
                       e3,
                       128,
                       kernel_size=kernel_size)
    d2 = decoder_layer(d1,
                       e2,
                       64,
                       kernel_size=kernel_size)
    d3 = decoder_layer(d2,
                       e1,
                       32,
                       kernel_size=kernel_size)
    outputs = Conv2DTranspose(channels,
                              kernel_size=kernel_size,
                              strides=1,
                              activation='sigmoid',
                              padding='same')(d3)

    generator = Model(inputs, outputs, name=name)

    return generator


def build_discriminator(input_shape,
                        kernel_size=3,
                        patchgan=True,
                        name=None):
    """ Dyskryminator jest 4-warstwowym koderem, którego wyjściem jest jednowymiarowa 
    lub n x n-wymiarowa) łatka prawdopodobieństw, że dane wejściowe są prawdziwe 

    Argumenty:
    input_shape (tuple): kształt danych wejściowych
    kernel_size (int): rozmiar jądra warstw dekodera
    patchgan (bool): czy wyjście jest łatką, czy tylko jednowymiarowe
    name (string): nazwa przypisana do modelu dyskryminatora

    Zwraca:
    discriminator (Model):

    """


    inputs = Input(shape=input_shape)
    x = encoder_layer(inputs,
                      32,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    x = encoder_layer(x,
                      64,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    x = encoder_layer(x,
                      128,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    x = encoder_layer(x,
                      256,
                      kernel_size=kernel_size,
                      strides=1,
                      activation='leaky_relu',
                      instance_norm=False)

    # Jeśli patchgan = True, użyj prawdopodobieństw wyjść n x n,
    # w innym przypadku użyj prawdopodobieństwa jednowymiarowego.

    if patchgan:
        x = LeakyReLU(alpha=0.2)(x)
        outputs = Conv2D(1,
                         kernel_size=kernel_size,
                         strides=2,
                         padding='same')(x)
    else:
        x = Flatten()(x)
        x = Dense(1)(x)
        outputs = Activation('linear')(x)


    discriminator = Model(inputs, outputs, name=name)

    return discriminator


def train_cyclegan(models,
                   data,
                   params,
                   test_params, 
                   test_generator):
    """ Wytrenuj CycleGAN
    1) Wytrenuj docelowy dyskryminator
    2) Wytrenuj źródłowy dyskryminator
    3) Wytrenuj cykle w przód i wstecz sieci współzawodniczącej

    Argumenty:
    models (Models): docelowy/źródłowy dyskryminator/generator, model sieci współzawodniczącej
    data (tuple): dane uczące źródłowe i docelowe
    params (tuple): parametry sieci
    test_params (tuple): parametry testowe
    test_generator (function): używane do generowania przewidywanych obrazów źródłowych i docelowych
    """

    # modele
    g_source, g_target, d_source, d_target, adv = models
    # parametry sieci
    batch_size, train_steps, patch, model_name = params
    # zbiór uczący
    source_data, target_data, test_source_data, test_target_data\
            = data

    titles, dirs = test_params

    # obrazy z generatora są zapisywane co 2000 kroków
    save_interval = 2000
    target_size = target_data.shape[0]
    source_size = source_data.shape[0]

    # czy jest używany patchgan, czy nie 
    if patch > 1:
        d_patch = (patch, patch, 1)
        valid = np.ones((batch_size,) + d_patch)
        fake = np.zeros((batch_size,) + d_patch)
    else:
        valid = np.ones([batch_size, 1])
        fake = np.zeros([batch_size, 1])

    valid_fake = np.concatenate((valid, fake))
    start_time = datetime.datetime.now()

    for step in range(train_steps):
        # próbka partii prawdziwych docelowych danych
        rand_indexes = np.random.randint(0, 
                                         target_size,
                                         size=batch_size)
        real_target = target_data[rand_indexes]

        # próbka partii prawdziwych źródłowych danych
        rand_indexes = np.random.randint(0, 
                                         source_size,
                                         size=batch_size)
        real_source = source_data[rand_indexes]
        # generowanie próbki fałszywych docelowych danych z prawdziwych danych źródłowych
        fake_target = g_target.predict(real_source)
        
        # połączenie prawdziwych i fałszywych w jedną partię
        x = np.concatenate((real_target, fake_target))
        # uczenie docelowego dyskryminatora z użyciem prawdziwych/fałszywych danych
        metrics = d_target.train_on_batch(x, valid_fake)
        log = "%d: [d_target loss: %f]" % (step, metrics[0])

        # generowanie partii fałszywych danych źródłowych z prawdziwych danych docelowych
        fake_source = g_source.predict(real_target)
        x = np.concatenate((real_source, fake_source))
        # uczenie źródłowego dyskryminatora z użyciem prawdziwych/fałszywych danych
        metrics = d_source.train_on_batch(x, valid_fake)
        log = "%s [d_source loss: %f]" % (log, metrics[0])

        # uczenie sieci współzawodniczącej przy użyciu cykli w przód i wstecz.
        # Wygenerowane dane źródłowe i docelowe próbują zmylić dyskryminator
        x = [real_source, real_target]
        y = [valid, valid, fake_source, fake_target]
        metrics = adv.train_on_batch(x, y)
        elapsed_time = datetime.datetime.now() - start_time
        fmt = "%s [adv loss: %f] [time: %s]"
        log = fmt % (log, metrics[0], elapsed_time)
        print(log)
        if (step + 1) % save_interval == 0:
            test_generator((g_source, g_target),
                           (test_source_data, test_target_data),
                           step=step+1,
                           titles=titles,
                           dirs=dirs,
                           show=False)

    # zapisanie modeli po wytrenowaniu generatorów
    g_source.save(model_name + "-g_source.h5")
    g_target.save(model_name + "-g_target.h5")


def build_cyclegan(shapes,
                   source_name='source',
                   target_name='target',
                   kernel_size=3,
                   patchgan=False,
                   identity=False
                   ):
    """Konstruowanie CycleGAN
    1) konstruuj dyskryminator źródłowy i docelowy
    2) konstruuj generator źródłowy i docelowy
    3) konstruuj sieć współzawodniczącą

    Argumenty:
    shapes (tuple): wymiary danych źródłowych i docelowych
    source_name (string): łańcuch znaków do dodania do modeli generatora/dyskryminatora
    target_name (string): łańcuch znaków do dodania do modeli generatora/dyskryminatora
    kernel_size (int): rozmiar jądra dla kodera/dekodera lub dyskryminatora/generatora modelu
    patchgan (bool): czy używać patchgan na dyskryminatorze
    identity (bool): czy używać funkcji straty tożsamości

    Zwraca:
    (lista): 2 generatory, 2 dyskryminatory oraz 1 model sieci współzawodniczącej 

    """


    source_shape, target_shape = shapes
    lr = 2e-4
    decay = 6e-8
    gt_name = "gen_" + target_name
    gs_name = "gen_" + source_name
    dt_name = "dis_" + target_name
    ds_name = "dis_" + source_name

    # konstruowanie generatorów: źródłowego i docelowego
    g_target = build_generator(source_shape,
                               target_shape,
                               kernel_size=kernel_size,
                               name=gt_name)
    g_source = build_generator(target_shape,
                               source_shape,
                               kernel_size=kernel_size,
                               name=gs_name)
    print('---- GENERATOR DOCELOWY  ----')
    g_target.summary()
    print('---- GENERATOR ZRODLOWY  ----')
    g_source.summary()

    # konstruowanie dyskryminatorów: źródłowego i docelowego
    d_target = build_discriminator(target_shape,
                                   patchgan=patchgan,
                                   kernel_size=kernel_size,
                                   name=dt_name)
    d_source = build_discriminator(source_shape,
                                   patchgan=patchgan,
                                   kernel_size=kernel_size,
                                   name=ds_name)
    print('---- DYSKRYMINATOR DOCELOWY  ----')
    d_target.summary()
    print('---- DYSKRYMINATOR ZRODLOWY  ----')
    d_source.summary()

    optimizer = RMSprop(lr=lr, decay=decay)
    d_target.compile(loss='mse',
                     optimizer=optimizer,
                     metrics=['accuracy'])
    d_source.compile(loss='mse',
                     optimizer=optimizer,
                     metrics=['accuracy'])

    d_target.trainable = False
    d_source.trainable = False

    # konstruowanie grafu obliczeniowego dla modelu sieci współzawodniczącej
    # sieć cyklu w przód i docelowy dyskryminator
    source_input = Input(shape=source_shape)
    fake_target = g_target(source_input)
    preal_target = d_target(fake_target)
    reco_source = g_source(fake_target)

    # sieć cyklu wstecz i dyskryminator źródłowy
    target_input = Input(shape=target_shape)
    fake_source = g_source(target_input)
    preal_source = d_source(fake_source)
    reco_target = g_target(fake_source)

    # Jeśli używamy funkcji starty dla tożsamości, dodaj 2 dodatkowe człony i wyjścia.
    if identity:
        iden_source = g_source(source_input)
        iden_target = g_target(target_input)
        loss = ['mse', 'mse', 'mae', 'mae', 'mae', 'mae']
        loss_weights = [1., 1., 10., 10., 0.5, 0.5]
        inputs = [source_input, target_input]
        outputs = [preal_source,
                   preal_target,
                   reco_source,
                   reco_target,
                   iden_source,
                   iden_target]
    else:
        loss = ['mse', 'mse', 'mae', 'mae']
        loss_weights = [1., 1., 10., 10.]
        inputs = [source_input, target_input]
        outputs = [preal_source,
                   preal_target,
                   reco_source,
                   reco_target]

    # konstruowanie modelu współzawodniczącego
    adv = Model(inputs, outputs, name='adversarial')
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    adv.compile(loss=loss,
                loss_weights=loss_weights,
                optimizer=optimizer,
                metrics=['accuracy'])
    print('---- ADVERSARIAL NETWORK ----')
    adv.summary()

    return g_source, g_target, d_source, d_target, adv


def graycifar10_cross_colorcifar10(g_models=None):
    """Buduje i przeprowadza trening sieci CycleGAN, która wykonuje operację
      obraz w skali szarości <--> obraz kolorowy ze zbioru cifar10
    """

    model_name = 'cyclegan_cifar10'
    batch_size = 32
    train_steps = 100000
    patchgan = True
    kernel_size = 3
    postfix = ('%dp' % kernel_size) \
            if patchgan else ('%d' % kernel_size)

    data, shapes = cifar10_utils.load_data()
    source_data, _, test_source_data, test_target_data = data
    titles = ('CIFAR10 predicted source images.',
              'CIFAR10 predicted target images.',
              'CIFAR10 reconstructed source images.',
              'CIFAR10 reconstructed target images.')
    dirs = ('cifar10_source-%s' % postfix, \
            'cifar10_target-%s' % postfix)

    # generowanie przewidywanych docelowych (kolorowe) i źródłowych (skala szarości) obrazów
    if g_models is not None:
        g_source, g_target = g_models
        other_utils.test_generator((g_source, g_target),
                                   (test_source_data, \
                                           test_target_data),
                                   step=0,
                                   titles=titles,
                                   dirs=dirs,
                                   show=True)
        return

    # budowanie sieci CycleGAN do kolorowania obrazów cifar10
    models = build_cyclegan(shapes,
                            "gray-%s" % postfix,
                            "color-%s" % postfix,
                            kernel_size=kernel_size,
                            patchgan=patchgan)
    # Rozmiar łatki jest podzielny przez 2^n ze względu na to, że wejście jest próbkowane w dół (ang. downsampled)
    # w dyskryminatorze co 2^n (tzn. używamy n-krotnie kroku 2 [strides = 2]).
    patch = int(source_data.shape[1] / 2**4) if patchgan else 1
    params = (batch_size, train_steps, patch, model_name)
    test_params = (titles, dirs)
    # uczenie CycleGAN
    train_cyclegan(models,
                   data,
                   params,
                   test_params,
                   other_utils.test_generator)


def mnist_cross_svhn(g_models=None):
    """Skonstruuj i wytrenuj sieć CycleGAN dokonującą konwersji mnist <--> svhn
    """


    model_name = 'cyclegan_mnist_svhn'
    batch_size = 32
    train_steps = 100000
    patchgan = True
    kernel_size = 5
    postfix = ('%dp' % kernel_size) \
            if patchgan else ('%d' % kernel_size)

    data, shapes = mnist_svhn_utils.load_data()
    source_data, _, test_source_data, test_target_data = data
    titles = ('MNIST predicted source images.',
              'SVHN predicted target images.',
              'MNIST reconstructed source images.',
              'SVHN reconstructed target images.')
    dirs = ('mnist_source-%s' \
            % postfix, 'svhn_target-%s' % postfix)

    # generowanie docelowych (svhn) i źródłowych (mnist) przewidywanych obrazów 
    if g_models is not None:
        g_source, g_target = g_models
        other_utils.test_generator((g_source, g_target),
                                   (test_source_data, \
                                           test_target_data),
                                   step=0,
                                   titles=titles,
                                   dirs=dirs,
                                   show=True)
        return

    # konstruowanie cyclegan do krzyżówki mnist oraz svhn
    models = build_cyclegan(shapes,
                            "mnist-%s" % postfix,
                            "svhn-%s" % postfix,
                            kernel_size=kernel_size,
                            patchgan=patchgan)
    # Rozmiar łatki jest podzielny przez 2^n ze względu na to,
    # że wejście jest próbkowane w dół (ang. downsampled)
    # w dyskryminatorze co 2^n (tzn. używamy n-krotnie kroku 2 [strides = 2]).
    patch = int(source_data.shape[1] / 2**4) if patchgan else 1
    params = (batch_size, train_steps, patch, model_name)
    test_params = (titles, dirs)
    # uczenie cyclegan
    train_cyclegan(models,
                   data,
                   params,
                   test_params,
                   other_utils.test_generator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Zaladuj model generatora zrodlowego cifar10 z pliku h5"
    parser.add_argument("--cifar10_g_source", help=help_)
    help_ = "Zaladuj model generatora docelowego cifar10 z pliku h5"
    parser.add_argument("--cifar10_g_target", help=help_)

    help_ = "Zaladuj model generatora zrodlowego mnist_svhn z pliku h5"
    parser.add_argument("--mnist_svhn_g_source", help=help_)
    help_ = "Zaladuj model generatora docelowego mnist_svhn z pliku h5"
    parser.add_argument("--mnist_svhn_g_target", help=help_)

    help_ = "Uruchom uczenie kolorowania cifar10"
    parser.add_argument("-c",
                        "--cifar10",
                        action='store_true',
                        help=help_)
    help_ = "Wytrenuj miedzydomenową (mnist-svhn) siec cyclegan"
    parser.add_argument("-m",
                        "--mnist-svhn",
                        action='store_true',
                        help=help_)
    args = parser.parse_args()

    # załadowanie wstępnie wytrenowanych generatorów: źródłowego i docelowego 
    # dla cifar10
    if args.cifar10_g_source:
        g_source = load_model(args.cifar10_g_source)
        if args.cifar10_g_target:
            g_target = load_model(args.cifar10_g_target)
            g_models = (g_source, g_target)
            graycifar10_cross_colorcifar10(g_models)
    # załadowanie wstępnie wytrenowanych generatorów: źródłowego i docelowego 
    # dla mnist-svhn
    elif args.mnist_svhn_g_source:
        g_source = load_model(args.mnist_svhn_g_source)
        if args.mnist_svhn_g_target:
            g_target = load_model(args.mnist_svhn_g_target)
            g_models = (g_source, g_target)
            mnist_cross_svhn(g_models)
    # uczenie sieci CycleGAN dla cifar10
    elif args.cifar10:
        graycifar10_cross_colorcifar10()
    # uczenie sieci CycleGAN dla mnist-svhn 
    else:
        mnist_cross_svhn()
