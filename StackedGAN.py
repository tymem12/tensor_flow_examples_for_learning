'''Uczenie sieci StackedGAN na zbiorze danych MNIST z użyciem Keras

W sieci StackedGAN używamy Kodera, Generatora i Dyskryminatora.
Koder jest klasyfikatorem cyfr MNIST korzystającym z sieci CNN.
Koder dostarcza cechy niejawne (feature1) oraz etykiety których uczy się generator przez odwrócenie procesu.
Generator wykorzystuje etykiety warunkujące i kody niejawne (z0 oraz z1) by syntetyzować obrazy oszukując dyskryminator.
Etykiety, kody z0 oraz z1 są kodami rozplątanymi używanymi do kontrolowania atrybutów syntetyzowanych obrazów. Dyskryminator decyduje o tym, czy obraz i cechy feature1 są prawdziwe czy fałszywe. Równocześnie dokonuje szacowania kodów niejawnych które są użyte do generowania obrazu i cech feature1.

[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

[2] Huang, Xun, et al. "Stacked generative adversarial networks." 
IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 
Vol. 2. 2017.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K
from keras.layers import concatenate

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse

import sys
sys.path.append("..")
import gan

def build_encoder(inputs, num_labels=10, feature1_dim=256):
    """ Konstruowanie modelu podsieci klasyfikatora (kodera)

    Dwie podsieci: 
    1) Koder0: obraz na feature1 (pośrednia cecha niejawna)
    2) Koder1: feature1 na etykiety

    # Argumenty
        inputs (Layers): x — obrazy, feature1 — 
            warstwa wyjściowa feature1 (cechy1)
        num_labels (int): liczba klas etykiet
        feature1_dim (int): wymiarowość feature1

    # Zwraca
        enc0, enc1 (Models): opis poniżej 
    """
    kernel_size = 3
    filters = 64

    x, feature1 = inputs
   # Koder0 lub enc0
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu')(x)
    y = MaxPooling2D()(y)
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu')(y)
    y = MaxPooling2D()(y)
    y = Flatten()(y)
    feature1_output = Dense(feature1_dim, activation='relu')(y)
    # Koder0 lub enc0: obraz (x lub feature0) na feature1  
    enc0 = Model(inputs=x, outputs=feature1_output, name="encoder0")
    
    # Koder1 lub enc1
    y = Dense(num_labels)(feature1)
    labels = Activation('softmax')(y)
    # Encoder1 or enc1: feature1 to class labels (feature2)
    enc1 = Model(inputs=feature1, outputs=labels, name="encoder1")

    # zwraca zarówno enc0, jak i enc1
    return enc0, enc1


def build_generator(latent_codes, image_size, feature1_dim=256):
    """Konstruowanie modelu podsieci generatora

    Dwie podsieci: 1) klasa i szum dla feature1 (cecha pośrednia)
              2) feature1 na obraz

    # Argumenty
        latent_codes (Layers): kod dyskretny (etykiety),
            szum i cechy feature1
        image_size (int): docelowy rozmiar jednego boku obrazu 
            (zakładamy, że jest kwadratem)
        feature1_dim (int): wymiarowość feature1 

    # Zwraca
        gen0, gen1 (Models): opis poniżej
    """

    # kody niejawne i parametry sieci
    labels, z0, z1, feature1 = latent_codes
    # image_resize = image_size // 4
    # kernel_size = 5
    # layer_filters = [128, 64, 32, 1]

    # wejścia gen1
    inputs = [labels, z1]      # 10 + 50 = 62 wymiary
    x = concatenate(inputs, axis=1)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    fake_feature1 = Dense(feature1_dim, activation='relu')(x)
    # gen1: klasy i szum (feature2+z1) na feature1
    gen1 = Model(inputs, fake_feature1, name='gen1')

     # gen0: feature1+z0 na feature0 (obraz)
    gen0 = gan.generator(feature1, image_size, codes=z0)

    return gen0, gen1


def build_discriminator(inputs, z_dim=50):
    """Konstruuje model dyskryminatora 1

    Klasyfikuje feature1 (features) jako obraz prawdziwy/fałszywy i odzyskuje
    szum wejścia lub kod niejawny (przez minimalizację strat entropii)

    # Argumenty
        inputs (Layer): feature1
        z_dim (int): wymiarowość szumu

    # Zwraca
        dis1 (Model): feature1 jako prawdziwe/fałszywe i odzyskany kod niejawny
    """

    # wejście to feature1 o 256 wymiarach

    x = Dense(256, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)

    # pierwsze wyjście jest prawdopodobieństwem, że feature1 jest prawdziwa
    f1_source = Dense(1)(x)
    f1_source = Activation('sigmoid',
                           name='feature1_source')(f1_source)

    # rekonstrukcja z1 (sieć Q1)
    z1_recon = Dense(z_dim)(x) 
    z1_recon = Activation('tanh', name='z1')(z1_recon)
    
    discriminator_outputs = [f1_source, z1_recon]
    dis1 = Model(inputs, discriminator_outputs, name='dis1')
    return dis1


def train(models, data, params):
    """Uczenie dyskryminatora i sieci współzawodniczących

    Naprzemienne uczenie dyskryminatora i sieci współzawodniczącej na partiach. 
    Najpierw uczony jest dyskryminator z prawdziwymi i fałszywymi obrazami, 
    odpowiadającymi im etykietami OH i kodami niejawnymi. 
    Następnie uczony jest oponent z obrazami fałszywymi udającymi prawdziwe, 
    z odpowiadającymi im etykietami OH i kodami niejawnymi.
    Generowanie przykładowych obrazów co save_interval.

    # Argumenty
        models (Models): koder, generator, dyskryminator, modele sieci współzawodniczącej 
        data (tuple):dane uczące x_train, y_train
        params (tuple): parametry sieci

    """
    # modele sieci StackedGAN i koderów

    enc0, enc1, gen0, gen1, dis0, dis1, adv0, adv1 = models
    # parametry sieci
    batch_size, train_steps, num_labels, z_dim, model_name = params
    # zbiór uczący
    (x_train, y_train), (_, _) = data
    # Obrazy z generatora są zapisywane co 500 kroków
    save_interval = 500

    # kody etykiet i szumu do testowania generatora
    z0 = np.random.normal(scale=0.5, size=[16, z_dim])
    z1 = np.random.normal(scale=0.5, size=[16, z_dim])
    noise_class = np.eye(num_labels)[np.arange(0, 16) % num_labels]
    noise_params = [noise_class, z0, z1]
    # liczba elementów w zbiorze uczącym
    train_size = x_train.shape[0]
    print(model_name,
          "Labels for generated images: ",
          np.argmax(noise_class, axis=1))

    for i in range(train_steps):
        # uczenie dyskryminatora1 na 1 partii
        # 1 partia prawdziwych (etykieta = 1.0) i fałszywych (etykieta = 0.0) feature1 (cecha1)
        # losowy wybór prawdziwych obrazów ze zbioru danych
        rand_indexes = np.random.randint(0, 
                                         train_size, 
                                         size=batch_size)
        real_images = x_train[rand_indexes]
        # prawdziwa feature1 (cecha 1) wyjścia koder0
        real_feature1 = enc0.predict(real_images)
        # generowanie losowego 50-wymiarowego kodu niejawnego z1
        real_z1 = np.random.normal(scale=0.5,
                                   size=[batch_size, z_dim])
        # prawdziwe etykiety ze zbioru danych
        real_labels = y_train[rand_indexes]

        # generowanie fałszywej feature1 (cechy1) przez generator1 na
        # podstawie prawdziwych etykiet i 50-wymiarowego kodu niejawnego z1
        fake_z1 = np.random.normal(scale=0.5,
                                   size=[batch_size, z_dim])
        fake_feature1 = gen1.predict([real_labels, fake_z1])

        # dane prawdziwe + fałszywe
        feature1 = np.concatenate((real_feature1, fake_feature1))
        z1 = np.concatenate((fake_z1, fake_z1))

        # etykieta pierwszej połowy jako prawdziwa i drugiej połowy jako fałszywa
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0

        # uczenie dyskryminatora1 do klasyfikowania feature1
        # jako prawdziwe/fałszywe i odzyskanie kodu niejawnego (z1) 
        # prawdziwe = z kodera1, fałszywe = z geneneratora1 
        # łączne uczenie przy użyciu straty sieci współzawodniczącej1 i entropii1 z dyskryminatora
        metrics = dis1.train_on_batch(feature1, [y, z1])
        # zapisujemy tylko stratę całkowitą
        log = "%d: [dis1_loss: %f]" % (i, metrics[0])

         
        # uczenie dyskryminatora0 na 1 partii
        # 1 partia prawdziwych (etykieta = 1.0) i fałszywych (etykieta = 0.0) obrazów 
        # generowanie losowego, 50-wymiarowego kodu niejawnego z0
        fake_z0 = np.random.normal(scale=0.5, size=[batch_size, z_dim])
        # generowanie fałszywych obrazów z prawdziwej cechy1 (feature1) i fałszywego z0
        fake_images = gen0.predict([real_feature1, fake_z0])
       
        # dane prawdziwe+fałszywe
        x = np.concatenate((real_images, fake_images))
        z0 = np.concatenate((fake_z0, fake_z0))

        # uczenie dyskryminatora0 do klasyfikowania obrazów 
        # jako prawdziwe/fałszywe i odzyskania kodu niejawnego (z0)
        # łączne uczenie (strata sieci współzawodniczącej0 i entropia0) dyskryminatora
        metrics = dis0.train_on_batch(x, [y, z0])
        # zapisujemy tylko stratę całkowitą (używając dis0.metrics_names)
        log = "%s [dis0_loss: %f]" % (log, metrics[0])

        # uczenie sieci współzawodniczącej 
        # generowanie fałszywego z1, etykiet
        fake_z1 = np.random.normal(scale=0.5, 
                                   size=[batch_size, z_dim])
        # dane wejściowe generatora1 są próbkami z prawdziwych etykiet
        # i 50-wymiarowego kodu niejawnego
        gen1_inputs = [real_labels, fake_z1]

        # oznacz fałszywą feature1 jako prawdziwą
        y = np.ones([batch_size, 1])
    
        # uczenie generatora1 (przez sieć współzawodniczącą) przez oszukiwanie dyskryminatora
        # i przybliżenie generatora feature1 (cechy1) kodera1
        # łączne uczenie: współzawodnicząca1, entropia1, warunkowa1
        metrics = adv1.train_on_batch(gen1_inputs,
                                      [y, fake_z1, real_labels])
        fmt = "%s [adv1_loss: %f, enc1_acc: %f]"
        # zapisywanie całkowitej straty i dokładności klasyfikacji
        log = fmt % (log, metrics[0], metrics[6])

        # wejście na generator0 jest prawdziwą feature1
        # i 50-wymiarowym kodem niejawnym z0
        fake_z0 = np.random.normal(scale=0.5,
                                   size=[batch_size, z_dim])
        gen0_inputs = [real_feature1, fake_z0]

        # uczenie generator0 (przez współzawodniczącą) przez oszukiwanie 
        # dyskryminatora i przybliżanie obrazów koder1 
        # łączne uczenie generatora źródłowego: 
        # współzawodnicząca0, entropia0, warunkowa0
        metrics = adv0.train_on_batch(gen0_inputs,
                                      [y, fake_z0, real_feature1])
        # zapisujemy tylko całkowitą stratę
        log = "%s [adv0_loss: %f]" % (log, metrics[0])

        print(log)
        if (i + 1) % save_interval == 0:
            generators = (gen0, gen1)
            plot_images(generators,
                        noise_params=noise_params,
                        show=False,
                        step=(i + 1),
                        model_name=model_name)

    # zapis modeli po treningu generator0 & 1
    # wytrenowany generator może być ponownie wykorzystany
    # w przyszłości do generowania cyfr MNIST
    gen1.save(model_name + "-gen1.h5")
    gen0.save(model_name + "-gen0.h5")
    

def plot_images(generators,noise_params,show=False,step=0,model_name="gan"):
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
    gen0, gen1 = generators
    noise_class, z0, z1 = noise_params
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    feature1 = gen1.predict([noise_class, z1])
    images = gen0.predict([feature1, z0])
    print(model_name,
          "Etykiety dla generowanych obrazów: ",
          np.argmax(noise_class, axis=1))

    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_class.shape[0]))
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


def train_encoder(model,
                  data, 
                  model_name="stackedgan_mnist", 
                  batch_size=64):
    """ Train the Encoder Model (enc0 and enc1)

    # Arguments
        model (Model): Encoder
        data (tensor): Train and test data
        model_name (string): model name
        batch_size (int): Train batch size
    """

    (x_train, y_train), (x_test, y_test) = data
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train,
              y_train,
              validation_data=(x_test, y_test),
              epochs=10,
              batch_size=batch_size)

    model.save(model_name + "-encoder.h5")
    score = model.evaluate(x_test,
                           y_test, 
                           batch_size=batch_size,
                           verbose=0)
    print("\nDokładnosc na zbiorze testowym: %.1f%%" % (100.0 * score[1]))


def build_and_train_models():
    """Załadowanie zbioru, skonstruowanie dyskryminatora,
    generatora i sieci współzawodniczącej StackedGAN
    Wywołanie procedury uczącej StackedGAN.
    """

    # załadowanie zbioru MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # zmiana wymiarów i normalizacja obrazów
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255 

    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_test = x_test.astype('float32') / 255

    # liczba etykiet
    num_labels = len(np.unique(y_train))
    # na wektor OH
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model_name = "stackedgan_mnist"
    # parametry sieci
    batch_size = 64
    train_steps = 10000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    label_shape = (num_labels, )
    z_dim = 50
    z_shape = (z_dim, )
    feature1_dim = 256
    feature1_shape = (feature1_dim, )

    # konstruowanie dyskryminatora0 i modeli sieci Q0
    inputs = Input(shape=input_shape, name='discriminator0_input')
    dis0 = gan.discriminator(inputs, num_codes=z_dim)
    # W [1] użyto Adam, ale dyskryminator łatwiej osiąga zbieżność z RMSprop
    optimizer = RMSprop(lr=lr, decay=decay)
    # funkcje straty: 1) prawdopodobieństwo, że obraz jest prawdziwy (strata współzawodniczącej0)
    # 2) strata rekonstrukcji MSE z0 (strata sieci Q0 lub entropii0)
    loss = ['binary_crossentropy', 'mse']
    loss_weights = [1.0, 10.0] 
    dis0.compile(loss=loss,
                 loss_weights=loss_weights,
                 optimizer=optimizer,
                 metrics=['accuracy'])
    dis0.summary() # dyskryminator obrazu, estymator z0 

    # konstruowanie dyskryminatora1 i modeli sieci Q1
    input_shape = (feature1_dim, )
    inputs = Input(shape=input_shape, name='discriminator1_input')
    dis1 = build_discriminator(inputs, z_dim=z_dim )
    # funkcje straty: 1) prawdopodobieństwo że obraz jest prawdziwy (strata współzawodniczącej1)
    # 2) strata rekonstrukcji MSE z1 (strata sieci Q1 lub entropii1)
    loss = ['binary_crossentropy', 'mse']
    loss_weights = [1.0, 1.0] 
    dis1.compile(loss=loss,
                 loss_weights=loss_weights,
                 optimizer=optimizer,
                 metrics=['accuracy'])
    dis1.summary() # dyskryminator feature1 (cecha1), estymator z1 

    # tworzenie modeli generatorów
    feature1 = Input(shape=feature1_shape, name='feature1_input')
    labels = Input(shape=label_shape, name='labels')
    z1 = Input(shape=z_shape, name="z1_input")
    z0 = Input(shape=z_shape, name="z0_input")
    latent_codes = (labels, z0, z1, feature1)
    gen0, gen1 = build_generator(latent_codes, image_size)
    gen0.summary() # generator obrazu
    gen1.summary() # generator feature1 (cecha1)

    # konstruowanie modeli koderów
    input_shape = (image_size, image_size, 1)
    inputs = Input(shape=input_shape, name='encoder_input')
    enc0, enc1 = build_encoder((inputs, feature1), num_labels)
    enc0.summary() # koder obraz na feature1 (cecha1)
    enc1.summary() # feature1 (cecha1) na etykiety kodera (klasyfikator)
    encoder = Model(inputs, enc1(enc0(inputs)))
    encoder.summary() # obraz na etykiety kodera (klasyfikator)

    data = (x_train, y_train), (x_test, y_test)
    train_encoder(encoder, data, model_name=model_name)

    # konstruowanie modelu współzawodnicząca0 =
    # generator0+dyskryminator0+koder0
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    # zamrożenie wag koder0
    enc0.trainable = False
    # zamrożenie wag dyskryminator0
    dis0.trainable = False
    gen0_inputs = [feature1, z0]
    gen0_outputs = gen0(gen0_inputs)
    adv0_outputs = dis0(gen0_outputs) + [enc0(gen0_outputs)]
    # feature1+z0 na prawdopodobieństwo, że feature1 jest prawdziwa 
    # + rekonstrukcja z0 + rekonstrukcja feature0/obraz

    adv0 = Model(gen0_inputs, adv0_outputs, name="adv0")
    # funkcje straty: 1) prawdopodobieństwo, że feature1 (cecha 1) jest prawdziwa (strata współzawodniczącej0)
    # 2) strata sieci Q0 lub entropii0
    # 3) strata warunkowa0
    loss = ['binary_crossentropy', 'mse', 'mse']
    loss_weights = [1.0, 10.0, 1.0] 
    adv0.compile(loss=loss,
                 loss_weights=loss_weights,
                 optimizer=optimizer,
                 metrics=['accuracy'])
    adv0.summary()

    # konstruowanie modelu współzawodnicząca1 =
    # generator1+dyskryminator1+koder1
    # zamrożenie wag koder1
    enc1.trainable = False
    # zamrożenie wag dyskryminator1
    dis1.trainable = False
    gen1_inputs = [labels, z1]
    gen1_outputs = gen1(gen1_inputs)
    adv1_outputs = dis1(gen1_outputs) + [enc1(gen1_outputs)]
    # etykiety+z0 na prawdopodobieństwo, że etykiety są prawdziwe+rekonstrukcja z1 + rekonstrukcja feature1
    adv1 = Model(gen1_inputs, adv1_outputs, name="adv1")
    # funkcje straty: 1) prawdopodobieństwo, że etykiety są prawdziwe (strata współzawodniczącej1)
    # 2) strata sieci Q1 lub entropii1
    # 3) strata warunkowa1 (błąd klasyfikatora)
    loss_weights = [1.0, 1.0, 1.0] 
    loss = ['binary_crossentropy', 
            'mse',
            'categorical_crossentropy']
    adv1.compile(loss=loss,
                 loss_weights=loss_weights,
                 optimizer=optimizer,
                 metrics=['accuracy'])
    adv1.summary()

    # uczenie dyskryminatora i sieci współzawodniczącej
    models = (enc0, enc1, gen0, gen1, dis0, dis1, adv0, adv1)
    params = (batch_size, train_steps, num_labels, z_dim, model_name)
    train(models, data, params)


def test_generator(generators, params, z_dim=50):
    class_label, z0, z1, p0, p1 = params
    step = 0
    if class_label is None:
        num_labels = 10
        noise_class = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_class = np.zeros((16, 10))
        noise_class[:,class_label] = 1
        step = class_label

    if z0 is None:
        z0 = np.random.normal(scale=0.5, size=[16, z_dim])
    else:
        if p0:
            a = np.linspace(-4.0, 4.0, 16)
            a = np.reshape(a, [16, 1])
            z0 = np.ones((16, z_dim)) * a
        else:
            z0 = np.ones((16, z_dim)) * z0
        print("z0: ", z0[:,0])

    if z1 is None:
        z1 = np.random.normal(scale=0.5, size=[16, z_dim])
    else:
        if p1:
            a = np.linspace(-1.0, 1.0, 16)
            a = np.reshape(a, [16, 1])
            z1 = np.ones((16, z_dim)) * a
        else:
            z1 = np.ones((16, z_dim)) * z1
        print("z1: ", z1[:,0])

    noise_params = [noise_class, z0, z1]

    plot_images(generators,
                noise_params=noise_params,
                show=True,
                step=step,
                model_name="test_outputs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Zaladuj generator 0 modelu z pliku h5 z wagami po wytrenowaniu"
    parser.add_argument("-g", "--generator0", help=help_)
    help_ = "Zaladuj generator 1 modelu z pliku h5 z wagami po wytrenowaniu"
    parser.add_argument("-k", "--generator1", help=help_)
    # help_ = "Zaladuj model kodera z pliku h5 z wagami po wytrenowaniu"
    # parser.add_argument("-e", "--encoder", help=help_)
    help_ = "Podaj konkretną cyfre ktora chcesz wygenerowac"
    parser.add_argument("-d", "--digit", type=int, help=help_)
    help_ = "Podaj kod szumu z0 (jako 50-wymiarowy przy ustalonym z0)"
    parser.add_argument("-z", "--z0", type=float, help=help_)
    help_ = "Podaj kod szumu z1 (jako 50-wymiarowy przy ustalonym z1)"
    parser.add_argument("-x", "--z1", type=float, help=help_)
    help_ = "Wydrukuj cyfry z z0 w zakresie od -n1 do +n2"
    parser.add_argument("--p0", action='store_true', help=help_)
    help_ = "Wydrukuj cyfry z z1 w zakresie od -n1 do +n2"
    parser.add_argument("--p1", action='store_true', help=help_)
    args = parser.parse_args()
    # if args.encoder:
    #    encoder = args.encoder
    #else:
    #    encoder = None
    if args.generator0:
        gen0 = load_model(args.generator0)
        if args.generator1:
            gen1 = load_model(args.generator1)
        else:
            print("Trzeba podać modele zarówno generatora 0 oraz generatora 1")
            exit(0)
        class_label = args.digit
        z0 = args.z0
        z1 = args.z1
        p0 = args.p0
        p1 = args.p1
        params = (class_label, z0, z1, p0, p1)
        test_generator((gen0, gen1), params)
    else:
        build_and_train_models()
