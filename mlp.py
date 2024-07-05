'''
Sieć MLP do klasyfikacji cyfr MNIST

Dokładność na zbiorze testowym 98.3% w 20 epokach

https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist

# załadowanie zbioru MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Zliczenie liczby etykiet
num_labels = len(np.unique(y_train))

# konwersja na wektor „jeden-aktywny” (OH)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# wymiary obrazów (przy założeniu, że są kwadratami)
image_size = x_train.shape[1]
input_size = image_size * image_size

# zmiana rozmiaru i normalizacja
x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255

# parametry sieci
batch_size = 128
hidden_units = 256
dropout = 0.45

# modelem jest trójwarstwowy MLP z ReLU i pomijaniem (ang. dropout) w każdej warstwie
model = Sequential()
model.add(Dense(hidden_units, input_dim=input_size)) # first dense layer need to know what is input 
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(num_labels))
# to jest wyjście dla wektora „jeden-aktywny” (OH) 
model.add(Activation('softmax'))
model.summary()

# funkcja straty dla wektora OH, optymalizator Adam
# dokładność jest odpowiednią miarą do oceny jakości klasyfikatora
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# trenowanie sieci
model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

# sprawdzenie poprawności działania modelu na zbiorze testowym, by ocenić zdolność uogólniania
_, acc = model.evaluate(x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose=0)
print("\nDokładność na zbiorze testowym: %.1f%%" % (100.0 * acc))