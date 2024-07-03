''' Klasyfikacja cyfr MNIST z użyciem CNN

Trójwarstwowa CNN do klasyfikacji cyfr MNIST
- Pierwsze dwie warstwy to Conv2D-ReLU-MaxPool
- Trzecia warstwa to Conv2D-ReLU-Dropout
Czwarta warstwa to Dense(10)
Funkcja aktywacji wyjscia to softmax (propabilities sum to 1)
Optymalizator to Adam

99.4% dokładność na zbiorze testowym po 10 epokach

https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist

# LOADING THE DATASET
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# UNIQUE LABELS
num_labels = len(np.unique(y_train))

# Converting the label to the vector woth only one 1 and rest 0 (OH)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1] # normally shape is (60000, 28, 28) so taking [1] return the 28

x_train = np.reshape(x_train, [-1, image_size, image_size, 1]) # we make sure that the x_train is (60000, 28, 28, 1)
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])

# normalization so we will have the 0-1 values
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (image_size, image_size, 1) #input shape (how many inputs would be)
batch_size = 128        # how much would be passed in one learning step (before callulating the loss function in SGD)
kernel_size = 3         # length of the kernel (3,3,1)                    
pool_size = 2           # length of the pooling matrix
filters = 64            # How many maps to find the traits
dropout = 0.2           # regularization- percentage of the neurons that would not be active durining training (dirrent neurons per example no per batch)

# relu = max(0, x)
# model creation
model = Sequential()
model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu'))
model.add(Flatten())
model.add(Dropout(dropout))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

# now we need to compile model we need to set up the optimalization and loss function
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics  = ['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size= batch_size)
_, acc = model.evaluate(x_test, y_test, batch_size = batch_size, verbose= 0)
print("\nAccuaracy: ", acc)
