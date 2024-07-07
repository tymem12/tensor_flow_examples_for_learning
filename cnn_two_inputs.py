'''
NETWORK Y
Implementation of the network with 2 images as input. It is the same image but due to the fact that the the processing of input in one sub layer would be different then in the second one, we will have different weights and we are able to find better network.
The structure is like this:
- First image is normal cnn with dilatation rate equal to 1 (kernel is 3 x 3) so the receptive field is also 3 x 3
- Second image is cnn with dilatation rate equal to 1 so kernel is 3 x 3 but receptive field is 5 x 5 (the distance between values in the kernel is 2 not 1 so the field that is used for analyzing the image is bigger)

Both inputs are then combined into one and then standard operations like:  flattening, dropout, dense(10), softmax

Additionally in this network every time after pooling we have twice as big number of filters to compensate that.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, concatenate
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import plot_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

input_shape = (image_size, image_size, 1)
batch_size = 32
kernel_size = 3
dropout = 0.4
n_filters = 32

# left branch of the Y network
left_inputs = Input(shape=input_shape)
x = left_inputs
filters = n_filters
# 3 layers of Conv2D-Dropout-MaxPooling2D
# number of filters after each set of layers would be doubled (32-64-128)
for i in range(3):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',  # we do not want to change the size
               activation='relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D()(x)
    filters *= 2  # doubled the filters

# right branch of the 
right_inputs = Input(shape=input_shape)
y = right_inputs
filters = n_filters
# 3 layers of Conv2D-Dropout-MaxPooling2D
# number of filters after each set of layers would be doubled (32-64-128)
for i in range(3):
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu',
               dilation_rate=2)(y)   # setting the dilatation rate to 2. 1 is th default one
    y = Dropout(dropout)(y)
    y = MaxPooling2D()(y)
    filters *= 2

# concatenation of outputs from two  
y = concatenate([x, y])
# transformation of the map of features before the connecting it to the dense layer
# - flattening
y = Flatten()(y)
# - dropout
y = Dropout(dropout)(y)
outputs = Dense(num_labels, activation='softmax')(y)

# creating the model by passing the list of inputs and the output (functional API)
model = Model([left_inputs, right_inputs], outputs)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit([x_train, x_train],
          y_train, 
          validation_data=([x_test, x_test], y_test),
          epochs=20,
          batch_size=batch_size)

score = model.evaluate([x_test, x_test],
                       y_test,
                       batch_size=batch_size,
                       verbose=0)
print("\nDokładność na zbiorze testowym: %.1f%%" % (100.0 * score[1]))
