'''
In RNN the activation from the previous sequence (timestamp) is used to calculate the current output.
ACTIVATION FUNCTION 
h_t = tanh(b + W * h_t_minus1 + U * x_t) 
b - bias
W - weight used with the activation from previous step
h_t_minus1 - activation from previous step
U - weight
x_t - current input

Simple RNN scenario
after going trough the activation function h_t goes as the input the the next sequence. Also h_t goes to the softmax so we could obtain y_t

RNN scenario
It is like Simple RNN but with additional step where before going to the softmax the h_t goes to the another function o_t = V * h_t + c, and the result of that goes to softmax
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist

# 1. load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. counting the labels
num_labels = len(np.unique(y_train))

# 3. convert to 0-1 vector
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 4. change shape and normalization
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size])
x_test = np.reshape(x_test,[-1, image_size, image_size])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# parameters
input_shape = (image_size, image_size)
batch_size = 128
units = 256
dropout = 0.2

# model is the RNN with 256 units and 28 as input
model = Sequential()
model.add(SimpleRNN(units=units,
                    dropout=dropout,
                    input_shape=input_shape))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

# loss function, optimizer and metric
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

_, acc = model.evaluate(x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose=0)
print("\nDokładność na zbiorze testowym: %.1f%%" % (100.0 * acc)) 
