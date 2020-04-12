import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, (2, 2), input_shape=x_train.shape[1:], activation="relu"))
model.add(keras.layers.Conv2D(32, (2, 2), activation="relu"))
model.add(keras.layers.MaxPooling2D((3, 3)))

model.add(keras.layers.Conv2D(64, (4, 4), activation="relu"))
model.add(keras.layers.Conv2D(64, (4, 4), activation="relu"))
model.add(keras.layers.MaxPooling2D((3, 3)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(1080, activation="relu"))
model.add(keras.layers.Dense(1080, activation="relu"))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(keras.optimizers.Adam(lr=0.00001, decay=1e-6),
              "categorical_crossentropy", metrics=['acc'])

model.summary()

datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=10,
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.1,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

model.fit_generator(datagen.flow(
    x_train, 
    y_train, 
    batch_size=32), 
    epochs=100, 
    validation_data=(x_test, y_test)
)

scores = model.evaluate(x_test, y_test)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model.save('../models/Cifar10.h5')
