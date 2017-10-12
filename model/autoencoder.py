# -*- coding: utf-8 -*-
import os

import tensorflow as tf
from keras import backend as K
from keras import metrics
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator

from data import load_noise_data
from exporter import export_model


def build_model():
    model = Sequential([
        # 28*28*1

        # Encoder
        Conv2D(
            input_shape=[28, 28, 1],
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'),
        MaxPooling2D(
            pool_size=(2, 2),
            padding='same'),
        # 14*14*32

        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'),
        MaxPooling2D(
            pool_size=(2, 2),
            padding='same'),
        # 7*7*32

        # Decoder
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'),
        UpSampling2D(
            size=(2, 2)),
        # 8*8*32

        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'),
        UpSampling2D(
            size=(2, 2)),
        # 32*32*16

        Conv2D(
            filters=1,
            kernel_size=(3, 3),
            activation='sigmoid',
            padding='same')
        # 32*32*1
    ])

    model.summary()
    model.compile(optimizer=Adadelta(),
                  loss=K.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])
    return model


def train(model, x_train, y_train, x_test, y_test, epochs=50, batch_size=128):
    model.fit(x=x_train, y=y_train,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(x_test, y_test),
              callbacks=[TensorBoard(
                  log_dir="/tmp/tensorflow/autoencoder",
                  write_images=True,
                  histogram_freq=5,
                  batch_size=batch_size
              )])


def train_with_augmentation(model, x_train, y_train, x_test, y_test, epochs=50, batch_size=128):
    gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.08, shear_range=0.3,
                             height_shift_range=0.08, zoom_range=0.3)
    test_gen = ImageDataGenerator()
    gen.fit(x_train)

    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
    test_generator = test_gen.flow(x_test, y_test, batch_size=batch_size)

    model.fit_generator(train_generator,
                        steps_per_epoch=500,
                        epochs=epochs,
                        validation_steps=50,
                        validation_data=test_generator,
                        callbacks=[TensorBoard(
                            log_dir="/tmp/tensorflow/autoencoder",
                            write_images=True,
                            histogram_freq=0,
                            batch_size=batch_size
                        )])


def main():
    x_train, x_train_noisy, _, x_test, x_test_noisy, _ = load_noise_data()
    model = build_model()
    train(model, x_train_noisy, x_train, x_test_noisy, x_test, epochs=50)

    if not os.path.exists('out'):
        os.mkdir('out')

    export_model(tf.train.Saver(), ["conv2d_1_input"], ["conv2d_5/Sigmoid"], "mnist_autoencoder")
    model.save("out/autoencoder.h5")


if __name__ == '__main__':
    main()
