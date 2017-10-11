# -*- coding: utf-8 -*-
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Lambda
from keras.models import Sequential
from keras.optimizers import Adadelta

from data import load_data


def build_model():
    """
    Build convolution autoencoder
    :return:
    :rtype Sequential:
    """

    model = Sequential([
        noise_layer(
            input_shape=[28, 28, 1],
            noise_factor=0.5),

        # Encoder
        # 28*28*1
        Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'),
        MaxPooling2D(
            pool_size=(2, 2),
            padding='same'),
        # 14*14*16

        Conv2D(
            filters=8,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'),
        MaxPooling2D(
            pool_size=(2, 2),
            padding='same'),
        # 7*7*8

        Conv2D(
            filters=8,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'),
        MaxPooling2D(
            pool_size=(2, 2),
            padding='same'),
        # 4*4*8

        # Decoder
        Conv2D(
            filters=8,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'),
        UpSampling2D(
            size=(2, 2)),
        # 8*8*8

        Conv2D(
            filters=8,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'),
        UpSampling2D(
            size=(2, 2)),
        # 16*16*8

        Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation='relu',
            padding='valid'),
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
    return model


def train(model, x_train, y_train, x_test, y_test):
    """

    :param model:
    :type model: Sequential
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    model.compile(optimizer=Adadelta(), loss=K.binary_crossentropy)
    model.fit(x=x_train,
              y=x_train,
              epochs=50,
              batch_size=128,
              shuffle=True,
              validation_data=(x_test, x_test),
              callbacks=[TensorBoard(
                  log_dir="/tmp/tensorflow/autoencoder",
                  write_images=False,
                  histogram_freq=0,
                  batch_size=128
              )])


def noise_layer(input_shape, noise_factor=0.5):
    def add_noise(x):
        return K.clip(x + K.random_normal(
            shape=K.shape(x),
            mean=0.5,
            stddev=noise_factor), min_value=0., max_value=1.)

    return Lambda(add_noise, name='noiser', input_shape=input_shape)


def main():
    x_train, y_train, x_test, y_test = load_data()
    model = build_model()
    train(model, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
