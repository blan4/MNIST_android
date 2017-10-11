# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt

from data import load_noise_data


def plot(x=(), y=(), n=10):
    plt.figure(figsize=(20, 2))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(y[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


if __name__ == '__main__':
    x_train, x_train_noisy, y_train, x_test, x_test_noisy, y_test = load_noise_data()
    plot(x_train_noisy, x_train)
