from keras.datasets import mnist
from libs.dataset_processing import BatchDataLoader
import numpy as np


def get_train_validation_mnist_generators(batch_size, shuffle=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))


    train_loader = BatchDataLoader([[x_train[i], x_train[i]] for i in range(len(y_train))],
                                   batch_size, shuffle=shuffle)
    test_loader = BatchDataLoader([[x_test[i], x_test[i]] for i in range(len(y_test))],
                                  batch_size, shuffle=shuffle)

    return train_loader, test_loader
