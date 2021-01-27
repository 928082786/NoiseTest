import numpy as np


def load_mnist(path: str, raw: bool = False):
    """
    Loads MNIST dataset from `config.ART_DATA_PATH` or downloads it if necessary.

    :param path: dataset path.
    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :return: `(x_train, y_train), (x_test, y_test), min, max`.
    """
    dict_mnist = np.load(path)
    x_train = dict_mnist["x_train"]
    y_train = dict_mnist["y_train"]
    x_test = dict_mnist["x_test"]
    y_test = dict_mnist["y_test"]
    dict_mnist.close()

    # Add channel axis
    min_, max_ = 0.0, 255.0
    if not raw:
        min_, max_ = 0.0, 1.0
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
    x_train = x_train/255.
    x_test = x_test/255.
    return x_train, y_train, x_test, y_test, min_, max_
