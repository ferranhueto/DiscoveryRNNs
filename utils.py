"""
Cough Classification utils.

These functions are used to assist in the classification of coughs using TL
with the Librispeech Tensorflow dataset.
"""

import numpy as np
import tensorflow as tf

FILE_DIR = "/home/fhueto/code/covid_lstm"
DB_DIR_COVID = f"{FILE_DIR}/dataset"
MODEL_DIR = f"{FILE_DIR}/models"


def get_dataset_from_pickle(var="gender"):
    """Get COVID19 dataset from pickle files."""
    X = np.load(f"{DB_DIR_COVID}/covid19_clean_{var}_x.pkl",
                allow_pickle=True)
    Y = np.load(f"{DB_DIR_COVID}/covid19_clean_{var}_y.pkl",
                allow_pickle=True)
    return X, Y


def reshape_timesteps(X, width=4, stride=2):
    """Reshape timesteps so as to have them overlap."""
    new_X = []
    for x in X:
        new_x = [np.concatenate(x[0:2]),
                 np.concatenate(x[1:3]),
                 np.concatenate(x[2:4]),
                 np.concatenate(x[3:5])]
        new_X.append(new_x)

    return np.asarray(new_X)


def augment_dataset(X, Y):
    """Augment with poisson noise."""
    new_X = []
    new_Y = []
    count = 0
    for x, y in zip(X, Y):
        count += 1
        new_X.append(x)
        new_Y.append(y)
        for i in range(3):
            new_x = poisson_noise(x)
            new_X.append(new_x)
            new_Y.append(y)
        print(f"Sample {count} of {len(X)} augmented.", end="\r")

    return np.asarray(new_X), np.asarray(new_Y)


def reshape_dataset(X_train, y_train, X_test, y_test):
    """Reshape dataset for Convolution."""
    X_test = X_test.reshape(X_test.shape[0],
                            X_test.shape[1],
                            X_test.shape[2],
                            1).astype('float32')
    X_train = X_train.reshape(X_train.shape[0],
                              X_train.shape[1],
                              X_train.shape[2],
                              1).astype('float32')
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    return (X_train, y_train), (X_test, y_test)


def normalize_dataset(X_train):
    """Normalize speech recognition and computer vision datasets."""
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = (X_train-std)/mean
    return X_train


def poisson_noise(img, weight=1):
    """Generate poisson noise mask for input image."""
    noise_mask = (np.random.poisson(np.abs(img * 255.0 * weight))/weight/255).astype(float)
    return noise_mask


def switch_to_binary(Y):
    """Switch categorical dataset to binary."""
    Y_new = []
    for y in Y:
        if y[0]==0:
            Y_new.append(0)
        else:
            Y_new.append(1)

    return np.asarray(Y_new)
