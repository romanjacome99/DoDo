import tensorflow as tf
import os
from matplotlib import pyplot as plt

from os import listdir
from os.path import isfile, join

import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import scipy.io
import pandas as pd
from scipy.io import loadmat
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from os.path import isfile, join
from pathlib import Path


# ---------------------defined function used -------------------------------------------------------------------

# Defined protocol of input and ouput (it is recommended only read, not applied operator here, but it is possible)
def Input_image(image):
    images = loadmat(image).get('cube').astype(np.float32)
    images = images/np.max(images)############
    x = 0#np.int(np.random.rand() * 48)
    y = 0#np.int(np.random.rand() * 48)
    return images[x:x+128, y:y+128, 3:-3]


def Oput_image(image):
    images = loadmat(image).get('cube').astype(np.float32)
    images = images/np.max(images)############
    x = 0#np.int(np.random.rand() * 48)
    y = 0#np.int(np.random.rand() * 48)
    #yi = cv2.resize(images[x:x+450, y:y+450, 3:-3], dsize=(150,150), interpolation=cv2.INTER_CUBIC)
    yi = images[x:x+128, y:y+128, 3:-3]
    #yi = tf.nn.avg_pool(images[x:x+450, y:y+450, 3:-3],
    #               [1, factor, factor, 1],
    #               strides=[1, factor, factor, 1],
    #               padding="VALID")
    return yi


def load_sambles(data):
    data = data[['inimg']]
    inimg_name = list(data.iloc[:, 0])
    samples = []
    for samp in inimg_name:
        samples.append(samp)
    return samples


class DataGenerator(Sequence):
    def __init__(self, samples, PATH, batch_size=1, dim_input=(512, 512, 3), shuffle=False, dim_oput=(512, 512, 3)):
        'Initialization'
        self.dim_input = dim_input
        self.dim = dim_oput
        self.dim_oput = dim_oput
        self.batch_size = batch_size
        self.list_images = samples
        self.shuffle = shuffle
        self.PATH = PATH
        self.on_epoch_end()
        self.shape = [None, None, None]

    # falta __data_generation
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.list_images) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        images_name = [self.list_images[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(images_name)

        return X, y

    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.list_images))
        if self.shuffle == False:
            np.random.shuffle(self.indexes)

    def __data_generation(self, images_names):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.empty((self.batch_size, *self.dim_input))  # Array de numpy con zeros de tamaÃƒÆ’Ã‚Â±o
        Y = np.empty((self.batch_size, *self.dim_oput))
        # Generate data
        for i, file_name in enumerate(images_names):
            # Store sample

            X[i,] = Input_image(self.PATH  + file_name)
            #Y[i,] = Oput_image(self.PATH + file_name)
            
            # Store class
            #path2 = str(Path(self.PATH).parents[0])
            #y[i,] = Oput_image(path2 + '/Formated/' + file_name)
        return X, X


class BATCH_SIZE_o:
    pass


def Build_data_set(IMG_WIDTH=128, IMG_HEIGHT=128, IMG_WIDTH_o=128, IMG_HEIGHT_o=128, L_bands=31, L_imput=12, BATCH_SIZE=4, PATH=None):
    # Random split
    #data_dir_list = os.listdir(PATH)
    data_dir_list = [f for f in listdir(PATH) if isfile(join(PATH, f))]
    N = len(data_dir_list)
    train_df = pd.DataFrame(columns=['inimg'])
    test_df = pd.DataFrame(columns=['inimg'])
    randurls = np.copy(data_dir_list)
    train_n = round(N * 1)
    np.random.shuffle(randurls)
    tr_urls = randurls[:train_n]
    ts_urls = randurls[train_n:N]
    for i in tr_urls:
        train_df = train_df.append({'inimg': i}, ignore_index=True)
    for i in ts_urls:
        test_df = test_df.append({'inimg': i}, ignore_index=True)

    params = {'dim_input': (IMG_WIDTH, IMG_HEIGHT, L_imput),
              'dim_oput': ( IMG_WIDTH_o,  IMG_HEIGHT_o, L_bands),
              'batch_size': BATCH_SIZE,
              'PATH': PATH,
              'shuffle': False}

    partition_Train = load_sambles(test_df)
    partition_Test = load_sambles(train_df)

    train_generator = DataGenerator(partition_Train, **params)
    test_generator = DataGenerator(partition_Test, **params)

    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        (tf.float32, tf.float32), output_shapes=(
            tf.TensorShape([BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, L_imput]),
            tf.TensorShape([ BATCH_SIZE, IMG_WIDTH_o, IMG_HEIGHT_o, L_bands])
        ))

    test_dataset = tf.data.Dataset.from_generator(
        lambda: test_generator,
        (tf.float32, tf.float32), output_shapes=(
            tf.TensorShape([BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, L_imput]),
            tf.TensorShape([BATCH_SIZE, IMG_WIDTH_o,  IMG_HEIGHT_o, L_bands])
        ))

    return train_dataset, test_dataset
