
import tensorflow as tf
import os
from matplotlib import pyplot as plt

from os import listdir
from os.path import isfile, join

import numpy as np
import scipy.io
import pandas as pd
from scipy.io import loadmat
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow import print as ptf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.constraints import MinMaxNorm

def InvSpiral(input_size=(128, 128, 3), depth=3, depth_out=25,name='Inverse_Spiral'):

    inputs = Input(input_size)
    conv1 = Conv2DTranspose(32, 12,  kernel_initializer = 'he_normal' )(inputs)
    conv2 = Conv2DTranspose(32, 12,  kernel_initializer = 'he_normal' )(conv1)
    conv3 = Conv2DTranspose(25, 12,  kernel_initializer = 'he_normal' )(conv2)
    Out = Conv2DTranspose(25, 12,  kernel_initializer = 'he_normal' )(conv3)
    Out2 = tf.keras.layers.Cropping2D(cropping=((22, 22), (22, 22)))(Out)
    #conv10 = tf.keras.activations.relu(conv10)
    # construct the CNN
    model = Model(inputs, Out2,name=name)
    return model