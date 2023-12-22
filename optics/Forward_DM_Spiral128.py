
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
from optics.Prop_layer_v4_sizes import Propagation
from optics.DOE_layer_v4_128 import DOE, DOE_Free
from optics.Sensing_layer import Sensing

def Forward_DM_Spiral(input_size=(128, 128,25), DOE_typeA='Zeros',name='Forward_Model',Train_c = True):

    # define the model input
    MSS = 128
    Minput = 128
    inputs = Input(shape=input_size)

    # Customer _layer_psfs
    #ptf("aca")
    In_DOE1a = Propagation(Mp=Minput, L=0.01, zi=0.06, Trai=False)(inputs) #AcÃ¡ hay que revisar dimensiones de la proyeccion y tamaÃ±os Hay que ajustar porque no está propagando
    Out_DOE1a = DOE(Mdoe=MSS, Mesce=Minput,  DOE_type=DOE_typeA, Trai=Train_c)(In_DOE1a)
    In_IPa = Propagation(Mp=MSS, L=0.006, zi=0.05, Trai=False)(Out_DOE1a)
    Out_DOE2a = DOE(Mdoe=MSS, Mesce=MSS,  DOE_type='Spiral', Trai=False)(In_IPa)###### true
    In_Sensora = Propagation(Mp=MSS, L=0.0048, zi=0.01, Trai=False)(Out_DOE2a)  # Para el T1 los puse a 0.05 para t2 desde 0.1
    Measurement = Sensing(Ms=MSS, Trai=False)(In_Sensora)
    
     


    # construct the CNN
    model = Model(inputs, Measurement,name=name)

    return model


def Forward_DM_Spiral_Free(input_size=(128, 128, 25), Nterms = 150, DOE_typeA='Zeros', name='Forward_Model', Train_c=True):
    # define the model input
    MSS = 128
    Minput = 128
    inputs = Input(shape=input_size)

    # Customer _layer_psfs
    # ptf("aca")
    In_DOE1a = Propagation(Mp=Minput, L=0.01, zi=0.06, Trai=False)(inputs)  # AcÃ¡ hay que revisar dimensiones de la proyeccion y tamaÃ±os Hay que ajustar porque no está propagando
    Out_DOE1a = DOE_Free(Mdoe=MSS, Nterms=Nterms, Mesce=Minput, DOE_type=DOE_typeA, Trai=Train_c)(In_DOE1a)
    In_IPa = Propagation(Mp=MSS, L=0.006, zi=0.05, Trai=False)(Out_DOE1a)
    Out_DOE2a = DOE(Mdoe=MSS, Mesce=MSS, DOE_type='Spiral', Trai=False)(In_IPa)  ###### true
    In_Sensora = Propagation(Mp=MSS, L=0.0048, zi=0.01, Trai=False)(
        Out_DOE2a)  # Para el T1 los puse a 0.05 para t2 desde 0.1
    Measurement = Sensing(Ms=MSS, Trai=False)(In_Sensora)

    # construct the CNN
    model = Model(inputs, Measurement, name=name)

    return model