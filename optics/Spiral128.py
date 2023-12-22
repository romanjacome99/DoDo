
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
from Three_shots_Current.Func.Prop_layer_v4_sizes import Propagation
from Three_shots_Current.Func.Concat_layer import Concatenation
from Three_shots_Current.Func.DOE_layer_v4_64 import DOE
from Three_shots_Current.Func.Sensing_layer import Sensing

def Spiral(input_size=(512, 512, 3), depth=3, depth_out=25, diam=3e-6):

    # define the model input
    MSS = 128
    Minput = 128
    inputs = Input(shape=input_size)

    # Customer _layer_psfs
    #ptf("aca")
    In_DOE1a = Propagation(Mp=Minput, L=1, zi=0.1, Trai=False)(inputs) #AcÃ¡ hay que revisar dimensiones de la proyeccion y tamaÃ±os

    In_IPa = Propagation(Mp=MSS, L=0.03, zi=0.01, Trai=False)(In_DOE1a)
  
    Out_DOE2a = DOE(Mdoe=MSS, Mesce=MSS,  DOE_type='Spiral' ,  Trai=False)(In_IPa)
    In_Sensora = Propagation(Mp=MSS, L=0.03, zi=0.03, Trai=False)(Out_DOE2a)  # Para el T1 los puse a 0.05 para t2 desde 0.1
    final = Sensing(Ms=MSS, Trai=False)(In_Sensora)
    

    # construct the CNN
    model = Model(inputs, final)    
   

    return model