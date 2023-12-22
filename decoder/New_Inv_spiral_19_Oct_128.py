
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
import copy
from scipy import interpolate
from Multishot_128x128.Func.Inv_Mask import *  # Net build

def kronecker_product(mat1, mat2):
    """Computes the Kronecker product two matrices."""
    m1, n1 = mat1.get_shape().as_list()
    mat1_rsh = tf.reshape(mat1, [1, m1, 1, n1, 1])
    m2, n2 = mat2.get_shape().as_list()
    mat2_rsh = tf.reshape(mat2, [1, 1, m2, 1, n2])
    return tf.reshape(tf.multiply(mat1_rsh, mat2_rsh), [1, m1 * m2, n1 * n2, 1])
    
def DA_1_1(tensor):
    Aux    = tf.ones([64, 64])
    Coded  = tf.constant([[1.0,0.0],[0.0,0.0]])
    Mask   = kronecker_product(Aux,Coded)
    tensor = tf.math.multiply(Mask,tensor)
    return tensor 

def DA_2_1(tensor):
    Aux    = tf.ones([64, 64])
    Coded  = tf.constant([[0.0,1.0],[0.0,0.0]])
    Mask   = kronecker_product(Aux,Coded)
    tensor = tf.math.multiply(Mask,tensor)
    return tensor 
    
def DA_1_2(tensor):
    Aux    = tf.ones([64, 64])
    Coded  = tf.constant([[0.0,0.0],[1.0,0.0]])
    Mask   = kronecker_product(Aux,Coded)
    tensor = tf.math.multiply(Mask,tensor)
    return tensor 
    
def DA_2_2(tensor):
    Aux    = tf.ones([64, 64])
    Coded  = tf.constant([[0.0,0.0],[0.0,1.0]])
    Mask   = kronecker_product(Aux,Coded)
    tensor = tf.math.multiply(Mask,tensor)
    return tensor 

def Both_mask(input_size=(128, 128, 3), depth=3, depth_out=25,name='Inverse_Spiral'):
    
    Inv1_1=Inv_Mask(input_size=(128,128,3),name='Spiral_Inverse1_1')
    Inv2_1=Inv_Mask(input_size=(128,128,3),name='Spiral_Inverse2_1')
    Inv1_2=Inv_Mask(input_size=(128,128,3),name='Spiral_Inverse1_2')
    Inv2_2=Inv_Mask(input_size=(128,128,3),name='Spiral_Inverse2_2')
    
    inputs = Input(input_size)
    
    R_DA_1_1 = tf.keras.layers.Lambda(DA_1_1, name="L_1_1")(inputs)
    RR_1_1_A   = Inv1_1(R_DA_1_1)
    RR_1_1   = tf.keras.layers.Lambda(DA_1_1, name="L_1_1_F")(RR_1_1_A)
    
    R_DA_2_1 = tf.keras.layers.Lambda(DA_2_1, name="L_2_1")(inputs)
    RR_2_1_A   = Inv2_1(R_DA_2_1)
    RR_2_1   = tf.keras.layers.Lambda(DA_2_1, name="L_2_1_F")(RR_2_1_A)
    
    R_DA_1_2 = tf.keras.layers.Lambda(DA_1_2, name="L_1_2")(inputs)
    RR_1_2_A   = Inv1_2(R_DA_1_2)
    RR_1_2   = tf.keras.layers.Lambda(DA_1_2, name="L_1_2_F")(RR_1_2_A)
    
    R_DA_2_2 = tf.keras.layers.Lambda(DA_2_2, name="L_2_2")(inputs)
    RR_2_2_A   = Inv2_2(R_DA_2_2)
    RR_2_2   = tf.keras.layers.Lambda(DA_2_2, name="L_2_2_F")(RR_2_2_A)
    
    add1 =tf.keras.layers.add([RR_1_1, RR_2_1, RR_1_2, RR_2_2])
    # construct the CNN
    model = Model(inputs, add1,name=name)
    return model
    
def primer_mask(input_size=(128, 128, 3), depth=3, depth_out=25,name='Inverse_Spiral'):
    
    Inv1_1=Inv_Mask(input_size=(128,128,3),name='Spiral_Inverse1_1')
    Inv2_1=Inv_Mask(input_size=(128,128,3),name='Spiral_Inverse2_1')
    Inv1_2=Inv_Mask(input_size=(128,128,3),name='Spiral_Inverse1_2')
    Inv2_2=Inv_Mask(input_size=(128,128,3),name='Spiral_Inverse2_2')
    
    inputs = Input(input_size)
    
    R_DA_1_1 = tf.keras.layers.Lambda(DA_1_1, name="L_1_1")(inputs)
    RR_1_1_A   = Inv1_1(R_DA_1_1)
    #RR_1_1   = tf.keras.layers.Lambda(DA_1_1, name="L_1_1_F")(RR_1_1_A)
    
    R_DA_2_1 = tf.keras.layers.Lambda(DA_2_1, name="L_2_1")(inputs)
    RR_2_1_A   = Inv2_1(R_DA_2_1)
    #RR_2_1   = tf.keras.layers.Lambda(DA_2_1, name="L_2_1_F")(RR_2_1_A)
    
    R_DA_1_2 = tf.keras.layers.Lambda(DA_1_2, name="L_1_2")(inputs)
    RR_1_2_A   = Inv1_2(R_DA_1_2)
    #RR_1_2   = tf.keras.layers.Lambda(DA_1_2, name="L_1_2_F")(RR_1_2_A)
    
    R_DA_2_2 = tf.keras.layers.Lambda(DA_2_2, name="L_2_2")(inputs)
    RR_2_2_A   = Inv2_2(R_DA_2_2)
    #RR_2_2   = tf.keras.layers.Lambda(DA_2_2, name="L_2_2_F")(RR_2_2_A)
    
    add1 =tf.keras.layers.add([RR_1_1_A, RR_2_1_A, RR_1_2_A, RR_2_2_A])
    # construct the CNN
    model = Model(inputs, add1,name=name)
    return model
    
def Sec_mask(input_size=(128, 128, 3), depth=3, depth_out=25,name='Inverse_Spiral'):
    
    Inv1_1=Inv_Mask(input_size=(128,128,3),name='Spiral_Inverse1_1')
    Inv2_1=Inv_Mask(input_size=(128,128,3),name='Spiral_Inverse2_1')
    Inv1_2=Inv_Mask(input_size=(128,128,3),name='Spiral_Inverse1_2')
    Inv2_2=Inv_Mask(input_size=(128,128,3),name='Spiral_Inverse2_2')
    
    inputs = Input(input_size)
    
    #R_DA_1_1 = tf.keras.layers.Lambda(DA_1_1, name="L_1_1")(inputs)
    RR_1_1_A   = Inv1_1(inputs)
    RR_1_1   = tf.keras.layers.Lambda(DA_1_1, name="L_1_1_F")(RR_1_1_A)
    
    #R_DA_2_1 = tf.keras.layers.Lambda(DA_2_1, name="L_2_1")(inputs)
    RR_2_1_A   = Inv2_1(inputs)
    RR_2_1   = tf.keras.layers.Lambda(DA_2_1, name="L_2_1_F")(RR_2_1_A)
    
    #R_DA_1_2 = tf.keras.layers.Lambda(DA_1_2, name="L_1_2")(inputs)
    RR_1_2_A   = Inv1_2(inputs)
    RR_1_2   = tf.keras.layers.Lambda(DA_1_2, name="L_1_2_F")(RR_1_2_A)
    
    #R_DA_2_2 = tf.keras.layers.Lambda(DA_2_2, name="L_2_2")(inputs)
    RR_2_2_A   = Inv2_2(inputs)
    RR_2_2   = tf.keras.layers.Lambda(DA_2_2, name="L_2_2_F")(RR_2_2_A)
    
    add1 =tf.keras.layers.add([RR_1_1, RR_2_1, RR_1_2, RR_2_2])
    # construct the CNN
    model = Model(inputs, add1,name=name)
    return model
    
def No_mask(input_size=(32, 32, 3), depth=3, depth_out=25,name='Inverse_Spiral'):
    
    Inv1_1=Inv_Mask(input_size=(150,150,3),name='Spiral_Inverse1_1')
    Inv2_1=Inv_Mask(input_size=(150,150,3),name='Spiral_Inverse2_1')
    Inv1_2=Inv_Mask(input_size=(150,150,3),name='Spiral_Inverse1_2')
    Inv2_2=Inv_Mask(input_size=(150,150,3),name='Spiral_Inverse2_2')
    
    inputs = Input(input_size)
    
    #R_DA_1_1 = tf.keras.layers.Lambda(DA_1_1, name="L_1_1")(inputs)
    RR_1_1_A   = Inv1_1(inputs)
    #RR_1_1   = tf.keras.layers.Lambda(DA_1_1, name="L_1_1_F")(RR_1_1_A)
    
    #R_DA_2_1 = tf.keras.layers.Lambda(DA_2_1, name="L_2_1")(inputs)
    RR_2_1_A   = Inv2_1(inputs)
    #RR_2_1   = tf.keras.layers.Lambda(DA_2_1, name="L_2_1_F")(RR_2_1_A)
    
    #R_DA_1_2 = tf.keras.layers.Lambda(DA_1_2, name="L_1_2")(inputs)
    RR_1_2_A   = Inv1_2(inputs)
    #RR_1_2   = tf.keras.layers.Lambda(DA_1_2, name="L_1_2_F")(RR_1_2_A)
    
    #R_DA_2_2 = tf.keras.layers.Lambda(DA_2_2, name="L_2_2")(inputs)
    RR_2_2_A   = Inv2_2(inputs)
    #RR_2_2   = tf.keras.layers.Lambda(DA_2_2, name="L_2_2_F")(RR_2_2_A)
    
    add1 =tf.keras.layers.add([RR_1_1_A, RR_2_1_A, RR_1_2_A, RR_2_2_A])
    # construct the CNN
    model = Model(inputs, add1,name=name)
    return model