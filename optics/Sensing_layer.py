import tensorflow as K  # se puede cambiar por from keras.import backend as K
from tensorflow.keras.layers import Layer
import numpy as np
from scipy.io import loadmat
import math as m

import os


class Sensing(Layer):

    def __init__(self, Ms=1000, wave_lengths=None, bgr_response=None, Trai=False, **kwargs):
        self.M = Ms
        '''
        if bgr_response is not None:
            self.bgr_response = K.cast(bgr_response, dtype=K.float32)
        else:
            self.R = K.cast(loadmat('Sensor_25_new3.mat').get('R'), K.float32)
            self.G = 1*K.cast(loadmat('Sensor_25_new3.mat').get('G'), K.float32)
            self.B = K.cast(loadmat('Sensor_25_new3.mat').get('B'), K.float32)
        '''    
        self.R = K.cast(loadmat('optics/Sensor_25_new3.mat').get('R'), K.float32)
        self.G = 1*K.cast(loadmat('optics/Sensor_25_new3.mat').get('G'), K.float32)
        self.B = K.cast(loadmat('optics/Sensor_25_new3.mat').get('B'), K.float32)    
        super(Sensing, self).__init__()

    def build(self, input_shape):
        super(Sensing, self).build(input_shape)

    def call(self, input, **kwargs):
        Kernel = np.ones((1, 3, 3,1))
        for NLam in range(25):
            if NLam > 0:
                y_med_r = y_med_r + K.math.abs(input[:, :, :, NLam]) * self.R[0, NLam]
                y_med_g = y_med_g + K.math.abs(input[:, :, :, NLam]) * self.G[0, NLam]
                y_med_b = y_med_b + K.math.abs(input[:, :, :, NLam]) * self.B[0, NLam]
            else:
                y_med_r = K.math.abs(input[:, :, :, NLam]) * self.R[0, NLam]
                y_med_g = K.math.abs(input[:, :, :, NLam]) * self.G[0, NLam]
                y_med_b = K.math.abs(input[:, :, :, NLam]) * self.B[0, NLam]
        y_med_r = K.expand_dims(y_med_r, 3)
        y_med_g = K.expand_dims(y_med_g, 3)
        y_med_b = K.expand_dims(y_med_b, 3)
        '''
        y_med_r = K.nn.conv2d(K.concat([y_med_r,y_med_r,y_med_r],axis=3), Kernel, strides=[1, 1, 1, 1],padding='SAME')
        y_med_g = K.nn.conv2d(K.concat([y_med_g,y_med_g,y_med_g],axis=3), Kernel, strides=[1, 1, 1, 1],padding='SAME')
        y_med_b = K.nn.conv2d(K.concat([y_med_b,y_med_b,y_med_b],axis=3), Kernel, strides=[1, 1, 1, 1],padding='SAME')

        y_med_r = K.nn.conv2d(K.concat([y_med_r,y_med_r,y_med_r],axis=3), Kernel, strides=[1, 1, 1, 1],padding='SAME')
        y_med_g = K.nn.conv2d(K.concat([y_med_g,y_med_g,y_med_g],axis=3), Kernel, strides=[1, 1, 1, 1],padding='SAME')
        y_med_b = K.nn.conv2d(K.concat([y_med_b,y_med_b,y_med_b],axis=3), Kernel, strides=[1, 1, 1, 1],padding='SAME')

        y_final = K.concat([y_med_r, y_med_g, y_med_b], axis=3)
        #y_final = K.concat([K.expand_dims(y_med_r, 3), K.expand_dims(y_med_g, 3), K.expand_dims(y_med_b, 3)], axis=3)

        #y_final = K.nn.conv2d(y_final, Kernel, strides=[1, 1, 1, 1], padding='SAME')
        #Kernel = np.zeros((1,150,150,1))
        #Kernel[0,70:80,70:80,0] = 1
        #Kernel =  K.cast(Kernel,K.complex64)
        #KF = K.signal.fft2d(Kernel)
        #Y_finalF = K.signal.fft2d(K.cast(y_final,K.complex64))
        #y_final = K.cast(K.math.abs(K.signal.ifft2d(K.math.multiply(Y_finalF, KF))),K.float32)
        #y_final = K.math.square(K.concat([K.expand_dims(y_med_r, 3), K.expand_dims(y_med_g, 3), K.expand_dims(y_med_b, 3)], axis=3))
        '''
        y_final = K.concat([y_med_r, y_med_g, y_med_b], axis=3)
        y_final = y_final / K.reduce_max(y_final)
        ## falta el cuadrado
        return y_final