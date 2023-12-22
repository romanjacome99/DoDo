import tensorflow as K  # se puede cambiar por from keras.import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import NonNeg
import numpy as np
import math as m
from matplotlib import pyplot as plt

class Propagation(Layer):

    def __init__(self, Mp=300, L=1, wave_lengths=None, zi=2, Trai=True, **kwargs):

        # self.z = z
        self.Mpi = Mp
        #self.Mdoei = Mdoe
        self.Li = L
        self.zi = zi
        self.Trai = Trai
        #self.r = Mdoe/Mp
        if wave_lengths is not None:
            self.wave_lengths = wave_lengths
        else:
            self.wave_lengths = np.linspace(420, 660, 25) * 1e-9

        super(Propagation, self).__init__()

    def build(self, input_shape):
        initializerC = K.constant_initializer(self.zi)
        self.z = self.add_weight("Distance", shape=[1], constraint=NonNeg(), initializer=initializerC, trainable=self.Trai)
        super(Propagation, self).build(input_shape)

    def call(self, input, **kwargs):
        L = self.Li
        Mp = self.Mpi
        Lambda = self.wave_lengths
        dx = L / Mp
        Ns = np.int(L * 2 / (2 * dx))
        # This need to be do it for all spectral bands
        fx = K.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / L, Ns)
        [FFx, FFy] = K.meshgrid(fx, fx)
        #H = K.zeros([Mp, Mp, 25])
        for NLam in range(25):
            Aux = -1j * m.pi * Lambda[NLam] * K.cast(self.z, K.complex64)
            Aux2 = K.cast(FFx ** 2 + FFy ** 2, K.complex64)
            #Ha = K.math.multiply(K.math.exp(1j * K.cast(self.z, K.complex64) * (2 * m.pi / Lambda[NLam])), K.math.exp(Aux * Aux2))
            Ha=K.math.exp(Aux * Aux2)
            Ha = K.expand_dims(K.signal.fftshift(Ha, axes=[0,1]), 2)
            #H[:,:, NLam] = Ha
            if NLam > 0 :
                H = K.concat([H, Ha], axis=2, name='stack')
            else:
                H = Ha
        Aux3 = K.signal.fftshift(K.cast(input, K.complex64),axes=[1,2])
        '''        
        
        u1f = K.signal.fft2d(Aux3)
        H = K.expand_dims(H, 0)
        u2f = K.math.multiply(u1f, H)
        u2 = K.signal.ifftshift(K.signal.ifft2d(u2f),axes=[1,2])
        '''
        '''
        for NLam in range(25):
            D=K.signal.fft2d(Aux3[:,:,:,NLam])
            D=K.expand_dims(D, axis=2)
            if NLam > 0 :
                u1f = K.concat([u1f, D], axis=2, name='stack')
            else:
                u1f = D
            #
        u1f=Permute((0,1,3,2),u1f)
        '''
        
        u1f = K.signal.fft2d(K.transpose(Aux3,(0,3,1,2)))
        u1f=K.transpose(u1f,(0,2,3,1))
        H = K.expand_dims(H, 0)
        u2f = K.math.multiply(u1f, H)
        u2 = K.transpose(K.signal.ifftshift(K.signal.ifft2d(K.transpose(u2f,(0,3,1,2))),axes=[2,3]),(0,2,3,1))
       
        #Mdo = K.floor(Mp * r)
        #if r < 1:
         #   u2 = u2[np.int(Mp / 2 - Mdo / 2): np.int(Mp / 2 + Mdo / 2),
          #       np.int(Mp / 2 - Mdo / 2): np.int(Mp / 2 + Mdo / 2)]

        return u2
        # return K.math.multiply(input, self.kernel)
'''

class PropagationABS(Layer):

    def __init__(self, Mp=5000, Mdoe=1000, L=0.15, wave_lengths=None, zi=2, Trai=True, **kwargs):

        # self.z = z
        self.Mpi = Mp
        self.Mdoei = Mdoe
        self.Li = L
        self.zi = zi
        self.Trai = Trai
        if wave_lengths is not None:
            self.wave_lengths = wave_lengths
        else:
            self.wave_lengths = np.linspace(420, 660, 25) * 1e-9

        super(PropagationABS, self).__init__(**kwargs)

    def build(self, input_shape):
        initializerC = K.constant_initializer(self.zi)
        self.z = self.add_weight("Distance", shape=[1], initializer=initializerC, constraint=NonNeg(), trainable=self.Trai)
        super(PropagationABS, self).build(input_shape)

    def call(self, input, **kwargs):
        L = self.Li
        Mp = self.Mpi
        Mdo = self.Mdoei
        Lambda = self.wave_lengths
        dx = L / Mp
        Ns = np.int(L * 2 / (2 * dx))
        # This need to be do it for all spectral bands
        fx = K.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / L, Ns)
        [FFx, FFy] = K.meshgrid(fx, fx)
        #H = K.zeros([Mp, Mp, 25])
        for NLam in range(25):
            Aux = -1j * m.pi * Lambda[NLam] * K.cast(self.z, K.complex64)
            Aux2 = K.cast(FFx ** 2 + FFy ** 2, K.complex64)
            Ha = K.math.exp(Aux * Aux2)
            Ha = K.expand_dims(K.signal.fftshift(Ha), 2)
            #H[:,:, NLam] = Ha
            if NLam > 0 :
                H = K.concat([H, Ha], axis=2, name='stack')
            else:
                H = Ha
        Aux3 = K.signal.fftshift(K.cast(input, K.complex64))
        u1f = K.signal.fft2d(Aux3)
        H = K.expand_dims(H, 0)
        u2f = K.math.multiply(u1f, H)
        u2 = K.math.abs(K.signal.ifftshift(K.signal.ifft2d(u2f)))
        if Mp > Mdo:
            u2 = u2[np.int(Mp / 2 - Mdo / 2): np.int(Mp / 2 + Mdo / 2),
                 np.int(Mp / 2 - Mdo / 2): np.int(Mp / 2 + Mdo / 2)]

        return u2
        # return K.math.multiply(input, self.kernel)

'''
'''
class Layer_H(Layer):
    def __init__(self):
        super(Layer_H, self).__init__()
        # self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=input_shape[1:4], initializer="random_normal", trainable=True)

    def call(self, input):
        return K.math.multiply(input, self.kernel)
'''