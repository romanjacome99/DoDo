import tensorflow as K  # se puede cambiar por from keras.import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import MinMaxNorm
import numpy as np
import random
import math as m
import poppy
import os
from matplotlib import pyplot as plt
from scipy.io import loadmat


class DOE(Layer):

    def __init__(self, Mdoe=128, Mesce=128, wave_lengths=None, DOE_type='New', Trai=True, **kwargs):

        self.Mdoei = Mdoe
        self.Mesce = Mesce
        self.DOE_type = DOE_type
        if wave_lengths is not None:
            self.wave_lengths = wave_lengths
        else:
            self.wave_lengths = np.linspace(420, 660, 25) * 1e-9
        if DOE_type == 'New' or DOE_type == 'Zeros':
            self.Trai = Trai  # Trai
            self.zernike_volume = loadmat('optics/Base_zernike_128x128_nopadd.mat').get('HmBase').astype(np.float32)
        else:
            Hm_DOE = loadmat('optics/Spiral_128x128_nopadd.mat').get('Hm').astype(np.float32)
            self.Hm_DOE = Hm_DOE

            self.Trai = False
        self.P = loadmat('optics/Spiral_128x128_nopadd.mat').get('P').astype(np.float32)
        super(DOE, self).__init__()

    def build(self, input_shape):
        if self.DOE_type == 'New':
            num_zernike_coeffs = self.zernike_volume.shape[2]
            zernike_inits = np.zeros((1, 1, num_zernike_coeffs))
            # zernike_inits[0] = -1  # This sets the defocus value to approximately focus the image for a distance of 1m.
            zernike_inits[0, 0, 0] = random.random() * 2 - 1
            zernike_inits[0, 0, 1] = random.random() * 2 - 1
            zernike_inits[0, 0, 2] = random.random() * 2 - 1
            zernike_inits[0, 0, 3] = random.random() * 2 - 1
            zernike_inits[0, 0, 4] = random.random() * 2 - 1
            zernike_inits[0, 0, 5] = random.random() * 2 - 1
            zernike_inits[0, 0, 6] = random.random() * 2 - 1
            zernike_inits[0, 0, 7] = random.random() * 2 - 1
            zernike_inits[0, 0, 8] = random.random() * 2 - 1
            zernike_inits[0, 0, 9] = random.random() * 2 - 1
            zernike_inits[0, 0, 10] = random.random() * 2 - 1
            zernike_inits[0, 0, 11] = random.random() * 2 - 1
            zernike_initializer = K.constant_initializer(zernike_inits)
            self.zernike_coeffs = self.add_weight(name='zernike_coeffs', shape=zernike_inits.shape,
                                                  constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0,
                                                                        axis=2),
                                                  initializer=zernike_initializer, trainable=self.Trai)
            super(DOE, self).build(num_zernike_coeffs)
        if self.DOE_type == 'Zeros':
            num_zernike_coeffs = self.zernike_volume.shape[2]
            zernike_inits = np.zeros((1, 1, num_zernike_coeffs))
            zernike_inits[0, 0, 0:11] = 0
            zernike_initializer = K.constant_initializer(zernike_inits)
            self.zernike_coeffs = self.add_weight(name='zernike_coeffs', shape=zernike_inits.shape,
                                                  constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0,
                                                                        axis=2),
                                                  initializer=zernike_initializer, trainable=False)
            super(DOE, self).build(num_zernike_coeffs)

    def call(self, input, **kwargs):
        # Hm = Hm  # Learnable
        # Lambda = Lambda  # Input to construct
        Lambda = self.wave_lengths
        Mdoe = self.Mdoei
        Mesce = self.Mesce
        XX = K.linspace(-Mdoe // 2, Mdoe // 2, Mdoe)
        [x, y] = K.meshgrid(XX, XX)

        # max_val = K.reduce_max(x)
        # r = K.math.sqrt(x ** 2 + y ** 2)
        # P = K.cast(r < max_val, K.complex64)
        P = K.cast(self.P, K.complex64)
        if self.DOE_type == 'New' or self.DOE_type == 'Zeros':

            Hm = K.cast(K.reduce_sum(self.zernike_coeffs * self.zernike_volume, axis=2), K.complex64)
        else:
            Hm = K.cast(self.Hm_DOE, K.complex64)
        for NLam in range(25):
            IdLens = 1.5375 + 0.00829045 * (Lambda[NLam] * 1e6) ** (-2) - 0.000211046 * (Lambda[NLam] * 1e6) ** (-4)
            IdLens = IdLens - 1
            # Falta construir el Hm
            # PD = np.int32((Mesce-Mdoe)/2)
            # paddings = K.constant([[PD, PD], [PD, PD], [0, 0]])
            # Aux = K.expand_dims(K.math.multiply(P, K.math.exp(1j * (2 * m.pi / Lambda[NLam]) * IdLens * Hm)), 2)
            Aux = K.expand_dims(K.math.exp(1j * (2 * 10.0* m.pi / Lambda[NLam]) * IdLens * Hm), 2)
            # Aux = K.pad(Aux, paddings, "CONSTANT")
            if NLam > 0:
                P_DOE = K.concat([P_DOE, Aux], axis=2, name='stack')
            else:
                P_DOE = Aux
        #### OJO IMPROVISACION
        input2 = input[:, ::np.int32(Mesce / Mdoe), ::np.int32(Mesce / Mdoe), :]
        u2 = K.math.multiply(input2, P_DOE)

        return u2
        # return K.math.multiply(input, self.kernel)


class DOE_Free(Layer):

    def __init__(self, Mdoe=128, Mesce=128, Nterms=150, wave_lengths=None, DOE_type='New', Trai=True, **kwargs):
        print(Nterms)
        self.Mdoei = Mdoe
        self.Mesce = Mesce
        self.DOE_type = DOE_type
        self.Nterms = Nterms
        if wave_lengths is not None:
            self.wave_lengths = wave_lengths
        else:
            self.wave_lengths = np.linspace(420, 660, 25) * 1e-9
        if DOE_type == 'New' or DOE_type == 'Zeros':
            self.Trai = Trai  # Trai
            if not os.path.exists('optics/zernike_volume1_%d_Nterms_%d.npy' % (Mdoe, Nterms)):
                znew = 1e-6 * poppy.zernike.zernike_basis(nterms=Nterms, npix=self.Mdoei, outside=0.0)

                self.zernike_volume = znew

                np.save('optics/zernike_volume1_%d_Nterms_%d.npy' % (Mdoe, Nterms), self.zernike_volume)

            else:
                self.zernike_volume = np.load('optics/zernike_volume1_%d_Nterms_%d.npy' % (Mdoe, Nterms))
        else:
            Hm_DOE = loadmat('optics/Spiral_128x128_nopadd.mat').get('Hm').astype(np.float32)
            self.Hm_DOE = Hm_DOE

            self.Trai = False
        self.P = loadmat('optics/Spiral_128x128_nopadd.mat').get('P').astype(np.float32)
        super(DOE_Free, self).__init__()

    def build(self, input_shape):
        if self.DOE_type == 'New':
            num_zernike_coeffs = self.Nterms

            zernike_inits = np.zeros((num_zernike_coeffs, 1, 1))
            # zernike_inits[0] = -1  # This sets the defocus value to approximately focus the image for a distance of 1m.

            zernike_initializer = K.constant_initializer(zernike_inits)
            self.zernike_coeffs = self.add_weight(name='zernike_coeffs', shape=zernike_inits.shape,
                                                  constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0,
                                                                        axis=2),
                                                  initializer=zernike_initializer, trainable=self.Trai)
            super(DOE_Free, self).build(num_zernike_coeffs)
        if self.DOE_type == 'Zeros':
            num_zernike_coeffs = self.zernike_volume.shape[0]
            zernike_inits = np.zeros((num_zernike_coeffs, 1, 1))
            # zernike_inits[0] = -1  # This sets the defocus value to approximately focus the image for a distance of 1m.
            zernike_inits[0, 0, 0:11] = 0
            zernike_initializer = K.constant_initializer(zernike_inits)
            self.zernike_coeffs = self.add_weight(name='zernike_coeffs', shape=zernike_inits.shape,
                                                  constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0,
                                                                        axis=2),
                                                  initializer=zernike_initializer, trainable=False)
            super(DOE_Free, self).build(num_zernike_coeffs)

    def call(self, input, **kwargs):
        # Hm = Hm  # Learnable
        # Lambda = Lambda  # Input to construct
        Lambda = self.wave_lengths
        Mdoe = self.Mdoei
        Mesce = self.Mesce
        XX = K.linspace(-Mdoe // 2, Mdoe // 2, Mdoe)
        [x, y] = K.meshgrid(XX, XX)

        # max_val = K.reduce_max(x)
        # r = K.math.sqrt(x ** 2 + y ** 2)
        # P = K.cast(r < max_val, K.complex64)
        P = K.cast(self.P, K.complex64)
        if self.DOE_type == 'New' or self.DOE_type == 'Zeros':
            Hm = K.cast(K.reduce_sum(self.zernike_coeffs * self.zernike_volume, axis=0), K.complex64)
        else:
            Hm = K.cast(self.Hm_DOE, K.complex64)
        for NLam in range(25):
            IdLens = 1.5375 + 0.00829045 * (Lambda[NLam] * 1e6) ** (-2) - 0.000211046 * (Lambda[NLam] * 1e6) ** (-4)
            IdLens = IdLens - 1
            # Falta construir el Hm
            # PD = np.int32((Mesce-Mdoe)/2)
            # paddings = K.constant([[PD, PD], [PD, PD], [0, 0]])
            # Aux = K.expand_dims(K.math.multiply(P, K.math.exp(1j * (2 * m.pi / Lambda[NLam]) * IdLens * Hm)), 2)
            Aux = K.expand_dims(K.math.exp(1j * (2  * m.pi / Lambda[NLam]) * IdLens * Hm), 2)
            # Aux = K.pad(Aux, paddings, "CONSTANT")
            if NLam > 0:
                P_DOE = K.concat([P_DOE, Aux], axis=2, name='stack')
            else:
                P_DOE = Aux
        #### OJO IMPROVISACION
        input2 = input[:, ::np.int32(Mesce / Mdoe), ::np.int32(Mesce / Mdoe), :]
        u2 = K.math.multiply(input2, P_DOE)
        ###
        # u2 = K.math.multiply(input, P_DOE)
        # u2 = u2[:,np.int(Mesce / 2 - Mdoe / 2): np.int(Mesce / 2 + Mdoe / 2),
        #          np.int(Mesce / 2 - Mdoe / 2): np.int(Mesce / 2 + Mdoe / 2),:]
        return u2
        # return K.math.multiply(input, self.kernel)