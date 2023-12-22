
import numpy as np
from deep_prior_networks import *
from regularizers import *
from tensorflow.keras.constraints import NonNeg

class ForwardCASSI(Layer):
    def __init__(self, input_dim=(512, 512, 31), noise=False, HO=(0.25, 0.5, 0.25), reg_param=0.5, ds=0.5, opt_H=True,
                 name=False, type_reg='sig', sig_param=10, shots=1, batch_size=1,snr=30,**kwargs):
    
        self.input_dim = input_dim    
        self.noise = noise
        self.ds = ds
        self.sig_param = sig_param
        self.shots = shots
        self.snr = snr
        self.batch_size = batch_size
        self.type_reg = type_reg
        self.sig_param = sig_param
        self.HO = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(HO, 0), 0), -1), -1)
        if type_reg == 'sig':
            self.my_regularizer = Reg_Binary_Sigmoid(parameter_s=sig_param,parameter_r=reg_param)
        else:
            self.my_regularizer = Reg_Binary_0_1(reg_param)
        self.Md = int(input_dim[0]*ds)
        self.opt_H = opt_H
        super(ForwardCASSI, self).__init__(name=name, **kwargs)

    def build(self, input_shape):

        if self.opt_H:
            print('Trainable CASSI')

            H_init = np.random.normal(0, 0.1, (1, self.Md, self.Md, 1,
                                                            self.shots))
            H_init = tf.constant_initializer(H_init)
            self.H = self.add_weight(name='H', shape=(1, self.Md, self.Md, 1, self.shots),
                                     initializer=H_init, trainable=True, regularizer=self.my_regularizer)
        else:
            print('Non-Trainable CASSI')
            H_init = tf.constant_initializer(
                np.random.randint(0, 2, size=(1, self.Md, self.Md, 1, self.shots)))
            self.H = self.add_weight(name='H', shape=(1, self.Md, self.Md, 1, self.shots),
                                     initializer=H_init, trainable=False)

    def call(self, inputs, **kwargs):

        H = self.H
        if self.type_reg == 'sig':
            H = tf.math.sigmoid(H*self.sig_param)
        L = self.input_dim[2]
        Md = self.Md
        HO = self.HO
        X = tf.image.resize(inputs,[Md,Md])
        X = tf.expand_dims(X, -1)


        X = tf.reduce_sum(tf.nn.conv3d(X, HO, strides=[1, 1, 1, 1, 1], padding='SAME'), axis=-1)
        X = tf.expand_dims(X,-1)


        X = tf.multiply(H,X)

        X = tf.pad(X, [[0, 0], [0, 0], [0, L - 1], [0, 0], [0, 0]])
        y = None
        for i in range(L):
            Tempo = tf.roll(X, shift=i, axis=2)
            if y is not None:
                y = tf.concat([y, tf.expand_dims(Tempo[:, :, :, i], -1)], axis=4)
            else:
                y = tf.expand_dims(Tempo[:, :, :, i], -1)
        y = tf.reduce_sum(y, -1)
        y = y / tf.math.reduce_max(y)
        
        if self.noise:
            sigma = tf.reduce_sum(tf.math.pow(y, 2)) / ((Md * (Md + (L - 1)) * self.batch_size) * 10 ** (self.snr / 10))
            y = y + tf.random.normal(shape=(self.batch_size, Md, Md + L - 1, 1), mean=0, stddev=tf.math.sqrt(sigma),
                                 dtype=y.dtype)
        
        X = None
        for i in range(L):
            Tempo = tf.roll(y, shift=-i, axis=2)
            if X is not None:
                X = tf.concat([X, tf.expand_dims(Tempo[:, :, 0:Md], -1)], axis=4)
            else:
                X = tf.expand_dims(Tempo[:, :, 0:Md], -1)
        X = tf.transpose(X, [0, 1, 2, 4, 3])
        X2 = None
        for i in range(self.shots):
            Aux2 = tf.expand_dims(X[:, :, :, :, i], -1)
            if X2 is not None:
                X2 = tf.concat([X2, tf.expand_dims(tf.reduce_sum(
                    tf.nn.conv3d_transpose(Aux2, HO, [self.batch_size, Md, Md, L, 1], strides=[1, 1, 1, 1, 1], padding='SAME'),
                    axis=-1), -1)], 4)
            else:
                X2 = tf.expand_dims(tf.reduce_sum(
                    tf.nn.conv3d_transpose(Aux2, HO, [self.batch_size, Md, Md, L, 1], strides=[1, 1, 1, 1, 1], padding='SAME'),
                    axis=-1), -1)

        X2 = tf.multiply(H, X2)
        X2 = tf.reduce_sum(X2, 4)
        X = X2 / tf.math.reduce_max(X2)

        return X, y, H


    def get_config(self):
        return super().get_config()
    def get_H(self):
        if self.type_reg:
            return tf.math.sigmoid(self.sig_param*self.H)
        else:
            return self.H
    
    def get_sig_param(self):
        return self.sig_param

    def update_sig_param(self,new_param):
        self.sig_param.assign(new_param)

    def randomize_H(self):
        rand_v = tf.random.normal(shape=self.H.shape,mean=0,stddev=0.5,dtype=self.H.dtype)
        self.H.assign(self.H+rand_v)


class ForwardMCFA(Layer):
    def __init__(self, input_dim=(512, 512, 31), noise=False, reg_param=0.5, dl=0.5, opt_H=True,
                 name=False, type_reg='sig', sig_param=10, shots=1, batch_size=1,snr=30,**kwargs):
        self.input_dim = input_dim
        self.shots = shots
        self.batch_size = batch_size
        self.noise = noise
        self.dl = dl
        self.Ld = int(self.input_dim[-1] * dl)
        self.opt_H = opt_H
        self.snr = snr
        self.type_reg = type_reg
        self.reg_param = reg_param
        self.sig_param = sig_param
        if type_reg == 'sig':
            self.my_regularizer = Reg_Binary_Sigmoid(parameter_s=sig_param,parameter_r=reg_param)
        else:
            self.my_regularizer = Reg_Binary_0_1(reg_param)

        super(ForwardMCFA, self).__init__(name=name, **kwargs)

    def build(self, input_shape):

        if self.opt_H:
            print('Trainable MCFA')
            H_init = tf.constant_initializer(np.random.normal(0, 0.1, (1, self.input_dim[0], self.input_dim[0], self.Ld,
                                                            self.shots)))
            self.H = self.add_weight(name='H',
                                     shape=(1, self.input_dim[0], self.input_dim[0], self.Ld, self.shots),
                                     initializer=H_init, trainable=True, regularizer=self.my_regularizer)
        else:
            print('Non-Trainable MCFA')

            H_init = tf.constant_initializer(
                np.random.randint(0, 2, size=(1, self.input_dim[0], self.input_dim[0], self.Ld,
                                              self.shots)))

            self.H = self.add_weight(name='H',
                                     shape=(1, self.input_dim[0], self.input_dim[0], self.Ld, self.shots),
                                     initializer=H_init, trainable=False)

    def call(self, inputs, **kwargs):

        K = self.shots
        H = self.H
        if self.type_reg == 'sig':
            H = tf.math.sigmoid(H*self.sig_param)
        L = self.input_dim[2]
        Ld = self.Ld
        M = self.input_dim[0]
        q = int(1/self.dl)
        kernel = np.zeros((1,1,L,L//q))
        for i in range(0,L//q):
            kernel[0,0,i*q:(i+1)*(q),i] = 1/q

        input_im = tf.expand_dims(tf.nn.conv2d(inputs,kernel,strides=[1,1,1,1],padding='SAME'),-1)
    
        y = tf.expand_dims(tf.reduce_sum(tf.multiply(H, input_im), -2), -1)
      
        if self.noise:
            sigma = tf.reduce_sum(tf.math.pow(y, 2)) / ((M * M  * self.batch_size) * 10 ** (self.snr / 10))
            y = y + tf.random.normal(shape=(self.batch_size, M, M , 1, 1), mean=0, stddev=tf.math.sqrt(sigma),
                                    dtype=y.dtype)

        X = None
        for _ in range(Ld):
            if X is not None:
                X = tf.concat([X, y], 4)
            else:
                X = y
        X = tf.transpose(X, [0, 1, 2, 4, 3])

        X = tf.multiply(H, X)
        X = tf.reduce_sum(X, 4)
        X = X / tf.math.reduce_max(X)

        return X, y, H     
      

    def get_H(self):
        if self.type_reg:
            return tf.math.sigmoid(self.sig_param*self.H)
        else:
            return self.H
    
    def get_sig_param(self):
        return self.sig_param

    def update_sig_param(self,new_param):
        self.sig_param.assign(new_param)

    def randomize_H(self):
        rand_v = tf.random.normal(shape=self.H.shape,mean=0,stddev=0.5,dtype=self.H.dtype)
        self.H.assign(self.H+rand_v)
        
    def get_config(self):
        return super().get_config()
