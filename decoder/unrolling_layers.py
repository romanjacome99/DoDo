
import numpy as np
from optics.Forward_DM_Spiral128 import Forward_DM_Spiral
from scipy.io import loadmat
from tensorflow.keras.constraints import NonNeg
import tensorflow as K  # se puede cambiar por from keras.import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from decoder.deep_prior_networks import *

class HQS_Update_DoDo(Layer):
    def __init__(self, input_size=(128,128, 25), name='HQS_update',rho_initial= 0.1,alpha_initial=0.1,mode='sthocastic',prior = 'hssp',transpose='unet', shots=4,**kwargs):
        super(HQS_Update_DoDo, self).__init__(name = name)
        self.input_size = input_size
        if prior == 'unet':
            self.prior = hssp_prior(input_size = input_size,Kernels_Size=(3, 3), num_filters=20, trainable=True)
        elif prior == 'mix':
            self.prior = mix_prior(input_size = input_size, L=input_size[-1], rank = kwargs['rank'],Layer_depth=kwargs['Layer_depth'],number_layer=kwargs['number_layer'])
        self.rho_initial = rho_initial
        self.alpha_initial = alpha_initial
        self.Grad = Gradient_DoDo(input_size=(128,128, 25),mode=mode,shots=shots,transpose=transpose)
    def build(self, input_shape):
            rho_init = tf.keras.initializers.Constant(self.rho_initial)
            self.rho = self.add_weight(name='alpha',trainable=True, constraint=NonNeg(),initializer=rho_init)

            alpha_init = tf.keras.initializers.Constant(self.alpha_initial)
            self.alpha = self.add_weight(name='alpha',trainable=True, constraint=NonNeg(),initializer=alpha_init)
            super(HQS_Update_DoDo, self).build(input_shape)

    def call(self, inputs):           
        [X,y,F,self.A] = inputs
        Xn = X - self.alpha*(self.Grad([X,y,F,self.A]) + self.rho*(X-self.prior(X)))
        return Xn


class Gradient_DoDo(Layer):
    def __init__(self, input_size=(128,128, 25), mode='sthocastic',shots=4,name='Grad_Spiral',transpose='unet',**kwargs):
        super(Gradient_DoDo, self).__init__(name=name,**kwargs)
        self.input_size = input_size
        self.shots = shots
        self.mode = mode
       
    def build(self, input_shape):
            
            super(Gradient_DoDo, self).build(input_shape)

    def call(self, inputs):
           
        [X,y,F,A] = inputs
        self.A = A
        if self.mode == 'all':
            yk = None        
            for i in range(self.shots):

                ytemp = F[i](X)
                if yk is not None:
                    yk = concatenate([yk,ytemp],axis=-1)
                else:
                    yk = ytemp
        else:

            yk = F(X)

        res = yk - y
        
        Xk = self.A(res)
        return Xk


