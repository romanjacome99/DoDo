from tensorflow.keras.layers import Dropout, MaxPool2D,Conv2D,UpSampling2D,Concatenate,Add,BatchNormalization,Conv2DTranspose, Input, Layer, concatenate, Average, Activation
import tensorflow as tf
from decoder.core.layers import *
from tensorflow.keras.models import Model
import numpy as np

import math

def deep_prior_unet(X, Bands=20, Kernels_Size=(3, 3), num_filters=20, trainable=True, latent_space=None):
    conv_r1 = Conv2D(num_filters, Kernels_Size, padding="same", kernel_initializer='he_normal', activation=None,
                     trainable=trainable)(X)

    down1 = MaxPool2D(pool_size=(2, 2))(conv_r1)

    conv_r2 = Conv2D(num_filters, Kernels_Size, padding="same", kernel_initializer='he_normal', activation=None,
                     trainable=trainable)(down1)

    down2 = MaxPool2D(pool_size=(2, 2))(conv_r2)

    conv_r3 = Conv2D(num_filters, Kernels_Size, padding="same", kernel_initializer='he_normal', activation="relu",
                     trainable=trainable)(down2)

    down3 = MaxPool2D(pool_size=(2, 2))(conv_r3)

    latent_space_1 = Conv2D(num_filters, Kernels_Size, padding="same", kernel_initializer='he_normal',
                            activation="relu", trainable=trainable)(down3)  # latent space
    if latent_space is not None:
        latent_space_1 = Concatenate()([latent_space,latent_space_1])

    up1 = UpSampling2D(size=(2, 2))(latent_space_1)

    merge1 = Concatenate(axis=3)([up1, conv_r3])

    conv_r5 = Conv2D(num_filters, Kernels_Size, padding="same", kernel_initializer='he_normal', activation="relu",
                     trainable=trainable)(merge1)

    up2 = UpSampling2D(size=(2, 2))(conv_r5)

    merge2 = Concatenate(axis=3)([up2, conv_r2])

    conv_r6 = Conv2D(num_filters, Kernels_Size, padding="same", kernel_initializer='he_normal', activation="relu"
                     , trainable=trainable)(merge2)

    up3 = UpSampling2D((2, 2))(conv_r6)

    merge3 = Concatenate(axis=3)([up3, conv_r1])

    conv_r7 = Conv2D(Bands, Kernels_Size, padding="same", kernel_initializer='he_normal', activation="relu"
                     , trainable=trainable)(merge3)
    

    res_op = Add()([X, conv_r7])

    conv_r8 = Conv2D(Bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None,
                     trainable=trainable)(res_op)

    return conv_r8,latent_space_1


def prior_highres(X, Bands=20, Kernels_Size=(3, 3), num_filters=20, trainable=True):
    Up1 = UpSampling2D([2, 2])(X)

    conv_r1 = Conv2D(Bands, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(Up1)

    conv_r2 = Conv2D(2 * Bands, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r1)

    Down1 = MaxPool2D(pool_size=[2, 2])(conv_r2)

    conv_r3 = Conv2D(2 * Bands, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(Down1)

    conv_r4 = Conv2D(Bands, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(conv_r3)

    conv_r4 = BatchNormalization()(conv_r4)

    conv_r4 = Add()([X, conv_r4])

    conv_r5 = Conv2D(Bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r4)

    return conv_r5

def hssp_prior(input_size = (128,128,25),Kernels_Size=(3, 3), num_filters=20, trainable=True):
    X = Input(input_size)
    Bands = input_size[-1]

    conv_r1 = Conv2D(Bands, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(X)

    conv_r2 = Conv2D(Bands, (3, 3), padding="same", kernel_initializer='he_normal', activation='relu')(conv_r1)

    conv_r3 = Conv2D( Bands, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r2)

    conv_r4 = Add()([X, conv_r3])

    conv_r5 = Conv2D(Bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r4)
    model = Model(X,conv_r5)
    return model





def residual_block(X, Bands=20, Kernels_Size=(3, 3), num_filters=20):
    X0 = Conv2D(num_filters, Kernels_Size, padding='same', kernel_initializer='he_normal', activation='relu')(X)
    X1 = BatchNormalization()(X0)
    X2 = Conv2D(num_filters, Kernels_Size, padding='same', kernel_initializer='he_normal', activation='relu')(X1)
    X3 = Add()([X0, X2])
    X4 = BatchNormalization()(X3)
    return X4


def initialization_network(X, Bands=20, Kernels_Size=(3, 3), num_filters=20):
    X0 = Conv2D(num_filters, Kernels_Size, padding='same', kernel_initializer='he_normal', activation='relu')(X)
    X1 = residual_block(X0, Bands=Bands, Kernels_Size=Kernels_Size, num_filters=num_filters)
    X2 = residual_block(X1, Bands=Bands, Kernels_Size=Kernels_Size, num_filters=num_filters)
    X3 = residual_block(X2, Bands=Bands, Kernels_Size=Kernels_Size, num_filters=num_filters)
    return X3



class DecoderMiniBlock(Layer):
    
    def __init__(self, n_filters=32, name='DecoderBlock'):
        super(DecoderMiniBlock, self).__init__(name=name)
        self.n_filters=n_filters
        self.conv1=Conv2DTranspose(
                 self.n_filters,
                 (3,3),    # Kernel size
                 strides=(2,2),
                 padding='same')
                 
        self.conv2=Conv2D(self.n_filters, 
                 3,     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')  
        self.conv3=Conv2D(self.n_filters,
                 3,   # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')         
    def call(self, inputs):
        """
        Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
        merges the result with skip layer results from encoder block
        Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
        The function returns the decoded layer output
        """
        prev_layer_input=inputs[0]
        skip_layer_input=inputs[1]
        # Start with a transpose convolution layer to first increase the size of the image
        up = self.conv1(prev_layer_input)
    
        # Merge the skip connection from previous block to prevent information loss
        merge = concatenate([up, skip_layer_input], axis=3)
        
        # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
        # The parameters for the function are similar to encoder
        conv = self.conv2(merge)
        conv = self.conv3(conv)
        return conv
    
class EncoderMiniBlock(Layer):
    def __init__(self, n_filters=32, dropout_prob=0.0, max_pooling=True, name='DecoderBlock'):
        super(EncoderMiniBlock, self).__init__(name=name)
        self.n_filters=n_filters
        self.dropout_prob=dropout_prob
        self.max_pooling=max_pooling
        self.conv1 = Conv2D(self.n_filters, 
                      3,   # Kernel size   
                      activation='relu',
                      padding='same',
                      kernel_initializer='HeNormal')  
        self.conv2 = Conv2D(self.n_filters, 
                      3,   # Kernel size
                      activation='relu',
                      padding='same',
                      kernel_initializer='HeNormal')    
        self.BN = BatchNormalization()  
        self.MP =tf.keras.layers.MaxPooling2D(pool_size = (2,2))
        self.Dout =tf.keras.layers.Dropout(self.dropout_prob)
    def call(self, inputs):
        """
        This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning. 
        Dropout can be added for regularization to prevent overfitting. 
        The block returns the activation values for next layer along with a skip connection which will be used in the decoder
        """
        # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow 
        # Proper initialization prevents from the problem of exploding and vanishing gradients 
        # 'Same' padding will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size) 
        conv = self.conv1(inputs)
        conv = self.conv2(conv)
        
        # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
        conv = self.BN(conv, training=False)
    
        # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
        if self.dropout_prob > 0:     
            conv = self.Dout(conv)
    
        # Pooling reduces the size of the image while keeping the number of channels same
        # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
        # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
        if self.max_pooling:
            next_layer = self.MP(conv)    
        else:
            next_layer = conv
    
        # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions      
        skip_connection = conv
        
        return next_layer, skip_connection

class UNetCompiled(Layer):
    
    def __init__(self,input_size=(128, 128, 3), n_filters=32, n_classes=25, name='Unet'):
        super(UNetCompiled, self).__init__(name=name)
        self.n_filters=n_filters
        self.input_size=input_size
        self.n_classes=n_classes
        self.cblock1 = EncoderMiniBlock( self.n_filters,dropout_prob=0, max_pooling=True)   
        self.cblock2 = EncoderMiniBlock(self.n_filters*2,dropout_prob=0, max_pooling=False)
        self.ublock9 = DecoderMiniBlock(self.n_filters)
        self.conv9=Conv2D(self.n_filters,
                     3,
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal')
        self.conv10 = Conv2D(self.n_classes, 1, padding='same')            
        
        
    def call(self, inputs):

    
        cblock1 = self.cblock1(inputs)
        cblock2 = self.cblock2(cblock1[0])
    
        ublock9 = self.ublock9([cblock2[0], cblock1[1]])
        
        conv9 = self.conv9(ublock9)
        conv10 = self.conv10(conv9)
        
        
        return conv10




class Endmemebers_Layer(Layer):
    def __init__(self, L=16, rank=1, initicial=None, **kwargs):
        self.L = L
        self.rank = rank
        if initicial is not None:
            self.initicial = np.expand_dims(np.expand_dims(np.expand_dims(initicial, 0), 0), 0)
        else:
            self.initicial = np.random.uniform(0, 1, (1, 1, 1, self.rank, self.L))

        super(Endmemebers_Layer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'L': self.L,
            'rank': self.rank})
        return config

    def build(self, input_shape):

        Endmemb = tf.constant_initializer(self.initicial)
        self.Endmemb = self.add_weight(name='Endmemb', shape=(1, 1, 1, self.rank, self.L), initializer=Endmemb,
                                       constraint=tf.keras.constraints.NonNeg(), trainable=True)

    def call(self, inputs, **kwargs):
        # Hy = Rank_operator(inputs,self.Endmemb)
        Ab = tf.expand_dims(inputs, -1)
        Hy = tf.reduce_sum(tf.multiply(self.Endmemb, Ab), axis=3)
        return Hy



def Abund_conv_net(inputs,rank,name='',L=16,number_layer=4):
    conv1 = Conv2D(L, 3, activation='relu', padding='same')(inputs)
    for layer_v in range(number_layer):
        conv1 = Conv2D(L, 3, activation='relu', padding='same')(conv1)
    final = Conv2D(rank, 1, activation='softmax', padding='same', name='Abund_final'+str(name))(conv1)

    return final

def Abund_net_autoencoder(inputs,rank,name='',L=16,number_layer=4):

    decrese = math.floor(number_layer/2)
    conv1 = Conv2D(L, 3, activation='relu', padding='same')(inputs)
    for layer_v in range(number_layer-1):
        if (layer_v<decrese):
          conv1 = Conv2D(np.int((layer_v+2)*L), 3, activation='relu', padding='same')(conv1)
        else:
          conv1 = Conv2D(np.int((decrese*2-layer_v)*L), 3, activation='relu', padding='same')(conv1)
    final = Conv2D(rank, 1, activation='softmax', padding='same', name='Abund_final'+str(name))(conv1)

    return final

def Abund_net_residual(inputs,rank,name='',L=16,number_layer=4):
    conv1 = Conv2D(L, 3, activation='relu', padding='same')(inputs)
    conv2 = Conv2D(L, 3, activation='relu', padding='same')(conv1)
    for layer_v in range(number_layer-2):
        conv2 = Conv2D(L, 3, activation='relu', padding='same')(conv2)
    conv_concat = concatenate([conv1,conv2], axis=3)
    final = Conv2D(rank, 1, activation='softmax', padding='same', name='Abund_final'+str(name))(conv_concat)

    return final

def mix_prior(input_size = (150,150,1), L=102, rank = 8,Layer_depth=16,number_layer=4,initial=None):
    inputs = Input(input_size)
    drop1 = Dropout(0.2)(inputs)
    # block 1
    Ab = Abund_net_autoencoder(drop1,rank,name='1',L=Layer_depth,number_layer=number_layer)
    hy = Endmemebers_Layer(L=L, rank=rank,initicial=initial,name='estimation_1')(Ab)

    
    # block 1
    Ab2 = Abund_net_autoencoder(hy,rank,name='2',L=Layer_depth,number_layer=number_layer)
    hy2 = Endmemebers_Layer(L=L, rank=rank,initicial=initial,name='estimation_2')(Ab2)
    model = Model(inputs, hy2)
    return model




def FCAIDE(img_x,img_y):
    
    input_shape = (img_x,img_y,3)

    #with tf.device("/cpu:0"):
        # initialize the model
    input_layer = Input(shape=input_shape)
        
    layer_A = input_layer
    layer_B = input_layer
    layer_C = input_layer
    units = 25
    layer_A = NAIDE_Conv2D_Q1(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = NAIDE_Conv2D_E1(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = NAIDE_Conv2D_DOWN1(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg1 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg1 = Activation('relu')(layer_avg1)
    layer_avg1_1by1 = layer_avg1
    layer_avg1_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg1_1by1)
    layer_avg1_1by1 = Activation('relu')(layer_avg1_1by1)
    layer_avg1_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg1_1by1)
    layer_avg1_1by1 = Average()([layer_avg1, layer_avg1_1by1])
    layer_avg1_1by1 = Activation('relu')(layer_avg1_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg2 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg2 = Activation('relu')(layer_avg2)
    layer_avg2_1by1 = layer_avg2
    layer_avg2_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg2_1by1)
    layer_avg2_1by1 = Activation('relu')(layer_avg2_1by1)
    layer_avg2_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg2_1by1)
    layer_avg2_1by1 = Average()([layer_avg2, layer_avg2_1by1])
    layer_avg2_1by1 = Activation('relu')(layer_avg2_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg3 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg3 = Activation('relu')(layer_avg3)
    layer_avg3_1by1 = layer_avg3
    layer_avg3_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg3_1by1)
    layer_avg3_1by1 = Activation('relu')(layer_avg3_1by1)
    layer_avg3_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg3_1by1)
    layer_avg3_1by1 = Average()([layer_avg3, layer_avg3_1by1])
    layer_avg3_1by1 = Activation('relu')(layer_avg3_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg4 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg4 = Activation('relu')(layer_avg4)
    layer_avg4_1by1 = layer_avg4
    layer_avg4_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg4_1by1)
    layer_avg4_1by1 = Activation('relu')(layer_avg4_1by1)
    layer_avg4_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg4_1by1)
    layer_avg4_1by1 = Average()([layer_avg4, layer_avg4_1by1])
    layer_avg4_1by1 = Activation('relu')(layer_avg4_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg5 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg5 = Activation('relu')(layer_avg5)
    layer_avg5_1by1 = layer_avg5
    layer_avg5_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg5_1by1)
    layer_avg5_1by1 = Activation('relu')(layer_avg5_1by1)
    layer_avg5_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg5_1by1)
    layer_avg5_1by1 = Average()([layer_avg5, layer_avg5_1by1])
    layer_avg5_1by1 = Activation('relu')(layer_avg5_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg6 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg6 = Activation('relu')(layer_avg6)
    layer_avg6_1by1 = layer_avg6
    layer_avg6_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg6_1by1)
    layer_avg6_1by1 = Activation('relu')(layer_avg6_1by1)
    layer_avg6_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg6_1by1)
    layer_avg6_1by1 = Average()([layer_avg6, layer_avg6_1by1])
    layer_avg6_1by1 = Activation('relu')(layer_avg6_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg7 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg7 = Activation('relu')(layer_avg7)
    layer_avg7_1by1 = layer_avg7
    layer_avg7_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg7_1by1)
    layer_avg7_1by1 = Activation('relu')(layer_avg7_1by1)
    layer_avg7_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg7_1by1)
    layer_avg7_1by1 = Average()([layer_avg7, layer_avg7_1by1])
    layer_avg7_1by1 = Activation('relu')(layer_avg7_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg8 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg8 = Activation('relu')(layer_avg8)
    layer_avg8_1by1 = layer_avg8
    layer_avg8_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg8_1by1)
    layer_avg8_1by1 = Activation('relu')(layer_avg8_1by1)
    layer_avg8_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg8_1by1)
    layer_avg8_1by1 = Average()([layer_avg8, layer_avg8_1by1])
    layer_avg8_1by1 = Activation('relu')(layer_avg8_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg9 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg9 = Activation('relu')(layer_avg9)
    layer_avg9_1by1 = layer_avg9
    layer_avg9_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg9_1by1)
    layer_avg9_1by1 = Activation('relu')(layer_avg9_1by1)
    layer_avg9_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg9_1by1)
    layer_avg9_1by1 = Average()([layer_avg9, layer_avg9_1by1])
    layer_avg9_1by1 = Activation('relu')(layer_avg9_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg10 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg10 = Activation('relu')(layer_avg10)
    layer_avg10_1by1 = layer_avg10
    layer_avg10_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg10_1by1)
    layer_avg10_1by1 = Activation('relu')(layer_avg10_1by1)
    layer_avg10_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg10_1by1)
    layer_avg10_1by1 = Average()([layer_avg10, layer_avg10_1by1])
    layer_avg10_1by1 = Activation('relu')(layer_avg10_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg11 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg11 = Activation('relu')(layer_avg11)
    layer_avg11_1by1 = layer_avg11
    layer_avg11_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg11_1by1)
    layer_avg11_1by1 = Activation('relu')(layer_avg11_1by1)
    layer_avg11_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg11_1by1)
    layer_avg11_1by1 = Average()([layer_avg11, layer_avg11_1by1])
    layer_avg11_1by1 = Activation('relu')(layer_avg11_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg12 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg12 = Activation('relu')(layer_avg12)
    layer_avg12_1by1 = layer_avg12
    layer_avg12_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg12_1by1)
    layer_avg12_1by1 = Activation('relu')(layer_avg12_1by1)
    layer_avg12_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg12_1by1)
    layer_avg12_1by1 = Average()([layer_avg12, layer_avg12_1by1])
    layer_avg12_1by1 = Activation('relu')(layer_avg12_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg13 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg13 = Activation('relu')(layer_avg13)
    layer_avg13_1by1 = layer_avg13
    layer_avg13_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg13_1by1)
    layer_avg13_1by1 = Activation('relu')(layer_avg13_1by1)
    layer_avg13_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg13_1by1)
    layer_avg13_1by1 = Average()([layer_avg13, layer_avg13_1by1])
    layer_avg13_1by1 = Activation('relu')(layer_avg13_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg14 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg14 = Activation('relu')(layer_avg14)
    layer_avg14_1by1 = layer_avg14
    layer_avg14_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg14_1by1)
    layer_avg14_1by1 = Activation('relu')(layer_avg14_1by1)
    layer_avg14_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg14_1by1)
    layer_avg14_1by1 = Average()([layer_avg14, layer_avg14_1by1])
    layer_avg14_1by1 = Activation('relu')(layer_avg14_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg15 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg15 = Activation('relu')(layer_avg15)
    layer_avg15_1by1 = layer_avg15
    layer_avg15_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg15_1by1)
    layer_avg15_1by1 = Activation('relu')(layer_avg15_1by1)
    layer_avg15_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg15_1by1)
    layer_avg15_1by1 = Average()([layer_avg15, layer_avg15_1by1])
    layer_avg15_1by1 = Activation('relu')(layer_avg15_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg16 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg16 = Activation('relu')(layer_avg16)
    layer_avg16_1by1 = layer_avg16
    layer_avg16_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg16_1by1)
    layer_avg16_1by1 = Activation('relu')(layer_avg16_1by1)
    layer_avg16_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg16_1by1)
    layer_avg16_1by1 = Average()([layer_avg16, layer_avg16_1by1])
    layer_avg16_1by1 = Activation('relu')(layer_avg16_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg17 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg17 = Activation('relu')(layer_avg17)
    layer_avg17_1by1 = layer_avg17
    layer_avg17_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg17_1by1)
    layer_avg17_1by1 = Activation('relu')(layer_avg17_1by1)
    layer_avg17_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg17_1by1)
    layer_avg17_1by1 = Average()([layer_avg17, layer_avg17_1by1])
    layer_avg17_1by1 = Activation('relu')(layer_avg17_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg18 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg18 = Activation('relu')(layer_avg18)
    layer_avg18_1by1 = layer_avg18
    layer_avg18_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg18_1by1)
    layer_avg18_1by1 = Activation('relu')(layer_avg18_1by1)
    layer_avg18_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg18_1by1)
    layer_avg18_1by1 = Average()([layer_avg18, layer_avg18_1by1])
    layer_avg18_1by1 = Activation('relu')(layer_avg18_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg19 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg19 = Activation('relu')(layer_avg19)
    layer_avg19_1by1 = layer_avg19
    layer_avg19_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg19_1by1)
    layer_avg19_1by1 = Activation('relu')(layer_avg19_1by1)
    layer_avg19_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg19_1by1)
    layer_avg19_1by1 = Average()([layer_avg19, layer_avg19_1by1])
    layer_avg19_1by1 = Activation('relu')(layer_avg19_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg20 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg20 = Activation('relu')(layer_avg20)
    layer_avg20_1by1 = layer_avg20
    layer_avg20_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg20_1by1)
    layer_avg20_1by1 = Activation('relu')(layer_avg20_1by1)
    layer_avg20_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg20_1by1)
    layer_avg20_1by1 = Average()([layer_avg20, layer_avg20_1by1])
    layer_avg20_1by1 = Activation('relu')(layer_avg20_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg21 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg21 = Activation('relu')(layer_avg21)
    layer_avg21_1by1 = layer_avg21
    layer_avg21_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg21_1by1)
    layer_avg21_1by1 = Activation('relu')(layer_avg21_1by1)
    layer_avg21_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg21_1by1)
    layer_avg21_1by1 = Average()([layer_avg21, layer_avg21_1by1])
    layer_avg21_1by1 = Activation('relu')(layer_avg21_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg22 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg22 = Activation('relu')(layer_avg22)
    layer_avg22_1by1 = layer_avg22
    layer_avg22_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg22_1by1)
    layer_avg22_1by1 = Activation('relu')(layer_avg22_1by1)
    layer_avg22_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg22_1by1)
    layer_avg22_1by1 = Average()([layer_avg22, layer_avg22_1by1])
    layer_avg22_1by1 = Activation('relu')(layer_avg22_1by1)


    layer_ = Average()([layer_avg17_1by1, layer_avg16_1by1, layer_avg15_1by1, layer_avg14_1by1, 
                        layer_avg13_1by1, layer_avg12_1by1, layer_avg11_1by1, layer_avg10_1by1, 
                        layer_avg9_1by1, layer_avg8_1by1, layer_avg7_1by1, layer_avg6_1by1,
                        layer_avg5_1by1, layer_avg4_1by1, layer_avg3_1by1, layer_avg2_1by1,
                        layer_avg1_1by1, layer_avg18_1by1, layer_avg19_1by1, layer_avg20_1by1,
                        layer_avg21_1by1, layer_avg22_1by1])
    units_1by1 = 25
    layer_ = Conv2D(units_1by1,(1,1), kernel_initializer='he_uniform',)(layer_)
    layer_ = Activation('relu')(layer_)

    layer_residual = layer_
    layer_reresidual_1 = layer_
    layer_residual = Conv2D(units_1by1,(1,1), kernel_initializer='he_uniform',)(layer_residual)
    layer_residual = Activation('relu')(layer_residual)
    layer_residual = Conv2D(units_1by1,(1,1), kernel_initializer='he_uniform',)(layer_residual)
    layer_ = Average()([layer_, layer_residual])
    layer_ = Activation('relu')(layer_)

    layer_residual = layer_
    layer_reresidual_2 = layer_
    layer_residual = Conv2D(units_1by1,(1,1), kernel_initializer='he_uniform',)(layer_residual)
    layer_residual = Activation('relu')(layer_residual)
    layer_residual = Conv2D(units_1by1,(1,1), kernel_initializer='he_uniform',)(layer_residual)
    layer_ = Average()([layer_, layer_residual])
    layer_ = Activation('relu')(layer_)

    layer_ = Average()([layer_, layer_reresidual_1, layer_reresidual_2])
    layer_ = Conv2D(25,(1,1), kernel_initializer='he_uniform',)(layer_)

    output_layer = layer_
    model = Model(inputs=[input_layer], outputs=[output_layer])

    return model