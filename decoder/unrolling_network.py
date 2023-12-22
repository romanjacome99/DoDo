
from cv2 import transpose
from decoder.unrolling_layers import *
from optics.Forward_DM_Spiral128 import *
from tensorflow.keras.models import Model
from math import ceil,floor
from utils.regularizers import *
import random

class Unrolling(tf.keras.Model):
    def __init__(self,input_dim=(128,128,25),shots=4,mode='sthocastic',stages=12,prior='unet',transpose='unet',Train_c=True,**kwargs):
        super(Unrolling,self).__init__()
        self.input_dim = input_dim
        self.shots  = shots
        self.mode = mode
        self.stages = stages 
        self.prior = prior
        self.transpose = transpose
        self.Train_c = Train_c
        self.F = []
        self.y = []
    
        for i in range(1,self.shots+1):
            if i!=4:
                self.F.append(Forward_DM_Spiral(input_size=input_dim, DOE_typeA='New', name='Foward_Model'+str(i),Train_c=Train_c))
            if i==4:    
                self.F.append(Forward_DM_Spiral(input_size=input_dim, DOE_typeA='New', name='Foward_Model'+str(i),Train_c=Train_c)) 
        if self.shots ==0:     
            i=0
            self.F.append(Forward_DM_Spiral(input_size=input_dim, DOE_typeA='Zeros', name='Foward_Model'+str(i),Train_c=False))
            self.shots = 1



        self.A =[]
        for i in range(self.shots):
            if transpose == 'unet':
                self.A.append(UNetCompiled(input_size=[input_dim[0],input_dim[1],3],n_filters=32,n_classes=input_dim[-1]))
    
        self.unr_layers = []
        for k in range(stages):
                
            self.unr_layers.append(HQS_Update_DoDo(input_size=input_dim,name='HQS_update_'+str(k),rho_initial=0.01,alpha_initial=0.01,prior=prior,**kwargs))
                
    def call(self,inputs):

        y = []
        for i in range(1,self.shots+1):

            y.append(self.F[i-1](inputs))   
                
        idx = 0
        Xt = []
        X = self.A[0](y[0])
        Xt.append(X)

        for k in range(self.stages):
        
            if idx>self.shots-1:
                idx = 0
            X = self.unr_layers[k]([X,y[idx],self.F[idx],self.A[idx]])
                
            X = X/tf.reduce_max(X)
            idx +=1
            Xt.append(X)
        return Xt


class Unrolling_Free(tf.keras.Model):
    def __init__(self, input_dim=(128, 128, 25), shots=4, Nterms = 150,mode='sthocastic', stages=12, prior='unet', transpose='unet',
                 Train_c=True, **kwargs):
        super(Unrolling_Free, self).__init__()
        self.input_dim = input_dim
        self.shots = shots
        self.mode = mode
        self.stages = stages
        self.prior = prior
        self.transpose = transpose
        self.Train_c = Train_c
        self.F = []
        self.y = []

        for i in range(1, self.shots + 1):
            if i != 4:
                self.F.append(Forward_DM_Spiral_Free(input_size=input_dim, DOE_typeA='New', name='Foward_Model' + str(i),
                                                Train_c=Train_c, Nterms=Nterms))
            if i == 4:
                self.F.append(Forward_DM_Spiral_Free(input_size=input_dim, DOE_typeA='New', name='Foward_Model' + str(i),
                                                Train_c=Train_c, Nterms=Nterms))
        if self.shots == 0:
            i = 0
            self.F.append(
                Forward_DM_Spiral(input_size=input_dim, DOE_typeA='Zeros', name='Foward_Model' + str(i), Train_c=False))
            self.shots = 1

        self.A = []
        for i in range(self.shots):
            if transpose == 'unet':
                self.A.append(
                    UNetCompiled(input_size=[input_dim[0], input_dim[1], 3], n_filters=32, n_classes=input_dim[-1]))

        self.unr_layers = []
        for k in range(stages):
            self.unr_layers.append(
                HQS_Update_DoDo(input_size=input_dim, name='HQS_update_' + str(k), rho_initial=0.01, alpha_initial=0.01,
                                prior=prior, **kwargs))

    def call(self, inputs):

        y = []
        for i in range(1, self.shots + 1):
            y.append(self.F[i - 1](inputs))

        idx = 0
        Xt = []
        X = self.A[0](y[0])
        Xt.append(X)

        for k in range(self.stages):

            if idx > self.shots - 1:
                idx = 0
            X = self.unr_layers[k]([X, y[idx], self.F[idx], self.A[idx]])

            X = X / tf.reduce_max(X)
            idx += 1
            Xt.append(X)
        return Xt



class Unrolling_TL(tf.keras.Model):
    def __init__(self,input_dim=(128,128,25),shots=4,mode='sthocastic',stages=12,prior='unet',transpose='unet',Train_c=True,**kwargs):
        super(Unrolling_TL,self).__init__()
        self.input_dim = input_dim
        self.shots  = shots
        self.mode = mode
        self.stages = stages 
        self.prior = prior
        self.transpose = transpose
        self.Train_c = Train_c
        self.F = []
        self.y = []
    
        for i in range(1,self.shots+1):
            if i == self.shots:
                self.F.append(Forward_DM_Spiral(input_size=input_dim, DOE_typeA='New', name='Foward_Model'+str(i),Train_c=True))
            else:
                self.F.append(Forward_DM_Spiral(input_size=input_dim, DOE_typeA='New', name='Foward_Model'+str(i),Train_c=False))
            
            
        if self.shots ==0:     
            i=0
            self.F.append(Forward_DM_Spiral(input_size=input_dim, DOE_typeA='Zeros', name='Foward_Model'+str(i),Train_c=False))
            self.shots = 1



        self.A =[]
        for i in range(self.shots):
            if transpose == 'unet':
                self.A.append(UNetCompiled(input_size=[input_dim[0],input_dim[1],3],n_filters=32,n_classes=input_dim[-1]))
    
        self.unr_layers = []
        for k in range(stages):
                
            self.unr_layers.append(HQS_Update_DoDo(input_size=input_dim,name='HQS_update_'+str(k),rho_initial=0.01,alpha_initial=0.01,prior=prior,**kwargs))
                
    def call(self,inputs):

        y = []
        for i in range(1,self.shots+1):

            y.append(self.F[i-1](inputs))   
                
        idx = 0
        Xt = []
        X = self.A[0](y[0])
        Xt.append(X)

        for k in range(self.stages):
        
            if idx>self.shots-1:
                idx = 0
            X = self.unr_layers[k]([X,y[idx],self.F[idx],self.A[idx]])
                
            X = X/tf.reduce_max(X)
            idx +=1
            Xt.append(X)
        return Xt


class Unrolling_TL_Free(tf.keras.Model):
    def __init__(self, input_dim=(128, 128, 25), shots=4, Nterms=15,mode='sthocastic', stages=12, prior='unet', transpose='unet',
                 Train_c=True, **kwargs):
        super(Unrolling_TL_Free, self).__init__()
        self.input_dim = input_dim
        self.shots = shots
        self.mode = mode
        self.stages = stages
        self.prior = prior
        self.transpose = transpose
        self.Train_c = Train_c
        self.F = []
        self.y = []

        for i in range(1, self.shots + 1):
            if i == self.shots:
                self.F.append(Forward_DM_Spiral_Free(input_size=input_dim, DOE_typeA='New', Nterms=Nterms,name='Foward_Model' + str(i),
                                                Train_c=True))
            else:
                self.F.append(Forward_DM_Spiral_Free(input_size=input_dim, DOE_typeA='New',Nterms =Nterms, name='Foward_Model' + str(i),
                                                Train_c=False))

        if self.shots == 0:
            i = 0
            self.F.append(
                Forward_DM_Spiral(input_size=input_dim, DOE_typeA='Zeros', name='Foward_Model' + str(i), Train_c=False))
            self.shots = 1

        self.A = []
        for i in range(self.shots):
            if transpose == 'unet':
                self.A.append(
                    UNetCompiled(input_size=[input_dim[0], input_dim[1], 3], n_filters=32, n_classes=input_dim[-1]))

        self.unr_layers = []
        for k in range(stages):
            self.unr_layers.append(
                HQS_Update_DoDo(input_size=input_dim, name='HQS_update_' + str(k), rho_initial=0.01, alpha_initial=0.01,
                                prior=prior, **kwargs))

    def call(self, inputs):

        y = []
        for i in range(1, self.shots + 1):
            y.append(self.F[i - 1](inputs))

        idx = 0
        Xt = []
        X = self.A[0](y[0])
        Xt.append(X)

        for k in range(self.stages):

            if idx > self.shots - 1:
                idx = 0
            X = self.unr_layers[k]([X, y[idx], self.F[idx], self.A[idx]])

            X = X / tf.reduce_max(X)
            idx += 1
            Xt.append(X)
        return Xt


def UnrollingNetwork(input_size=(128,128,25),shots=4,mode='sthocastic',stages=12,prior='unet',transpose='unet',Train_c=True,reg=True,reg_param=1.5,**kwargs):
    print(shots)
    inputs = Input(input_size)
    F = []
    y = []
        
    for i in range(1,shots+1):
        if i!=4:
            F.append(Forward_DM_Spiral(input_size=input_size, DOE_typeA='New', name='Foward_Model'+str(i),Train_c=False))
        if i==4:    
            F.append(Forward_DM_Spiral(input_size=input_size, DOE_typeA='New', name='Foward_Model'+str(i),Train_c=False))
            
        y.append(F[i-1](inputs))    
    if shots ==0:     
        i=0
        F.append(Forward_DM_Spiral(input_size=input_size, DOE_typeA='Zeros', name='Foward_Model'+str(i),Train_c=False))
        y.append(F[0](inputs))
        shots=1

    Xt = []
    if transpose == 'unet':
            A =[]
            for i in range(shots):
                A.append(UNetCompiled(input_size=[input_size[0],input_size[1],3],n_filters=32,n_classes=input_size[-1]))
    elif transpose == 'fcaide':
            A = []
            #print('check')
            for i in range(shots):
                A.append(FCAIDE(img_x=input_size[0],img_y = input_size[1]))
    v = ceil(stages/shots)
    Xt = []
    print(A)
    X = A[0](y[0])
    Xt.append(X)
    idx = 0
    for k in range(stages):
        
        #idx = random.randint(0,3)
        if idx>shots-1:
            idx = 0
        print(idx)
        X = HQS_Update_DoDo(input_size=input_size,name='HQS_update_'+str(k),rho_initial=0.01,alpha_initial=0.01,prior=prior,**kwargs)([X,y[idx],F[idx],A[idx]])
            
            
        idx +=1
        Xt.append(X)
    

    model = Model(inputs,Xt)


    if reg:
        correlation_regularizer(y,reg_param,model)

        

    return model
