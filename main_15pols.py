from operator import truediv
import os
from tabnanny import verbose 
import tensorflow as tf
import scipy.io as sio
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=21000)])


            # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          #  tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
        print(e)
from dataset.Read_Spectral import *
from dataset.Read_Spectral_train import *
from utils.callbacks_2 import *
from decoder.unrolling_network import *
from utils.loss_metrics import *
from dataset.data import *

def transfer_weights(model,model_prev,shots,stages):
    t = -1
    for i in range(shots):
        t+=1
        if t >=shots-1:
            t-=1
        model.layers[i].set_weights(model_prev.layers[t].get_weights())
    for i in range(shots,2*shots-1):
        model.layers[i].set_weights(model_prev.layers[i-1].get_weights())
        

    for i in range(stages):
        
        model.get_layer('HQS_update_'+str(i)).set_weights(model_prev.get_layer('HQS_update_'+str(i)).get_weights())

    return model
    


# parameters of the net
BATCH_SIZE = 8
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_WIDTH_o = 128
IMG_HEIGHT_o = 128
L_bands = 25
L_imput = 25
shots = 4
dataset_path = 'C:\Roman\datasets\Arad1k\Arad'
dataset = 'arad'
train_dataset, val_dataset, test_dataset = load_dataset(dataset, dataset_path, batch_size=BATCH_SIZE,
                                                            input_size=(IMG_HEIGHT,IMG_HEIGHT,L_bands))

image_train = 'C:\Roman\datasets\Arad1k\Arad/train.h5'
image_val = 'C:\Roman\datasets\Arad1k\Arad/test.h5'

for shots in [1,2,3,4]:
    
                                     
    results_folder = 'results_pols'
    stages = 12
    lay_Depth = 16
    Num_lay  = 6
    rank_Val = 10
    transpose = 'unet'
    prior = 'mix'
    mode = 'sthocastic_cyclic'
    Train_c = False
    reg = False
    reg_param = 1000.
    description = 'FINAL'


    kwargs = {'rank':rank_Val,'Layer_depth':lay_Depth,'number_layer':Num_lay}

    prior_name = prior+'rank_'+str(kwargs['rank'])+'_Layer_depth_'+str(kwargs['Layer_depth'])+'_Nume_layer_'+str(kwargs['number_layer'])+'_loss_weights'
    if shots != 0:
         model_prev_path = 'results_final_v4/Shots_'+ str(shots -1)+'/best_model.tf'


    #model_prev_path = results_folder + '/models_final_transfer_weights__stage_12_shots_'+str(shots)+'_transpose_unet_prior_mixrank_10_Layer_depth_16_Nume_layer_6_loss_weights_mode_sthocastic_cyclic/best_model.tf'
    
    model_prev = Unrolling_TL(input_dim=(128,128,25),shots=shots-1,mode=mode,stages=stages,Train_c=Train_c,prior=prior,transpose=transpose,reg=reg,reg_param=reg_param,**kwargs)

    model_prev.build([BATCH_SIZE,IMG_HEIGHT,IMG_HEIGHT,L_bands])

    model_prev.load_weights(model_prev_path)





    model = Unrolling_TL(input_dim=(128,128,25),shots=shots,mode=mode,stages=stages,Train_c=Train_c,prior=prior,transpose=transpose,reg=reg,reg_param=reg_param,**kwargs)

    model.build([BATCH_SIZE,IMG_HEIGHT,IMG_HEIGHT,L_bands])

    if shots == 1:
        model.load_weights(model_prev_path)
    else:
        model = transfer_weights(model,model_prev,shots,stages)



    #model.load_weights(model_prev_path)

    optimizad =  tf.keras.optimizers.Adam(learning_rate=0.001)

    [callbacks,path] = load_callbacks(description = description, results_folder=results_folder,stages=stages,shots=shots,transpose='unet',prior=prior_name,mode=mode,model=model,dataset_path=dataset_path)

    model.compile(optimizer=optimizad,loss=loss_total(1,1),metrics=[psnr,SSIM,cos_distance])#loss_weights=l_weights)


    model.summary()
    K.set_value(model.optimizer.learning_rate, 0.0007)
    print(model.optimizer.learning_rate)
    history = model.fit(train_dataset, validation_data=val_dataset,epochs=100, batch_size=BATCH_SIZE,callbacks=callbacks)