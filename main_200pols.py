from operator import truediv
import os
from tabnanny import verbose
import tensorflow as tf
import scipy.io as sio

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
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


def transfer_weights_2(model, model_prev, shots, stages):

    for i in range(shots):
        print(model.layers[i],model_prev.layers[i])
        coeff_v = np.zeros((Nterms, 1, 1))
        coeff = model_prev.layers[i].get_weights()
        coeff_pret = np.squeeze(model_prev.layers[i].get_weights()[1])

        coeff_v[3:15, 0, 0] = coeff_pret

        coeff[1] = coeff_v
        model.layers[i].set_weights(coeff)
    for i in range(shots,2*shots-1):
        model.layers[i].set_weights(model_prev.layers[i].get_weights())
    for m in range(stages):
        model.get_layer('HQS_update_' + str(m)).set_weights(model_prev.get_layer('HQS_update_' + str(m)).get_weights())

    return model


#
def transfer_weights(model, model_prev, shots, stages):
    t = -1
    for i in range(shots):
        t += 1

        if t >= shots - 1:
            t -= 1
        model.layers[i].set_weights(model_prev.layers[t].get_weights())
    for i in range(shots, 2 * shots - 1):
        model.layers[i].set_weights(model_prev.layers[i - 1].get_weights())

    for i in range(stages):
        model.get_layer('HQS_update_' + str(i)).set_weights(model_prev.get_layer('HQS_update_' + str(i)).get_weights())

    return model

def transfer_weights_3(model, model_prev, shots, stages):

    for i in range(shots, 2 * shots - 1):
        model.layers[i].set_weights(model_prev.layers[i - 1].get_weights())

    for i in range(stages):
        model.get_layer('HQS_update_' + str(i)).set_weights(model_prev.get_layer('HQS_update_' + str(i)).get_weights())

    return model

# parameters of the net
BATCH_SIZE = 15
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_WIDTH_o = 128
IMG_HEIGHT_o = 128
L_bands = 25
L_imput = 25
shots = 4
dataset_path = 'C:\Roman\datasets\Arad1k\Arad' # Change for your own path
dataset = 'arad'
train_dataset, val_dataset, test_dataset = load_dataset(dataset, dataset_path, batch_size=BATCH_SIZE,
                                                        input_size=(IMG_HEIGHT, IMG_HEIGHT, L_bands))

image_train = 'C:\Roman\datasets\Arad1k\Arad/train.h5' # Change for your own path
image_val = 'C:\Roman\datasets\Arad1k\Arad/test.h5' # Change for your own path

for shots in [1,2,3,4]:
    print('shots_'+str(shots))
    results_folder = 'models/results_200pols'
    stages = 12
    lay_Depth = 16
    Num_lay = 6
    rank_Val = 10
    transpose = 'unet'
    prior = 'mix'
    mode = 'sthocastic_cyclic'
    Train_c = True
    reg = False
    reg_param = 1000
    Nterms = 200
    description = 'models_mpz_transfer_weights_v2_Nterms_' +str(Nterms)

    kwargs = {'rank': rank_Val, 'Layer_depth': lay_Depth, 'number_layer': Num_lay}

    prior_name = prior + 'rank_' + str(kwargs['rank']) + '_Layer_depth_' + str(
        kwargs['Layer_depth']) + '_Nume_layer_' + str(kwargs['number_layer']) + '_loss_weights'

    model_prev_path = 'models/results_200pols/shots_'+str(shots-1)+'/best_model.tf'

    model_prev = Unrolling_TL_Free(input_dim=(128, 128, 25), shots=shots-1, mode=mode, stages=stages, Train_c=Train_c,
                              prior=prior,
                              transpose=transpose, reg=reg, reg_param=reg_param, Nterms=Nterms, **kwargs)

    model_prev.build([BATCH_SIZE, IMG_HEIGHT, IMG_HEIGHT, L_bands])

    model_prev.load_weights(model_prev_path)
    model_prev.summary()
    model = Unrolling_Free(input_dim=(128, 128, 25), shots=shots, mode=mode, stages=stages, Train_c=Train_c, prior=prior,
                         transpose=transpose, reg=reg, reg_param=reg_param,Nterms = Nterms, **kwargs)

    model.build([BATCH_SIZE, IMG_HEIGHT, IMG_HEIGHT, L_bands])
    lr = 1e-6

    optimizad = tf.keras.optimizers.Adam(learning_rate=lr)
    if shots == 1:
        model  = transfer_weights_3(model, model_prev, shots, stages)
    else:
        model = transfer_weights(model, model_prev, shots, stages)

    # model.load_weights(model_prev_path)



    [callbacks, path] = load_callbacks(description=description, results_folder=results_folder, stages=stages,
                                       shots=shots, transpose='unet', prior=prior_name, mode=mode, model=model,
                                       dataset_path=dataset_path)

    model.compile(optimizer=optimizad, loss=loss_total(1, 1),
                  metrics=[psnr, SSIM, cos_distance])  # loss_weights=l_weights)

    K.set_value(model.optimizer.learning_rate, lr)

    print(model.optimizer.learning_rate)
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=50, batch_size=BATCH_SIZE,
                         callbacks=callbacks)
