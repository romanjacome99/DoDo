from operator import truediv
import os
from tabnanny import verbose
import cv2
import matplotlib.pyplot as plt
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


PATH = '../Data/Arad/Train/'
PATH_test = '../Data/Arad/Test/'
# PATH = '/content/drive/MyDrive/HDSP/Double doe/5Image/'  

# parameters of the net
BATCH_SIZE = 15
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_WIDTH_o = 128
IMG_HEIGHT_o = 128
L_bands = 25
L_imput = 25
shots = 4
dataset_path = 'C:\Roman\datasets\Arad1k\Arad'  # change to your path
dataset = 'arad'
train_dataset, val_dataset, test_dataset = load_dataset(dataset, dataset_path, batch_size=BATCH_SIZE,
                                                        input_size=(IMG_HEIGHT, IMG_HEIGHT, L_bands))

image_train = 'C:\Roman\datasets\Arad1k\Arad'  # change to your path
image_val = 'C:\Roman\datasets\Arad1k\Arad' # change to your path

for shots in [1, 2, 3, 4]:

    results_folder = 'results_mpz'
    stages = 12
    lay_Depth = 16
    Num_lay = 6
    rank_Val = 10
    transpose = 'unet'
    prior = 'mix'
    mode = 'sthocastic_cyclic'
    Train_c = True
    reg = False
    reg_param = 1000.
    description = 'models_final_transfer_weights__'

    kwargs = {'rank': rank_Val, 'Layer_depth': lay_Depth, 'number_layer': Num_lay}

    prior_name = prior + 'rank_' + str(kwargs['rank']) + '_Layer_depth_' + str(
        kwargs['Layer_depth']) + '_Nume_layer_' + str(kwargs['number_layer']) + '_loss_weights'


    if shots == 0:
       path = results_folder + '/spiral_model'
    else:
        path = results_folder + '/models_mpz_transfer_weights_v2_Nterms_200_stage_12_shots_'+str(shots)+'_transpose_unet_prior_mixrank_10_Layer_depth_16_Nume_layer_6_loss_weights_mode_sthocastic_cyclic'


    # model_prev_path = results_folder + '/models_final_transfer_weights__stage_12_shots_'+str(shots)+'_transpose_unet_prior_mixrank_10_Layer_depth_16_Nume_layer_6_loss_weights_mode_sthocastic_cyclic/best_model.tf'



    model = Unrolling_TL_Free(input_dim=(128, 128, 25), shots=shots, mode=mode, stages=stages, Train_c=Train_c, prior=prior, Nterms=200,
                         transpose=transpose, reg=reg, reg_param=reg_param, **kwargs)

    model.build([BATCH_SIZE, IMG_HEIGHT, IMG_HEIGHT, L_bands])


    model.load_weights(path + '/best_model.tf')

    # model.load_weights(model_prev_path)

    optimizad = tf.keras.optimizers.Adam(learning_rate=0.001)



    model.compile(optimizer=optimizad, loss=loss_total(1, 1),
                  metrics=[psnr, SSIM, cos_distance])  # loss_weights=l_weights)

    model.summary()

    ssim_v = np.zeros((50, 13))
    psnr_v = np.zeros((50, 13))
    sam_v = np.zeros((50, 13))
    metrics_eval_train = np.array(model.evaluate(train_dataset,verbose=1))

    metrics_eval_val = np.array(model.evaluate(val_dataset,verbose=1))

    metrics_eval_test = np.array(model.evaluate(test_dataset,verbose=1))

    try:
        os.mkdir(path+'/images_test')
    except OSError as error:
        print(error)
    for i in range(50):

        with h5py.File(dataset_path + '/test.h5', 'r') as hf:
            val_img = hf['cube'][i]
            val_img = val_img.astype(np.float32)

            val_img = tf.image.central_crop(val_img, 0.25)[..., 3:28]
            val_img = tf.expand_dims(val_img / np.max([np.max(val_img), 1e-6]), 0)

        xr = model(val_img)
        start, end = 450, 650  # VariSpec VIS
        number_bands = 25
        color_space = "sRGB"

        cs = ColourSystem(cs=color_space, start=start, end=end, num=number_bands)
        psnr_v[i, :] = np.array(tf.image.psnr(tf.concat(xr, 0), val_img, 1))
        ssim_v[i, :] = np.array(tf.image.ssim(tf.concat(xr, 0), val_img, 1))
        sam_v[i, :] = np.array(cos_distance(tf.concat(xr, 0), val_img))

        img_rgb = cs.spec_to_rgb(xr[-1])
        img_rgb_gt = cs.spec_to_rgb(val_img)

        xr_i = np.array(xr[-1])
        xgt_i = np.array(val_img)


    metrics_eval_train = np.array(model.evaluate(train_dataset,verbose=0))

    metrics_eval_val = np.array(model.evaluate(val_dataset,verbose=0))

    metrics_eval_test = np.array(model.evaluate(test_dataset,verbose=0))

