from operator import truediv
import os 
import tensorflow as tf
import scipy.io as sio
from scipy.io import savemat
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = '1'        

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
from utils.callbacks import *
from decoder.unrolling_network import *
from utils.loss_metrics import *
from dataset.data import *

# PATH = '../Data/Arad/Train/'
# PATH_test = '../Data/Arad/Test/'
# # PATH = '/content/drive/MyDrive/HDSP/Double doe/5Image/'  
modelos=[]
# parameters of the net
BATCH_SIZE = 28
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_WIDTH_o = 128
IMG_HEIGHT_o = 128
L_bands = 25
L_imput = 25
shots = 4
dataset_path = 'C:\Roman\datasets\Arad1k\Arad' # change to your path
dataset = 'arad'
train_dataset, val_dataset, test_dataset = load_dataset(dataset, dataset_path, batch_size=BATCH_SIZE,
                                                            input_size=(IMG_HEIGHT,IMG_HEIGHT,L_bands))
snr = 40
path = 'C:\Roman\datasets\Arad1k\Arad'
#for snr in [5,10,15,20,25,30,35]:
for it in range(5):
    Xt = np.zeros((10,128,128,25))
    with h5py.File(path, 'r') as hf:
                for i,X in enumerate(hf['cube'][it*10:(it+1)*10]):
                    x = X.astype(np.float32)# / (2 ** 16 - 1)
                    #x = x/np.max([np.max(x),1e-6])
                    X = tf.image.central_crop(x,0.25)
                    X = X/np.max([np.max(X),1e-6])
                    Xt[i,:,:,:] = X[:,:,3:28]
                    # transformaciones
                    

                    
    for shots in range(5):

        #lay_Depth = 20
        #rank_val = 10
                                        
        results_folder = 'models/results_200pols'
        stages = 12
        lay_Depth = 16
        Num_lay  = 6
        rank_Val = 10
        transpose = 'unet'
        prior = 'mix'
        mode = 'sthocastic_cyclic'
        Train_c = True
        reg = False
        reg_param = 1000.
        description = 'training_data_norm_experiment_test_arad1k_initialization_testss'
        optimizad =  tf.keras.optimizers.Adam(learning_rate=0.001)
        kwargs = {'rank':rank_Val,'Layer_depth':lay_Depth,'number_layer':Num_lay}
        prior_name = prior+'rank_'+str(kwargs['rank'])+'_Layer_depth_'+str(kwargs['Layer_depth'])+'_Nume_layer_'+str(kwargs['number_layer'])+'_loss_weights'

        model = Unrolling_Free(input_dim=(128,128,25),shots=shots,mode=mode,stages=stages,Nterms=200,Train_c=Train_c,prior=prior,transpose=transpose,reg=reg,reg_param=reg_param,noise=False,snr=snr,**kwargs)



        optimizad =  tf.keras.optimizers.Adam(learning_rate=0.001)
        #l_weights = [1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,100]
        l_weights = np.logspace(-1,1,stages+1)
        model.load_weights('results_mpz_2/shots_'+str(shots)+'/best_model.tf')
        model.compile(optimizer=optimizad,loss=loss_total(1,1),metrics=[psnr,SSIM])#loss_weights=l_weights)
        Xrec = np.array(model(Xt))
        psnr = tf.image.psnr(Xt,Xrec,1)
        print(psnr)
        #a = np.array(model.evaluate(test_dataset))
        # scipy.io.savemat(f'Visual_results/200_recon shots_{shots}_{it}.mat',{'Xrec':Xrec,'Xgt':Xt})



