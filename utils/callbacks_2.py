import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import rc
from utils.spec2rgb import ColourSystem
import scipy.io as sio

import numpy as np
from utils.tensorboard import *
import h5py


def plot_results(model,test_dataset,path):

  img = []
  for i in test_dataset.as_numpy_iterator():
    img.append(tf.expand_dims(i[0][0],0))
  imgf = [img[8],img[3]]
  X = []
  X.append(model(imgf[0])[-1])
  X.append(model(imgf[1])[-1])


  color_space = "sRGB"
  start, end = 450, 650 # VariSpec VIS
  number_bands = 25

  cs = ColourSystem(cs=color_space, start=start, end=end, num=number_bands)
  img_rgb_gt = []
  img_rgb_r = []
  img_rgb_gt.append(np.squeeze(cs.spec_to_rgb(imgf[0])))
  img_rgb_gt.append(np.squeeze(cs.spec_to_rgb(imgf[1])))
  img_rgb_r.append(np.squeeze(cs.spec_to_rgb(X[0])))
  img_rgb_r.append(np.squeeze(cs.spec_to_rgb(X[1])))

  plt.rcParams.update({'font.size': 40})
  fig,ax = plt.subplots(2,2,figsize=(30,30),constrained_layout=True)


  for i in range(2):
    ax[i,0] = plot_zoom(ax[i,0],img_rgb_gt[i])
    ax[i,1] = plot_zoom(ax[i,1],img_rgb_r[i])
    if i == 0:
      ax[i,0].set_title('Ground Truth')
      psnr = round(tf.image.psnr(X[i],imgf[i],1).numpy()[0],3)
      ax[i,1].set_title('Recovery \n PSNR = '+str(psnr)+ ' [dB]')
    else: 
      psnr = round(tf.image.psnr(X[i],imgf[i],1).numpy()[0],3)
      ax[i,1].set_title('PSNR = '+str(psnr) + ' [dB]')

  fig.savefig(path+'/recon.png')
  print('Saved Image Recon at '  + path + '/recon.png')

def plot_zoom(ax,img):
  ax.imshow(img,  origin="lower")
  axins = zoomed_inset_axes( ax, 2, loc=3)
  for axis in ['top','bottom','left','right']:
      axins.spines[axis].set_linewidth(1.2)
      axins.spines[axis].set_color('w')
  axins.imshow(img,  origin="lower")
  axins.set_xlim(50, 80)
  axins.set_ylim(20, 50)
  plt.xticks(visible=False)
  mark_inset(ax, axins, loc1=1, loc2=2, fc="none",ec='w',lw=1.2)
  ax.invert_yaxis()
  axins.invert_yaxis()
  ax.axis("Off")
  axins.set_xticks([])
  axins.set_yticks([])
  return ax
 
class save_each_epoch(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, logs=None):
        print('Model Saved at: ' + self.checkpoint_dir)
        self.model.save_weights(self.checkpoint_dir)

def lr_scheduler(epoch, lr):
    decay_step = 40
    if epoch % decay_step == 0 and epoch:
        lr = lr/2
        tf.print(' Learning rate ='+ str(lr))        
        return lr
    
    return lr
class Aument_parameters(tf.keras.callbacks.Callback):
    def __init__(self, p_aum,p_step):
        super().__init__()
        self.p_aum = p_aum
        self.p_step = p_step
        
    def on_epoch_end(self, epoch, logs=None):
        current_param=self.model.layers[1].my_regularizer.parameter
        current_param = tf.keras.backend.get_value(current_param)
        print('\n regularizator ='+ str(current_param))
        
        if epoch%self.p_step==0 and epoch>50:
            
            new_param = current_param * self.p_aum
            self.model.layers[1].my_regularizer.parameter.assign(new_param)    
            print('\n regularizator updated to '+ str(new_param))



def load_callbacks(description='exp',results_folder='results',stages=12,shots=4,transpose='unet',prior='hssp',mode='sthocastic',model=None,dataset_path='image_val'):
    
    try:
        os.mkdir(results_folder)
    except OSError as error:
        print(error)

    experiment = 'Shots_' +str(shots)
    
    path =os.path.join(results_folder,experiment ,'')

    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
    csv_file = os.path.join(path,'results.csv')


    
    model_path = os.path.join(path,'best_model.tf')


    check_point = tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor="val_output_13_psnr",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        save_freq="epoch",
        verbose=1)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(path,'tensoboard',''), histogram_freq=1, write_graph = False)

    fw_val = tf.summary.create_file_writer(os.path.join(path+"tensorboard", 'image_val'))
    fw_train = tf.summary.create_file_writer(os.path.join(path+"tensorboard", 'image_train'))
    fw_test = tf.summary.create_file_writer(os.path.join(path+"tensorboard", 'image_test'))


    with h5py.File(dataset_path+'/train.h5', 'r') as hf:
        train_img = hf['cube'][150]
        train_img = train_img.astype(np.float32)# / (2 ** 16 - 1)
        train_img = tf.image.central_crop(train_img,0.25)[...,3:28]
        train_img = tf.expand_dims(train_img/np.max([np.max(train_img),1e-6]),0)
    with h5py.File(dataset_path+'/val.h5', 'r') as hf:
        val_img = hf['cube'][20]
        val_img = val_img.astype(np.float32)
        
        val_img = tf.image.central_crop(val_img,0.25)[...,3:28]
        val_img = tf.expand_dims(val_img/np.max([np.max(val_img),1e-6]),0)
        
    with h5py.File(dataset_path+'/test.h5', 'r') as hf:
        test_img = hf['cube'][30]
        test_img = test_img.astype(np.float32)
        test_img = tf.image.central_crop(test_img,0.25)[...,3:28] 
        test_img = tf.expand_dims(test_img/np.max([np.max(test_img),1e-6]),0)
    


    images_val_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_xnca(epoch, logs, model = model, val_img = val_img, fw_results = fw_val, name = "val"))    
    images_train_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_xnca(epoch, logs, model = model, val_img = train_img, fw_results = fw_train, name = "train"))
    images_test_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_xnca(epoch, logs, model = model, val_img = test_img, fw_results = fw_test , name = "test"))

    lr_s = LearningRateScheduler(lr_scheduler, verbose=1)
    # lr_s = tf.keras.callbacks.ReduceLROnPlateau(
    # monitor='output_13_psnr',
    # factor=0.5,
    # patience=20,
    # verbose=1,
    # mode='max',
    # min_delta=0.0001,
    # cooldown=0,
    # min_lr=1e-7,
    
    # )
    
    callbacks = [tf.keras.callbacks.CSVLogger(csv_file, separator=',', append=False),
                lr_s,
                check_point,
                tensorboard_callback,
                images_val_cb,
                images_train_cb,
                images_test_cb]
    
    return callbacks, path


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png',bbox_inches='tight')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def plot_xnca(Xgt, Xt):

    """Return a 5x3 grid of the validation images as a matplotlib figure."""

    color_space = "sRGB"
    start, end = 450, 650 # VariSpec VIS
    number_bands = 25

    cs = ColourSystem(cs=color_space, start=start, end=end, num=number_bands)

    figure, axs_t = plt.subplots(1,len(Xt)+1, figsize=(20,5))
    img_rgb = cs.spec_to_rgb(Xgt)
    axs_t[0].imshow(tf.squeeze(img_rgb).numpy())
    axs_t[0].axis('off')
    axs_t[0].set_title('GT')

    for i in range(1,len(Xt)+1):
        Xi = tf.cast(Xgt,Xt[i-1].dtype)
        psnr = np.round(tf.image.psnr(Xt[i-1],Xi,tf.reduce_max([tf.reduce_max(Xt[i-1]),tf.reduce_max(Xi)]))[0],3)
        img_rgb = cs.spec_to_rgb(Xt[i-1])
        axs_t[i].imshow(tf.squeeze(img_rgb).numpy())
        axs_t[i].axis('off')
        axs_t[i].set_title('PSNR = ' +str(psnr))
    figure.tight_layout()
    return figure


def log_xnca(epoch, logs, model, val_img, fw_results, name):
    # Use the model to predict the values from the validation dataset.
    xgt = val_img
    xgt = xgt[0:1,...]
    Xt= model(xgt)
    # Log the results images as an image summary.
    figure = plot_xnca(xgt, Xt)
    image_resutls = plot_to_image(figure)

    # Log the results images as an image summary.
    with fw_results.as_default():
        tf.summary.image(name, image_resutls, step=epoch)