import os
import random

import h5py
import numpy as np
from scipy.io import loadmat
from skimage.transform import resize

import tensorflow as tf
from tensorflow.keras import layers


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# simulated scene
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

def load_scene(scene_path, spectral_bands):
    scene_mat = loadmat(scene_path)
    spectral_scene = np.double(scene_mat['hyperimg'])[np.newaxis, ...]
    spectral_scene = spectral_scene[..., ::int(spectral_scene.shape[-1] / spectral_bands + 1)]
    spectral_scene = spectral_scene / np.max(spectral_scene)

    spectral_scene = tf.image.central_crop(spectral_scene, 256 / spectral_scene.shape[1])
    spectral_scene = spectral_scene.numpy()  # [..., ::int(31 / 7 + 1)]

    rgb_colors = (15, 13, 6) if spectral_bands <= 16 else (25, 22, 11)
    scene_rgb = spectral_scene[..., rgb_colors]
    input_size = spectral_scene.shape

    return input_size, spectral_scene, scene_rgb


def get_scene_data(sensing_name, model, spectral_scene, only_measure=False, only_transpose=False):
    try:
        sensing_model = model.get_layer(sensing_name)
        data = sensing_model(spectral_scene, only_measure=only_measure, only_transpose=only_transpose)

    except:
        raise 'sensing was not in the model, please check the code implementation'

    return data


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# real data
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

def load_real_data(real_path, H_path, y_path):
    H_real = loadmat(os.path.join(real_path, H_path))['H'][..., 0]
    y_real = loadmat(os.path.join(real_path, y_path))['Y'][np.newaxis, ..., 0]

    return H_real, y_real


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# dataset
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

def load_dataset(name, path, batch_size=1, input_size=(512, 512, 31),img=5):
    if name == 'arad':
        train_path = os.path.join(path, 'train.h5')
        val_path = os.path.join(path, 'val.h5')
        test_path = os.path.join(path, 'test.h5')

        train_dataset = get_arad_dataset_train(train_path, batch_size, input_size, augment=False)
        val_dataset = get_arad_dataset(val_path, batch_size, input_size)
        test_dataset = get_arad_dataset(test_path, batch_size, input_size)

        return train_dataset, val_dataset, test_dataset

    else:
        raise 'You should load a dataset'


def get_arad_dataset(path, batch_size, input_size, augment=False):
    dataset = ARADDataset(path, input_size)

    augmentation = []
    if augment:
        augmentation.extend([
            layers.RandomRotation(0.3),
            layers.RandomZoom((-0.5, 0.5)),
            layers.RandomTranslation(0.0, 0.1),
            layers.RandomFlip('horizontal')
        ])
    augmentation = tf.keras.Sequential(augmentation)

    dataset_pipeline = (dataset
                        .cache()
                        #.shuffle(batch_size)
                        .batch(batch_size, drop_remainder=True)
                        .map(lambda x: augmentation(x, training=True), num_parallel_calls=tf.data.AUTOTUNE)
                        .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
                        .prefetch(tf.data.AUTOTUNE))

    return dataset_pipeline


class ARADDataset(tf.data.Dataset):
    def _generator(path):
        with h5py.File(path, 'r') as hf:
            for X in hf['cube']:
                x = X.astype(np.float32)# / (2 ** 16 - 1)
                #x = x/np.max([np.max(x),1e-6])
                X = tf.image.central_crop(x,0.25)
                X = X/np.max([np.max(X),1e-6])
                # transformaciones
                

                yield X[...,3:28]

    def __new__(cls, path, input_size=(512, 512, 31)):
        print(f'input_size: {input_size}')
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=tf.TensorSpec(shape=input_size, dtype=tf.float32),
            args=(path,)
        )


def get_arad_dataset_test(path, batch_size, input_size, augment=False,img=5):
    dataset = ARADDataset_test(path, input_size,img)

    augmentation = []
    if augment:
        augmentation.extend([
            layers.RandomRotation(0.3),
            layers.RandomZoom((-0.5, 0.5)),
            layers.RandomTranslation(0.0, 0.1),
            layers.RandomFlip('horizontal')
        ])
    augmentation = tf.keras.Sequential(augmentation)

    dataset_pipeline = (dataset
                        .cache()
                        #.shuffle(batch_size)
                        .batch(batch_size, drop_remainder=True)
                        .map(lambda x: augmentation(x, training=False), num_parallel_calls=tf.data.AUTOTUNE)
                        .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
                        .prefetch(tf.data.AUTOTUNE))

    return dataset_pipeline

class ARADDataset_test(tf.data.Dataset):
    def _generator(path,img):
        with h5py.File(path, 'r') as hf:
            #for X in hf['cube'][:1]:
                x = hf['cube'][img].astype(np.float32)# / (2 ** 16 - 1)
                x = x/np.max([np.max(x),1e-6])
                X = tf.image.central_crop(x,0.25)
                #X = X/np.max([np.max(X),1e-6])
                # transformaciones
                

                yield X[...,3:28]

    def __new__(cls, path, input_size=(512, 512, 31),img=5):
        print(f'input_size: {input_size}')
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=tf.TensorSpec(shape=input_size, dtype=tf.float32),
            args=(path,img,)
        )




def get_arad_dataset_train(path, batch_size, input_size, augment=False):
    dataset = ARADDataset_train(path, input_size)

    augmentation = []
    if augment:
        augmentation.extend([
            layers.RandomRotation(0.3),
            layers.RandomZoom((-0.5, 0.5)),
            layers.RandomTranslation(0.0, 0.1),
            layers.RandomFlip('horizontal')
        ])
    augmentation = tf.keras.Sequential(augmentation)

    dataset_pipeline = (dataset
                        .cache()
                        #.shuffle(batch_size)
                        .batch(batch_size, drop_remainder=True)
                        .map(lambda x: augmentation(x, training=True), num_parallel_calls=tf.data.AUTOTUNE)
                        .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
                        .prefetch(tf.data.AUTOTUNE))

    return dataset_pipeline

class ARADDataset_train(tf.data.Dataset):
    def _generator(path):
        with h5py.File(path, 'r') as hf:
            for X in hf['cube']:
                x = X.astype(np.float32)# / (2 ** 16 - 1)
                
                X = tf.image.random_crop(x,[128,128,31])
                X = X/np.max([np.max(X),1e-6])

                # transformaciones
                

                yield X[...,3:28]

    def __new__(cls, path, input_size=(512, 512, 31)):
        print(f'input_size: {input_size}')
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=tf.TensorSpec(shape=input_size, dtype=tf.float32),
            args=(path,)
        )
