import tensorflow as tf
import tensorflow.keras.backend as K

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=K.max(y_true))


def cos_distance(y_true, y_pred):
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.maximum(x, K.epsilon()) / K.maximum(norm, K.epsilon())

    y_true = l2_normalize(y_true, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return K.mean(tf.math.acos(K.sum(y_true * y_pred, axis=-1)))


def psnr_std(y_true, y_pred):
    return tf.reduce_std(tf.image.psnr(y_true, y_pred, max_val=K.max(y_true)))


def cos_distance_std(y_true, y_pred):
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.maximum(x, K.epsilon()) / K.maximum(norm, K.epsilon())

    y_true = l2_normalize(y_true, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return tf.reduce_std(tf.math.acos(K.sum(y_true * y_pred, axis=-1)))

def SSIM_std(y_true,y_pred):
    return tf.reduce_std(tf.image.ssim(y_pred,y_true,K.max(y_true)))


def relRMSE(y_true,y_pred):
    true_norm = K.sqrt(K.sum(K.square(y_true), axis=-1))
    return K.mean(K.sqrt(tf.keras.losses.mean_squared_error(y_true, y_pred))/true_norm)
def SSIM(y_true,y_pred):
    return tf.image.ssim(y_pred,y_true,K.max(y_true))

def root_mean_squared_error(y_true, y_pred):
    return 40 * tf.reduce_mean(tf.norm(y_true - y_pred, ord=2, axis=-1)) + 1*tf.reduce_mean(
        tf.norm(y_true - y_pred, ord='fro', axis=[1, 2]))


def l2_fro(y_true, y_pred):
    return 40 * tf.reduce_mean(tf.norm(y_true - y_pred, ord=2, axis=-1)) + 1*tf.reduce_mean(
        tf.norm(y_true - y_pred, ord='fro', axis=[1, 2]))
def loss_total(rho_s,rho_l):
    def lossimage_2(y_true, y_pred):
        spatial_loss = tf.reduce_mean(1 - tf.image.ssim(y_pred, y_true, 1)) + tf.reduce_mean(tf.norm(y_true-y_pred,ord=1))
        a_b = tf.math.reduce_sum(tf.multiply(y_pred, y_true), axis=-1)
        mag_a = tf.sqrt(tf.reduce_sum(y_pred ** 2, axis=-1))
        mag_b = tf.sqrt(tf.reduce_sum(y_true ** 2, axis=-1))
        spectral_loss = tf.reduce_mean(tf.abs(a_b - tf.multiply(mag_a, mag_b)))
        val = rho_l * spectral_loss + rho_s * spatial_loss
        return val

    return lossimage_2

def spatial_loss_():
    def lossimage_2(y_true, y_pred):
        spatial_loss = tf.reduce_mean(1 - tf.image.ssim(y_pred, y_true, 1)) + tf.reduce_mean(tf.norm(y_true-y_pred,ord=1))    
        return spatial_loss
    return lossimage_2

def spectral_loss_():
    def lossimage_2(y_true, y_pred):
        a_b = tf.math.reduce_sum(tf.multiply(y_pred, y_true), axis=-1)
        mag_a = tf.sqrt(tf.reduce_sum(y_pred ** 2, axis=-1))
        mag_b = tf.sqrt(tf.reduce_sum(y_true ** 2, axis=-1))
        spectral_loss = tf.reduce_mean(tf.pow(a_b/tf.multiply(mag_a, mag_b),2)-1)
        return spectral_loss
    return lossimage_2
