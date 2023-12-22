import tensorflow as tf
from itertools import combinations
def correlation_regularizer(y,reg_param,model):
    ssim_t = 0
    for i in list(combinations(list(range(len(y))),2)):
       ssim_t += tf.reduce_mean(tf.image.ssim(y[i[0]],y[i[1]],1))
    print(ssim_t.shape)
    model.add_loss(ssim_t)
    model.add_metric(ssim_t, name='loss_ssim', aggregation='mean')

        