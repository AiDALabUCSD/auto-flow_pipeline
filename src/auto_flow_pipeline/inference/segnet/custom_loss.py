import tensorflow as tf
import numpy as np
#from tensorflow.keras import backend as K
K = tf.keras.backend

def custom_dice(y_true, y_pred):
    # Convert dtypes
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    smooth = 1e-7
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=-1)
    union = tf.reduce_sum(y_true_f, axis=-1) + tf.reduce_sum(y_pred_f, axis=-1)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = (1. - dice)

    return tf.cast(dice_loss, dtype=tf.float32)