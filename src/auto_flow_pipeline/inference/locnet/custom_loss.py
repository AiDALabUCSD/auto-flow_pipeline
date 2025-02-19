import tensorflow as tf
import numpy as np
#from tensorflow.keras import backend as K
K = tf.keras.backend

def custom_mse(y_true, y_pred):
    # Enforce float32 dtype
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    ## enforce mask to prevent empty data from contributin to loss
    #y_flags = tf.Variable(tf.ones(tf.shape(y_true), dtype=tf.dtypes.float32))*5
    y_flags = tf.ones(tf.shape(y_true), dtype=tf.dtypes.float32)*5
    y_mask = tf.math.logical_not(tf.math.equal(y_true,y_flags))
    
    y_true = tf.where(y_mask,y_true,0)
    y_pred = tf.where(y_mask,y_pred,0)

    y_attention = tf.where(y_true > 0.0, 1.0, 0.0)
    
    ALPHA = tf.constant(0.015)
    BETA = tf.constant(1.0)
    # We restrict to the 0th channel for MSE of AV
    
    channel = 0
    mse_AV = tf.reduce_mean( K.batch_flatten(BETA*y_attention[...,channel] + ALPHA) * (K.square(K.batch_flatten(y_true[...,channel]) - K.batch_flatten(y_pred[...,channel]))), axis=-1)
    
    channel = 1
    mse_STJ = tf.reduce_mean( K.batch_flatten(BETA*y_attention[...,channel] + ALPHA) * (K.square(K.batch_flatten(y_true[...,channel]) - K.batch_flatten(y_pred[...,channel]))), axis=-1)
    
    channel = 2
    mse_AA_pts = tf.reduce_mean( K.batch_flatten(BETA*y_attention[...,channel] + ALPHA) * (K.square(K.batch_flatten(y_true[...,channel]) - K.batch_flatten(y_pred[...,channel]))), axis=-1)
    
    channel = 3
    mse_aorta_heat = tf.reduce_mean( K.batch_flatten(BETA*y_attention[...,channel] + ALPHA) * (K.square(K.batch_flatten(y_true[...,channel]) - K.batch_flatten(y_pred[...,channel]))), axis=-1)
    
    channel = 4
    mse_PV = tf.reduce_mean( K.batch_flatten(BETA*y_attention[...,channel] + ALPHA) * (K.square(K.batch_flatten(y_true[...,channel]) - K.batch_flatten(y_pred[...,channel]))), axis=-1)

    channel = 5
    mse_main_PA = tf.reduce_mean( K.batch_flatten(BETA*y_attention[...,channel] + ALPHA) * (K.square(K.batch_flatten(y_true[...,channel]) - K.batch_flatten(y_pred[...,channel]))), axis=-1)

    channel = 6
    mse_PA_pts = tf.reduce_mean( K.batch_flatten(BETA*y_attention[...,channel] + ALPHA) * (K.square(K.batch_flatten(y_true[...,channel]) - K.batch_flatten(y_pred[...,channel]))), axis=-1)

    channel = 7
    mse_pulm_heat = tf.reduce_mean( K.batch_flatten(BETA*y_attention[...,channel] + ALPHA) * (K.square(K.batch_flatten(y_true[...,channel]) - K.batch_flatten(y_pred[...,channel]))), axis=-1)
    
    # loss as the sum of components
    loss_mse = 2*(mse_AV + mse_STJ + mse_AA_pts) + mse_aorta_heat + 2*(mse_PV + mse_main_PA + mse_PA_pts) + (mse_pulm_heat)
    
    return tf.cast(loss_mse, dtype = tf.float16)
