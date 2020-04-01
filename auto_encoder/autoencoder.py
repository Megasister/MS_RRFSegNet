'''
sparse auto-encoder for supervoxel embedding representation
'''

import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util

def get_model_SAE(point_cloud, is_training, bn_decay=None):

    inputs = tf.expand_dims(point_cloud, -2)
    encoder = tf_util.conv2d(inputs, 16, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             activation_fn=tf.nn.sigmoid, scope='conv1', bn_decay=bn_decay)

    decoder = tf_util.conv2d(encoder, 3, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             activation_fn=None, scope='conv2', bn_decay=bn_decay)

    decoder = tf.squeeze(decoder, [2])
    encoder = tf.reduce_max(encoder, axis=1, keep_dims=True)
    return encoder, decoder

def kl_divergence(p, p_hat):
    return tf.reduce_sum(p*tf.log(tf.clip_by_value(p, 1e-8, tf.reduce_max(p)))
                          - p*tf.log(tf.clip_by_value(p_hat, 1e-8, tf.reduce_max(p_hat)))
                          + (1-p)*tf.log(tf.clip_by_value(1-p, 1e-8, tf.reduce_max(1-p)))
                          - (1-p)*tf.log(tf.clip_by_value(1-p_hat,1e-8, tf.reduce_max(1-p_hat))))

def get_loss_sparsity(encoder, decoder, label, p, lamda=0.05):
  loss1 = tf.reduce_mean(tf.square(decoder - label))
  p_hat = tf.reduce_mean(encoder, 0)
  p_hat = tf.reduce_mean(p_hat, 0)
  p_hat = tf.squeeze(p_hat)
  kl = kl_divergence(p, p_hat)
  return loss1 + lamda*kl
