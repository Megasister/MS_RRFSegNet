import tensorflow as tf
import os
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
from rrfrm import regional_relation_features_reasoning_layers, get_triads


def get_ms_rrfsegnet_model(supervoxel, NUM_CLASSES, is_training, nn_idx1, nn_idx2, k,
                          is_dist=True, weight_decay=0.0004, bn_decay=None):
    """ build ms_rrfsegnet model """
    num_point = supervoxel.get_shape()[1].value
    ### layer_1
    triads_scale_1 = get_triads(supervoxel, nn_idx=nn_idx1, k=k)
    triads_scale_2 = get_triads(supervoxel, nn_idx=nn_idx2, k=k)

    net_1_s1 = regional_relation_features_reasoning_layers('layer_1_s1',
                                                           triads_scale_1,
                                                           is_training, bn_decay,
                                                           [64, 64, 64],
                                                           weight_decay,
                                                           is_dist=is_dist)

    net_1_s2 = regional_relation_features_reasoning_layers('layer_1_s2',
                                                           triads_scale_2,
                                                           is_training, bn_decay,
                                                           [64, 64, 64],
                                                           weight_decay,
                                                           is_dist=is_dist)

    net_1 = tf_util.conv2d(tf.concat([net_1_s1, net_1_s2], axis=-1), 128, [1, 1],
                                                           padding='VALID', stride=[1, 1],
                                                           bn=True, is_training=is_training,
                                                           scope='mpl_agg_1', bn_decay=bn_decay,
                                                           is_dist=is_dist)

    ### layer_2
    triads_scale_1 = get_triads(net_1, nn_idx=nn_idx1, k=k)
    triads_scale_2 = get_triads(net_1, nn_idx=nn_idx2, k=k)
    net_2_s1 = regional_relation_features_reasoning_layers('layer_2_s1',
                                                           triads_scale_1,
                                                           is_training,
                                                           bn_decay,
                                                           [128, 128, 128],
                                                           weight_decay,
                                                           is_dist=is_dist)

    net_2_s2 = regional_relation_features_reasoning_layers('layer_2_s2',
                                                           triads_scale_2,
                                                           is_training,
                                                           bn_decay,
                                                           [128, 128, 128],
                                                           weight_decay,
                                                           is_dist=is_dist)

    net_2 = tf_util.conv2d(tf.concat([net_2_s1, net_2_s2], axis=-1), 256, [1, 1],
                                                           padding='VALID', stride=[1, 1],
                                                           bn=True, is_training=is_training,
                                                           scope='mpl_agg_2', bn_decay=bn_decay,
                                                           is_dist=is_dist)

    ###global features
    global_net = tf_util.conv2d(net_2, 1024, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=True, is_training=is_training,
                                scope='mpl_global', bn_decay=bn_decay,
                                is_dist=is_dist)
    # global_net = tf_util.conv2d(tf.concat([net_1, net_2]), 1024, [1, 1],
    #                             padding='VALID', stride=[1, 1],
    #                             bn=True, is_training=is_training,
    #                             scope='mpl_global', bn_decay=bn_decay,
    #                             is_dist=is_dist)

    global_net = tf.reduce_max(global_net, axis=1, keep_dims=True)
    global_net = tf.tile(global_net, [1, num_point, 1, 1])

    ###
    concat = tf.concat(axis=3, values=[global_net,
                                       net_1,
                                       net_2])
    # CONV
    net = tf_util.conv2d(concat, 256, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='swsl/conv1',
                         weight_decay=weight_decay, is_dist=is_dist, bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout_05')
    net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='swsl/conv2',
                         weight_decay=weight_decay, is_dist=is_dist, bn_decay=bn_decay)

    net = tf_util.conv2d(net, NUM_CLASSES, [1, 1], padding='VALID', stride=[1, 1],
                         activation_fn=None, scope='swsl/conv3',
                         weight_decay=weight_decay, is_dist=is_dist)

    net = tf.nn.softmax(net, name='softmax_layer')
    net = tf.squeeze(net, [2])
    return net

def compute_class_weights(labels, num_classes, scale_parameter=0.5):
    classes = [i for i in range(num_classes)]
    all_f = [np.sum(labels == classes[i]) for i in range(num_classes)]
    all_f = np.array(all_f, dtype=np.float32)
    median_f = np.median(all_f)
    weights = np.zeros(len(classes))
    for i in range(len(classes)):
        if all_f[i] != 0:
            weights[i] = median_f / (all_f[i])
    weights = np.array(weights, dtype=np.float32)
    ###avoid weight overflow
    for i in range(len(classes)):
        if weights[i] != 0:
            if weights[i] - 1 >= 0:
                weights[i] = 1.0/(1+np.exp(-(weights[i]-1))) + scale_parameter
            else:
                weights[i] = np.exp(weights[i]-1)/(np.exp(weights[i]-1)+1) + scale_parameter
        else:
            continue
    return weights
# Median frequency balancing
def get_weight_cross_entropy_loss(class_weights, prediction, labels):
    loss = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)) * class_weights,
                                        reduction_indices=[2])
    cross_entropy = tf.reduce_mean(loss, name="weight_cross_entropy_loss")
    return cross_entropy

def get_cross_entropy_loss(prediction, labels):
    loss = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)),
                          reduction_indices=[2])
    cross_entropy = tf.reduce_mean(loss, name="cross_entropy_loss")
    return cross_entropy