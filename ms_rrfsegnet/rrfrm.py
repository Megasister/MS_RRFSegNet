import tensorflow as tf
import os
import sys
# from itertools import combinations
# from scipy.special import comb
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util

def regional_relation_features_reasoning_layers(name, inputs, is_training, bn_decay,
                              nodes_list, weight_decay, is_dist):
    """ Regional Relation Features Reasoning Module """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        net = tf_util.conv2d(inputs, nodes_list[0], [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=True, is_training=is_training,
                              scope='mpl_g1', bn_decay=bn_decay,
                              weight_decay=weight_decay, is_dist=is_dist)
        net = tf_util.conv2d(net, nodes_list[1], [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=True, is_training=is_training,
                              scope='mpl_g2', bn_decay=bn_decay,
                              weight_decay=weight_decay, is_dist=is_dist)
        ####both of them can work
        # net = tf.reduce_sum(net, axis=-2, keep_dims=True)
        net = tf.reduce_max(net, axis=-2, keep_dims=True)

        net = tf_util.conv2d(net, nodes_list[2], [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=True, is_training=is_training,
                              scope='mpl_f1', bn_decay=bn_decay,
                              weight_decay=weight_decay, is_dist=is_dist)
        return net

def get_triads(supervoxel_features, nn_idx, k):
    """ concatenate adjacent vertex pairs (vn, vm) and central vertex v into triads """
    batch_size = supervoxel_features.get_shape().as_list()[0]
    supervoxel_features = tf.squeeze(supervoxel_features)
    if batch_size == 1:
        supervoxel_features = tf.expand_dims(supervoxel_features, 0)
    ###
    supervoxel_shape = supervoxel_features.get_shape()
    num_upervoxels = supervoxel_shape[1].value
    num_dims = supervoxel_shape[2].value
    ###
    idx_ = tf.range(batch_size) * num_upervoxels
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])
    supervoxel_flat = tf.reshape(supervoxel_features, [-1, num_dims])
    supervoxel_neighbors = tf.gather(supervoxel_flat, nn_idx+idx_)
    ###here, we can view (supervoxel_neighbors - supervoxel_central) as new supervoxel_neighbors
    supervoxel_central = tf.expand_dims(supervoxel_features, axis=-2)
    supervoxel_central = tf.tile(supervoxel_central, [1, 1, k, 1])
    supervoxel_neighbors = supervoxel_neighbors - supervoxel_central
    ####
    #####complete permutation
    # num_vertex_pairs = int(comb(k, 2))
    # vertex_pairs_list = list(combinations(list(range(k)), 2))
    #####
    num_vertex_pairs = k
    vertex_pairs_list = [(i, i+1) for i in range(k-1)]
    vertex_pairs_list.append((k-1, 0))
    for i in range(num_vertex_pairs):
      temp_vertex_pairs = tf.concat([supervoxel_neighbors[:, :, vertex_pairs_list[i][0], :],
                                     supervoxel_neighbors[:, :, vertex_pairs_list[i][1], :]],
                                    axis=-1)
      temp_vertex_pairs = tf.expand_dims(temp_vertex_pairs, -2)
      #####
      if i == 0:
          vertex_pairs = temp_vertex_pairs
      else:
          vertex_pairs = tf.concat([vertex_pairs, temp_vertex_pairs], axis=-2)

    central_vertex = tf.expand_dims(supervoxel_features, axis=-2)
    central_vertex = tf.tile(central_vertex, [1, 1, num_vertex_pairs, 1])
    #######
    triads = tf.concat([central_vertex, vertex_pairs], axis=-1)

    return triads
