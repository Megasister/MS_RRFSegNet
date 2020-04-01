"""
testing stage--transform supervoxel-level prediction into dense point-level results.
"""
import argparse
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from ms_rrfsegnet import get_ms_rrfsegnet_model
import data_generator as DG

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID [default: 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_supervoxels', type=int, default=1024, help='supervoxel number [default: 1024]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size [default: 1]')
parser.add_argument('--classes', type=int, default=10, help='the number of class [default: 10]')
parser.add_argument('--k', type=int, default=7, help='the hyperparameter k [default: 7]')
parser.add_argument('--test_supervoxel_path', default='../data/testing_data/',
                    help='Make sure the test-supervoxel files path')
parser.add_argument('--test_supervoxel_feature_path', default='../data/testing_data_features/',
                    help='Make sure the test-supervoxel-feature files path')
parser.add_argument('--test_result_path', default='../data/results/',
                    help='Make sure the test results path')
FLAGS = parser.parse_args()

GPU_ID = FLAGS.gpu_id
NUM_SUPERVOXEL = FLAGS.num_supervoxels
BATCH_SIZE = FLAGS.batch_size
TRAINED_MODEL_PATH = FLAGS.log_dir
NUM_CLASSES = FLAGS.classes
K = FLAGS.k
TEST_SUPERVOXEL_PATH = FLAGS.test_supervoxel_path
TEST_SUPERVOXEL_FEATURE_PATH = FLAGS.test_supervoxel_feature_path
TEST_RESULT_PATH = FLAGS.test_result_path
if not os.path.exists(TEST_RESULT_PATH): os.mkdir(TEST_RESULT_PATH)

####
def test():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        with tf.device('/gpu:%d'%GPU_ID):
            #### xyz (3D) + supervoxel embedding represenation (16D) = 19D
            input_supervoxels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_SUPERVOXEL, 19))
            #### 1st-order k neighbors and central vertex
            knn_idx1 = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_SUPERVOXEL, K))
            #### 2nd-order k neighbors and central vertex
            knn_idx2 = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_SUPERVOXEL, K))
            ####train or verify
            is_training = tf.placeholder(tf.bool, shape=())
            #### get the ms_rrfsegnet model
            pred = get_ms_rrfsegnet_model(input_supervoxels,
                                          NUM_CLASSES,
                                          is_training,
                                          knn_idx1,
                                          knn_idx2,
                                          k=K)

        saver = tf.train.Saver(tf.global_variables())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        saver.restore(sess, tf.train.latest_checkpoint(TRAINED_MODEL_PATH))

        ####
        test_set = os.listdir(TEST_SUPERVOXEL_FEATURE_PATH)
        test_generator = DG.minibatch_data_generator_for_testing(TEST_SUPERVOXEL_FEATURE_PATH, BATCH_SIZE, test_set, K)
        ###
        for i in tqdm(range(len(test_set))):

            test_supervoxel_feature, supervoxel_id, k_idx1, k_idx2, file_list = next(test_generator)
            ####
            temp_filename, _ = os.path.splitext(file_list[0])
            dense_points_supervoxels = np.loadtxt(TEST_SUPERVOXEL_PATH + temp_filename + '.txt')
            dense_supervoxel_id = dense_points_supervoxels[:, -1]
            dense_points_prediction = np.ones_like(dense_supervoxel_id)
            ####
            feed_dict = {input_supervoxels: test_supervoxel_feature,
                         is_training: False,
                         knn_idx1: k_idx1,
                         knn_idx2: k_idx2}

            pred_ = sess.run(pred, feed_dict=feed_dict)
            pred_ = np.squeeze(np.argmax(pred_, 2))

            #### transform supervoxel-level prediction into dense point-level results
            for k in range(len(supervoxel_id[0])):
                temp_supervoxel_id = supervoxel_id[0][k]
                temp_index = np.where(dense_supervoxel_id == temp_supervoxel_id)
                dense_points_prediction[temp_index[0]] = pred_[k]

            dense_points_prediction = np.expand_dims(dense_points_prediction, 1)
            dense_prediction = np.concatenate([dense_points_supervoxels[:, :3], dense_points_prediction], -1)
            np.savetxt(TEST_RESULT_PATH + 'dense_result_sample%d.txt' % i, dense_prediction)

if __name__ == "__main__":
    test()

