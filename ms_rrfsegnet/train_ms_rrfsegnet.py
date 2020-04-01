"""
training stage
"""
import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from ms_rrfsegnet import get_ms_rrfsegnet_model, compute_class_weights, get_weight_cross_entropy_loss
import data_generator as DG

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID [default: 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_supervoxels', type=int, default=1024, help='supervoxel number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=3, help='Batch Size [default: 24]')
parser.add_argument('--classes', type=int, default=10, help='the number of class [default: 10]')
parser.add_argument('--k', type=int, default=7, help='the hyperparameter k [default: 7]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=30000, help='Decay step for lr decay [default: 30000]')
parser.add_argument('--decay_rate', type=float, default=0.98, help='Decay rate for lr decay [default: 0.98]')
parser.add_argument('--tra_data_path', default='../data/training_data_features/',
                    help='Make sure the training-data files path')
parser.add_argument('--tra_neighbors_1st_order_path', default='../data/training_data_features_1st_order_neighbors/',
                    help='Make sure the training 1st neighbors path')
parser.add_argument('--tra_neighbors_2nd_order_path', default='../data/training_data_features_2nd_order_neighbors/',
                    help='Make sure the training 2nd neighbors path')
###In experiments, the below setting should be set to the related path of validation dataset.
parser.add_argument('--val_data_path', default='../data/training_data_features/',
                    help='Make sure the validataion-data files path')
parser.add_argument('--val_neighbors_1st_order_path', default='../data/training_data_features_1st_order_neighbors/',
                    help='Make sure the validataion 1st neighbors path')
parser.add_argument('--val_neighbors_2nd_order_path', default='../data/training_data_features_2nd_order_neighbors/',
                    help='Make sure the validataion 2nd neighbors path')
FLAGS = parser.parse_args()


GPU_ID = FLAGS.gpu_id
BATCH_SIZE = FLAGS.batch_size
NUM_SUPERVOXEL = FLAGS.num_supervoxels
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
LOG_DIR = FLAGS.log_dir
NUM_CLASSES = FLAGS.classes
K = FLAGS.k
TRAIN_DATA_PATH = FLAGS.tra_data_path
TRAIN_NEIGHBORS_S1 = FLAGS.tra_neighbors_1st_order_path
TRAIN_NEIGHBORS_S2 = FLAGS.tra_neighbors_2nd_order_path

VALIDATION_PATH = FLAGS.val_data_path
VALIDATION_NEIGHBORS_S1 = FLAGS.tra_neighbors_1st_order_path
VALIDATION_NEIGHBORS_S2 = FLAGS.tra_neighbors_2nd_order_path

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

if not os.path.exists(LOG_DIR):os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def eval_iou_accuracy(batch, pred_label, gt_label, NUM_CLASSES):
    gt_classes = [0 for _ in range(NUM_CLASSES)]
    positive_classes = [0 for _ in range(NUM_CLASSES)]
    true_positive_classes = [0 for _ in range(NUM_CLASSES)]
    for i in range(batch):
        temp_pred_label = pred_label[i, :]
        temp_gt_label = gt_label[i, :]
        for j in range(gt_label.shape[1]):
            gt_l = int(temp_gt_label[j])
            pred_l = int(temp_pred_label[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l == pred_l)
    OA = sum(true_positive_classes) / float(sum(positive_classes))
    iou_list = []
    ####
    for i in range(NUM_CLASSES):
        iou = true_positive_classes[i] / float(gt_classes[i] + positive_classes[i] - true_positive_classes[i])
        iou_list.append(iou)
    avg_IoU = sum(iou_list) / NUM_CLASSES

    return OA, iou_list, avg_IoU

def one_hot(label):
    one_hot_ = np.zeros((label.shape[0], label.shape[1], NUM_CLASSES), dtype=label.dtype)
    for i in range(NUM_CLASSES):
        one_hot_[:, :, i] = (label == i)
    return one_hot_

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        batch = tf.Variable(0, trainable=False)
        bn_decay = get_bn_decay(batch)
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:%d'%GPU_ID):
                #### xyz (3D) + supervoxel embedding represenation (16D) = 19D
                input_supervoxels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_SUPERVOXEL, 19))
                #### one-hot labels
                labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_SUPERVOXEL, NUM_CLASSES))
                #### 1st-order k neighbors and central vertex
                knn_idx1 = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_SUPERVOXEL, K))
                #### 2nd-order k neighbors and central vertex
                knn_idx2 = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_SUPERVOXEL, K))
                ####train or verify
                is_training = tf.placeholder(tf.bool, shape=())
                #### the loss weighting terms
                class_weight = tf.placeholder(tf.float32, shape=[NUM_CLASSES])
                #### get the ms_rrfsegnet model
                pred = get_ms_rrfsegnet_model(input_supervoxels,
                                              NUM_CLASSES,
                                              is_training,
                                              knn_idx1,
                                              knn_idx2,
                                              k=K,
                                              weight_decay=0.0001,
                                              bn_decay=bn_decay)
                #####loss function
                weifht_CE_loss = get_weight_cross_entropy_loss(class_weight, pred, labels)
                loss = weifht_CE_loss + tf.add_n(tf.get_collection('losses'))
                ###optimizers
                learning_rate = get_learning_rate(batch)
                if OPTIMIZER == 'Momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(loss, global_step=batch)
        ####only save one trained model
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        init = tf.group(tf.global_variables_initializer())
        sess.run(init)

        ops = {'input_supervoxels': input_supervoxels,
               'labels': labels,
               'is_training': is_training,
               'class_weight': class_weight,
               'knn_idx1': knn_idx1,
               'knn_idx2': knn_idx2,
               'pred': pred,
               'loss': loss,
               'train_op': train_op}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            ####prepare batch training data
            tra_set = DG.get_files_set(TRAIN_DATA_PATH)
            val_set = DG.get_files_set(VALIDATION_PATH)
            tra_generator = DG.minibatch_data_generator_for_training(TRAIN_DATA_PATH,
                                                                     TRAIN_NEIGHBORS_S1,
                                                                     TRAIN_NEIGHBORS_S2,
                                                                     BATCH_SIZE,
                                                                     tra_set)

            val_generator = DG.minibatch_data_generator_for_training(VALIDATION_PATH,
                                                                     VALIDATION_NEIGHBORS_S1,
                                                                     VALIDATION_NEIGHBORS_S2,
                                                                     BATCH_SIZE,
                                                                     val_set)
            ####
            train_one_epoch(sess, tra_set, tra_generator, ops)
            # val_one_epoch(sess, val_set, val_generator, ops)

            # Save the variables to disk.
            save_path = saver.save(sess, os.path.join(LOG_DIR, 'epoch_' + str(epoch) + '.ckpt'))
            log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, tra_set, tra_generator, ops):
    """training stage"""
    num_batches = len(tra_set) // BATCH_SIZE
    total_correct_training = 0.0
    total_seen_training = 0.0
    loss_sum_training = 0.0
    print('---------------start to train---------------')
    for _ in tqdm(range(num_batches)):
        batch_train_data, batch_train_label, k_idx1, k_idx2 = next(tra_generator)
        weights = compute_class_weights(batch_train_label, NUM_CLASSES)
        one_hot_labels = one_hot(batch_train_label)
        feed_dict = {ops['input_supervoxels']: batch_train_data,
                     ops['labels']: one_hot_labels,
                     ops['is_training']: True,
                     ops['class_weight']: weights,
                     ops['knn_idx1']: k_idx1,
                     ops['knn_idx2']: k_idx2}
        _, loss, pred_training,  = sess.run([ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        pred_training = np.argmax(pred_training, 2)
        correct_training = np.sum(pred_training == batch_train_label)
        total_correct_training += correct_training
        total_seen_training += (BATCH_SIZE * NUM_SUPERVOXEL)
        loss_sum_training += loss
    log_string('training loss: %f, training OA: %f' % (loss_sum_training / float(num_batches), total_correct_training / float(total_seen_training)))

def val_one_epoch(sess, val_set, val_generator, ops):
    """validation stage"""
    num_batches_val = len(val_set) // BATCH_SIZE
    loss_sum_val = 0.0
    total_correct_val = 0.0
    total_seen_val = 0.0
    print('--------------start to verify----------------')
    for val_batch in tqdm(range(num_batches_val)):
        batch_val_data, batch_val_label, k_idx1, k_idx2 = next(val_generator)
        weights = compute_class_weights(batch_val_label, NUM_CLASSES)
        one_hot_labels = one_hot(batch_val_label)
        feed_dict = {ops['input_supervoxels']: batch_val_data,
                     ops['labels']: one_hot_labels,
                     ops['is_training']: False,
                     ops['class_weight']: weights,
                     ops['knn_idx1']: k_idx1,
                     ops['knn_idx2']: k_idx2}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 2)
        correct_val = np.sum(pred_val == batch_val_label)
        loss_sum_val += loss_val
        total_correct_val += correct_val
        total_seen_val += (BATCH_SIZE * NUM_SUPERVOXEL)
        ####
        if val_batch == 0:
            all_batch_pre = pred_val
            all_batch_label = batch_val_label
        else:
            all_batch_pre = np.concatenate([all_batch_pre, pred_val], axis=0)
            all_batch_label = np.concatenate([all_batch_label, batch_val_label], axis=0)
    #####
    OA, iou_list, mIoU = eval_iou_accuracy(num_batches_val, all_batch_pre, all_batch_label, NUM_CLASSES)
    log_string('val_loss: %f, OA: %f, mIoU:%f ' %(loss_sum_val / float(num_batches_val), OA, mIoU))


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
