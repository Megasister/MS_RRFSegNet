'''
train sparse auto encoder
'''
import argparse
import tensorflow as tf
import os
import autoencoder as AE
import data_generator as DG
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='the id of GPU [default: 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=150, help='the number of points in SP [default: 150]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=256, help='Batch Size [default: 256]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.01]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=30000, help='Decay step for lr decay [default: 30000]')
parser.add_argument('--decay_rate', type=float, default=0.95, help='Decay rate for lr decay [default: 0.95]')
parser.add_argument('--training_data_path', default='../data/training_data/',
                    help='Make sure the training-data files path')

FLAGS = parser.parse_args()

GPU_ID = FLAGS.gpu_id
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BATCH_SIZE = FLAGS.batch_size
BASE_LEARNING_RATE = FLAGS.learning_rate
LOG_DIR = FLAGS.log_dir
OPTIMIZER = FLAGS.optimizer
MOMENTUM = FLAGS.momentum
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
TRAIN_DATA_PATH = FLAGS.training_data_path

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.0001)  # CLIP THE LEARNING RATE!
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

    with tf.device('/gpu:%d' %GPU_ID):
        input_pointclouds = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
        reconsture_pointclouds = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
        is_training_pl = tf.placeholder(tf.bool, shape=())
        batch = tf.Variable(0)
        # Get model and loss
        encoder, decoder = AE.get_model_SAE(input_pointclouds, is_training_pl)
        loss = AE.get_loss_sparsity(encoder, decoder, reconsture_pointclouds, 0.01, 0.05)

        ####
        learning_rate = get_learning_rate(batch)
        if OPTIMIZER == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
        elif OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=batch)
        ###
        ###only save one model
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)

        ####prepare training data
        xyz, voxel_label, unique_label, label_size = DG.load_data(TRAIN_DATA_PATH)
        training_data_generator = DG.training_data_generator(xyz, BATCH_SIZE,
                                                             NUM_POINT, voxel_label,
                                                             unique_label, label_size)

        print('-----------------start to train---------------------')
        lowest_loss = 10000.0
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))

            loss_all = 0
            for _ in tqdm(range(int(label_size/BATCH_SIZE))):

                train_data = next(training_data_generator)
                feed_dict = {input_pointclouds: train_data,
                             reconsture_pointclouds: train_data,
                             is_training_pl: True}

                _, loss_ = sess.run([train_op, loss], feed_dict=feed_dict)
                loss_all += loss_

            avg_loss = loss_all/int(label_size/BATCH_SIZE)
            log_string('epoch %d, loss %.6f ' % (epoch, avg_loss))
            # Save model to disk.
            if lowest_loss > avg_loss:
                lowest_loss = avg_loss
            save_path = saver.save(sess, os.path.join(LOG_DIR, 'epoch_' + str(epoch) + '.ckpt'))
            log_string('Model saved in file: %s' %save_path)

if __name__ == "__main__":
    train()
    LOG_FOUT.close()