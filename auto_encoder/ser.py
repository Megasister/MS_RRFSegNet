'''
supervoxel_embedding_representation_inference
'''
import numpy as np
import tensorflow as tf
import autoencoder as AE
from tqdm import tqdm
import os
from collections import Counter

def supervoxel_embedding_representation_inference(Folder_list, Input_Path,
                                                  Output_path, MODEL_PATH, training):

    input_pointclouds = tf.placeholder(tf.float32, shape=(1, None, 3))
    is_training_pl = tf.placeholder(tf.bool, shape=())
    # Get model
    encoder, _ = AE.get_model_SAE(input_pointclouds, is_training_pl)
    saver = tf.train.Saver()
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))

    for i in range(len(Folder_list)):
        folder = Folder_list[i]
        out_path = Output_path + folder + '_features/'
        if not os.path.isdir(out_path): os.mkdir(out_path)
        ####
        path = Input_Path + folder + '/'
        files_list = os.listdir(path)
        for j in range(len(files_list)):
            ###training smaple (xyz_rgb_sp_id_label); testing sample (xyz_rgb_sp_id)
            data = np.loadtxt(path + files_list[j])
            filename, _ = os.path.splitext(files_list[j])

            if training:
                xyz = data[:, :3]
                voxel_label = data[:, 6] ###sp_id
                training_label = data[:, 7] ###label
                unique_label = np.unique(voxel_label)
                label_size = np.size(unique_label, 0)

                for k in tqdm(range(int(label_size))):
                    index = np.where(voxel_label == unique_label[k])
                    temp_data = xyz[index, :]
                    temp_position = np.mean(temp_data, 1)
                    temp_data -= temp_position
                    ###
                    temp_training_labels = training_label[index]
                    ####integrate training label
                    count = Counter(temp_training_labels)
                    temp_training_label = count.most_common(1)[0][0]
                    temp_training_label = np.expand_dims(np.expand_dims(np.array(temp_training_label), -1), 0)

                    encoder_ = sess.run(encoder, feed_dict={input_pointclouds: temp_data, is_training_pl: False})
                    encoder_ = np.squeeze(encoder_)
                    encoder_ = np.expand_dims(encoder_, 0)

                    features_label = np.concatenate([temp_position, encoder_, temp_training_label], -1)
                    if k == 0:
                        features = features_label
                    else:
                        features = np.concatenate([features, features_label], 0)
                np.save(out_path + filename + '.npy', features)

            else:
                xyz = data[:, :3]
                voxel_label = data[:, 6]  ###sp_id
                unique_label = np.unique(voxel_label)
                label_size = np.size(unique_label, 0)
                for k in tqdm(range(int(label_size))):
                    index = np.where(voxel_label == unique_label[k])
                    temp_data = xyz[index, :]
                    temp_position = np.mean(temp_data, 1)
                    temp_data -= temp_position
                    encoder_ = sess.run(encoder, feed_dict={input_pointclouds: temp_data, is_training_pl: False})
                    encoder_ = np.squeeze(encoder_)
                    encoder_ = np.expand_dims(encoder_, 0)

                    temp_voxel_label = np.expand_dims(np.expand_dims(unique_label[k], -1), -1)
                    features_label = np.concatenate([temp_position, encoder_, temp_voxel_label], -1)

                    if k == 0:
                        features = features_label
                    else:
                        features = np.concatenate([features, features_label], 0)
                np.save(out_path + filename + '.npy', features)


if __name__ == "__main__":

    Input_Path = '../data/'
    Output_path = '../data/'
    MODEL_PATH = './log/'

    Folder_list1 = ['training_data']
    supervoxel_embedding_representation_inference(Folder_list1, Input_Path,
                                                  Output_path, MODEL_PATH, training=True)
    # Folder_list2 = ['testing_data']
    # supervoxel_embedding_representation_inference(Folder_list2, Input_Path,
    #                                               Output_path, MODEL_PATH, training=False)
