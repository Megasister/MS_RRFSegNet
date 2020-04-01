'''
generate data for training SAE
'''
import numpy as np
import random
import os

def load_data(data_path):
    files = os.listdir(data_path)
    for i, file_name in enumerate(files):
        try:
            temp_data = np.loadtxt(data_path + file_name)
        except:
            temp_data = np.load(data_path + file_name)
        temp_xyz = temp_data[:, :3]
        temp_voxel_label = temp_data[:, 6]

        if i == 0:
            xyz = temp_xyz
            voxel_label = temp_voxel_label
            unique_label = np.unique(voxel_label)
            label_size = np.size(unique_label, 0)
        else:
            xyz = np.concatenate([xyz, temp_xyz], 0)
            temp_voxel_label = temp_voxel_label + label_size
            voxel_label = np.concatenate([voxel_label, temp_voxel_label], 0)
            unique_label = np.unique(voxel_label)
            label_size = np.size(unique_label, 0)

    return xyz, voxel_label, unique_label, label_size

def training_data_generator(xyz, batch_size, point_num,
                            voxel_label, unique_label, label_size):
    while True:
        batch = 0
        for _ in (range(label_size)):
            batch += 1
            random_i = random.randint(0, label_size-1)
            index = np.where(voxel_label == unique_label[random_i])
            temp_data = xyz[index, :]
            temp_data = np.squeeze(temp_data)
            temp_data_size = np.size(temp_data, 0)

            #for SP only has one point
            if temp_data_size == np.size(temp_data):
                temp_data = np.expand_dims(temp_data, 0)
                temp_data_size = np.size(temp_data, 0)

            temp_data_mean = np.mean(temp_data, 0)
            temp_data = temp_data - temp_data_mean

            if temp_data_size > point_num:
                temp_data = temp_data[:point_num, :]

            elif temp_data_size < point_num:
                deta_point_num = point_num - temp_data_size
                zeros_temp = np.zeros([deta_point_num, 3])
                temp_data = np.concatenate([temp_data, zeros_temp], 0)

            temp_data = np.expand_dims(temp_data, 0)
            if batch == 1:
                train_data = temp_data
            else:
                train_data = np.concatenate([train_data, temp_data], 0)

            if batch % batch_size == 0:
                yield train_data
                train_data = []
                batch = 0

