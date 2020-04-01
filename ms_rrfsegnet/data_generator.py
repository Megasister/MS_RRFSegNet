import random
import numpy as np
import os
from ms_associated_region import ms_associated_region_index

def load_data(path):
    try:
        sample_data = np.loadtxt(path)
    except:
        sample_data = np.load(path)
    return sample_data

def get_files_set(data_path):
    files_set = os.listdir(data_path)
    random.shuffle(files_set)
    return files_set

def minibatch_data_generator_for_training(data_path, neighbors_1st_order_path, neighbors_2nd_order_path, batch_size, train_set):
    while True:
        train_data = []
        train_label = []
        first_order_neighbors = []
        second_order_neighbors = []
        batch = 0
        for i in (range(len(train_set))):
            batch += 1
            url = train_set[i]
            temp_point_set = load_data(data_path+url)
            neighbors_1st_order = load_data(neighbors_1st_order_path + url)
            neighbors_2nd_order = load_data(neighbors_2nd_order_path + url)
            ###
            training_points = temp_point_set[:, :-1]
            xyz = training_points[:, :3]
            temp_xyz_min = np.min(xyz, 0)
            temp_xyz_max = np.max(xyz, 0)
            xyz = (xyz - temp_xyz_min) / (temp_xyz_max - temp_xyz_min)
            training_points = np.concatenate([xyz, training_points[:, 3:]], axis=-1)
            ###
            label_points = temp_point_set[:, -1]
            ###
            train_data.append(training_points)
            train_label.append(label_points)
            first_order_neighbors.append(neighbors_1st_order)
            second_order_neighbors.append(neighbors_2nd_order)

            if batch % batch_size == 0:
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                first_order_neighbors = np.array(first_order_neighbors)
                second_order_neighbors = np.array(second_order_neighbors)
                yield [train_data, train_label, first_order_neighbors, second_order_neighbors]
                train_data = []
                train_label = []
                first_order_neighbors = []
                second_order_neighbors = []
                batch = 0


def minibatch_data_generator_for_testing(data_path, batch_size, test_set, k):

    while True:
        test_data = []
        file_list = []
        supervoxel_id = []
        first_order_neighbors = []
        second_order_neighbors = []

        batch = 0
        for i in (range(len(test_set))):
            batch += 1
            url = test_set[i]
            temp_data_set = load_data(data_path+url)
            #### xyz + 19D supervoxel embedding representation + supervoxel_id
            test_supervoxels = temp_data_set[:, :-1]
            temp_supervoxel_id = temp_data_set[:, -1]
            ####Normalization
            xyz = test_supervoxels[:, :3]
            temp_xyz_min = np.min(xyz, 0)
            temp_xyz_max = np.max(xyz, 0)
            xyz = (xyz - temp_xyz_min) / (temp_xyz_max - temp_xyz_min)
            test_supervoxels = np.concatenate([xyz, test_supervoxels[:, 3:]], axis=-1)
            ###generate multi-scale associated region
            neighbors_1st_order, neighbors_2nd_order = ms_associated_region_index(xyz, k)
            ###
            test_data.append(test_supervoxels)
            file_list.append(url)
            supervoxel_id.append(temp_supervoxel_id)
            first_order_neighbors.append(neighbors_1st_order)
            second_order_neighbors.append(neighbors_2nd_order)

            if batch % batch_size == 0:
                test_data = np.array(test_data)
                supervoxel_id = np.array(supervoxel_id)
                first_order_neighbors = np.array(first_order_neighbors)
                second_order_neighbors = np.array(second_order_neighbors)

                yield [test_data, supervoxel_id, first_order_neighbors, second_order_neighbors, file_list]
                test_data = []
                file_list = []
                supervoxel_id = []
                first_order_neighbors = []
                second_order_neighbors = []
                batch = 0

