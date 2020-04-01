'''
build supervoxel-based graph G and generate the supervoxel index of multi-scale associated regions.
'''

import numpy as np
from scipy.spatial import Delaunay
from collections import defaultdict
from itertools import permutations
from sklearn.neighbors import NearestNeighbors
import random
import os

def ms_associated_region_index(data, k_nn):
    xyz = data[:, :3]
    num_points = np.size(data, 0)
    graph_G = Delaunay(xyz)
    nn = NearestNeighbors(n_neighbors=k_nn+1, algorithm='kd_tree').fit(xyz)
    _, neighbors = nn.kneighbors(xyz)

    #######
    neighbors_1st_order = defaultdict(set)
    for simplex in graph_G.vertices:
        for i, j, k in permutations(simplex, 3):
            neighbors_1st_order[i].add(j)
            neighbors_1st_order[i].add(k)
    ######
    neighbors_2nd_order = defaultdict(set)
    for i in range(num_points):
        temp_neighbors_idex = neighbors_1st_order[i]
        for j in temp_neighbors_idex:
            temp_neighbors_idex_ = neighbors_1st_order[j]
            for k in temp_neighbors_idex_:
                neighbors_2nd_order[i].add(k)

    neighbors_1st_order_np = []
    neighbors_2nd_order_np = []
    for i in range(num_points):
        temp_neighbors_idex1 = neighbors_1st_order[i]
        temp_neighbors_idex1.add(i)
        temp_neighbors_idex2_ = neighbors_2nd_order[i]
        temp_neighbors_idex2 = temp_neighbors_idex1 ^ temp_neighbors_idex2_
        temp_neighbors_idex1.remove(i)
        temp_neighbors_idex1 = list(temp_neighbors_idex1)
        temp_neighbors_idex2 = list(temp_neighbors_idex2)

        ####
        if len(temp_neighbors_idex1) >= k_nn:
            ###Supervoxel segmentation is spatially continuous...
            temp_neighbors_idex1.sort()
            temp_neighbors_idex1 = temp_neighbors_idex1[:k_nn]
            # temp_neighbors_idex1.insert(0, i)
        else:
            ###when the 1st-order has less than k neighbors
            ###use the knn as the 1st-order neighbors
            temp_neighbors_idex1 = neighbors[i][1:]

        random.shuffle(temp_neighbors_idex2)
        temp_neighbors_idex2 = temp_neighbors_idex2[:k_nn]
        ####insert central vertex
        # temp_neighbors_idex2.insert(0, i)
        #####
        neighbors_1st_order_np.append(temp_neighbors_idex1)
        neighbors_2nd_order_np.append(temp_neighbors_idex2)

    return np.array(neighbors_1st_order_np, dtype=np.int32), np.array(neighbors_2nd_order_np, dtype=np.int32)

if __name__ == "__main__":
    """
    In order to speed up the training, we can pre-generate 
    the supervoxel index of multi-scale associated region. 
    """
    data_path = '../data/training_data_features/'
    sample_list = os.listdir(data_path)
    k = 7
    ####
    neighbors_1st_order_path = '../data/training_data_features_1st_order_neighbors/'
    neighbors_2nd_order_path = '../data/training_data_features_2nd_order_neighbors/'
    if not os.path.exists(neighbors_1st_order_path): os.mkdir(neighbors_1st_order_path)
    if not os.path.exists(neighbors_2nd_order_path): os.mkdir(neighbors_2nd_order_path)
    ####
    for i in range(len(sample_list)):
        temp_data = np.load(data_path + sample_list[i])
        neighbors_1st_order, neighbors_2nd_order = ms_associated_region_index(temp_data[:, :3], k)
        np.save(neighbors_1st_order_path + sample_list[i], neighbors_1st_order)
        np.save(neighbors_2nd_order_path + sample_list[i], neighbors_2nd_order)





