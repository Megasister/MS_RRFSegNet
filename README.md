# MS-RRFSegNet

This repo contains the source code for our paper that has been published by IEEE Transactions on Geoscience and Remote Sensing (Early Access):
[**MS-RRFSegNet: Multi-Scale Regional Relation Feature Segmentation Network for Semantic Segmentation of Urban Scene Point Clouds**](https://ieeexplore.ieee.org/document/9080553/authors#authors)
<br>
IEEE Transactions on Geoscience and Remote Sensing
<br>


## Introduction
In this paper, we propose a novel method for urban scene point cloud semantic segmentation using deep learning. Firstly, we use homogeneous supervoxels to reorganize raw point clouds to effectively reduce the computational complexity and improve the non-uniform distribution. Then, we use supervoxels as basic processing units, which can further expand receptive fields to obtain more descriptive contexts. Next, a sparse auto-encoder (SAE) is presented for feature embedding representations of the supervoxels. Subsequently, we propose a regional relation feature reasoning module (RRFRM) inspired by [relation reasoning network](https://arxiv.org/abs/1706.01427) and design a multi-scale regional relation feature segmentation network (MS-RRFSegNet) based on the RRFRM to semantically label supervoxels. Finally, the supervoxel-level inferences are transformed into point-level fine-grained predictions.

## Dataset
Our proposed method has been evaluated on two open benchmarks -- [Paris-Lille-3D](https://npm3d.fr/paris-lille-3d) and [Semantic3D](http://www.semantic3d.net/).


## Requirements
- Python 3.6
- Tensorflow 1.15

## Code Structure
* `./auto_encoder/*` - Supervoxl embedding representation code.
* `./ms_rrfsegnet/*` - RRFRM and MS-RRFSegNet code.
* Note that the details of supervoxl segmentation can be found in the [Supervoxel-for-3D-point-clouds](https://github.com/yblin/Supervoxel-for-3D-point-clouds).
We also provide three training supervoxel samples and one testing supervoxel sample (`./data/*` ) for testing the code. Training data format: [x, y, z, r, g, b, supervoxel_id, training_label]. Testing data format: [x, y, z, r, g, b, supervoxel_id].

## Perform Experiments
* Download dataset, and then generate samples according to the related description in [our paper]().
* Over segment samples into supervoxels like the examples in `./data/*`.
* Run `./auto_encoder/train_SAE.py` to train a SAE model for supervoxel embedding representation (SER). 
* Run `./auto_encoder/ser.py` to convert all samples into SER.
* Run `./ms_rrfsegnet/ms_associated_region.py` to build supervoxel-based graph G and generate the supervoxel index of multi-scale associated regions. 
* Run `./ms_rrfsegnet/train_ms_rrfsegnet.py` to train MS-RRFSegNet for supervoxel-level semantically labeling.
* Run `./ms_rrfsegnet/test_spl2pl.py` to test the trained MS-RRFSegNet and transform supervoxel-level predictions into dense point-level results.



## Citation
If you find this project useful for your research, please kindly cite our paper:

Our paper is coming soon...

## Acknowledgment
We would like to acknowledge the provision of reference code by [Charles R. Qi](https://github.com/charlesq34/pointnet) and [Yue Wang](https://github.com/WangYueFt/dgcnn), respectively.

## Contact
If you have any questions, please contact [Haifeng Luo](h.feng.luo@outlook.com).
