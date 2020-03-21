import copy
from pathlib import Path
import pickle

import fire

import second.data.kitti_dataset as kitti_ds
import second.data.nuscenes_dataset as nu_ds
from second.data.all_dataset import create_groundtruth_database, create_groundtruth_database_with_sweep_info

import numpy as np
from tqdm import trange

def kitti_data_prep(root_path):
    kitti_ds.create_kitti_info_file(root_path)
    kitti_ds.create_reduced_point_cloud(root_path)
    create_groundtruth_database("KittiDataset", root_path, Path(root_path) / "kitti_infos_train.pkl")

def kitti_gt_fgm_data_prep(
    old_root_path, old_trainval_info_path, new_root_path, new_train_info_path
): 
    from second.core import box_np_ops 
    from second.data.dataset import get_dataset_class
    dataset = get_dataset_class('KittiDataset')(
        root_path = old_root_path, 
        info_path = old_trainval_info_path
    )
    for i in trange(len(dataset)):
        image_idx = i
        sensor_data = dataset.get_sensor_data(i)
        if 'image_idx' in sensor_data['metadata']:
            image_idx = sensor_data['metadata']['image_idx']
        points = sensor_data['lidar']['points']
        annos = sensor_data['lidar']['annotations']
        gt_boxes = annos['boxes']
        gt_mask = box_np_ops.points_in_rbbox(points, gt_boxes)
        points_aug = np.concatenate((points, gt_mask.max(axis=1, keepdims=True)), axis=1)
        points_aug = points_aug.astype(np.float32)
        velo_file = 'training/velodyne_reduced/%06d.bin' % (image_idx)
        with open(f'{new_root_path}/{velo_file}', 'w') as f: 
            points_aug.tofile(f)
    create_groundtruth_database(
        dataset_class_name='KittiFGMDataset', 
        data_path=new_root_path, 
        info_path=new_train_info_path
    )

def nuscenes_data_prep(root_path, version, dataset_name, max_sweeps=10):
    nu_ds.create_nuscenes_infos(root_path, version=version, max_sweeps=max_sweeps)
    name = "infos_train.pkl"
    if version == "v1.0-test":
        name = "infos_test.pkl"
    # create_groundtruth_database(dataset_name, root_path, Path(root_path) / name)

def nuscenes_data_prep_with_sweep_info(root_path, version, dataset_name, max_sweeps=10):
    # nu_ds.create_nuscenes_infos(root_path, version=version, max_sweeps=max_sweeps, with_sweep_info=True)
    name = "infos_train_with_sweep_info.pkl"
    if version == "v1.0-test": 
        name = "infos_test.pkl"  # because we will not have annotations
    create_groundtruth_database_with_sweep_info(dataset_name, root_path, Path(root_path) / name)


if __name__ == '__main__':
    fire.Fire()
