import pickle
from pathlib import Path

import numpy as np

from second.core import box_np_ops
from second.data.dataset import Dataset, get_dataset_class
from second.data.kitti_dataset import KittiDataset
import second.data.nuscenes_dataset as nuds
from second.utils.progress_bar import progress_bar_iter as prog_bar

from concurrent.futures import ProcessPoolExecutor

def create_groundtruth_database(dataset_class_name,
                                data_path,
                                info_path=None,
                                used_classes=None,
                                database_save_path=None,
                                db_info_save_path=None,
                                relative_path=True,
                                add_rgb=False,
                                lidar_only=False,
                                bev_only=False,
                                coors_range=None):
    dataset = get_dataset_class(dataset_class_name)(
        info_path=info_path,
        root_path=data_path,
    )
    root_path = Path(data_path)
    if database_save_path is None:
        database_save_path = root_path / 'gt_database'
    else:
        database_save_path = Path(database_save_path)
    if db_info_save_path is None:
        db_info_save_path = root_path / "dbinfos_train.pkl"
    database_save_path.mkdir(parents=True, exist_ok=True)
    all_db_infos = {}

    group_counter = 0
    for j in prog_bar(list(range(len(dataset)))):
        image_idx = j
        sensor_data = dataset.get_sensor_data(j)
        if "image_idx" in sensor_data["metadata"]:
            image_idx = sensor_data["metadata"]["image_idx"]

        points = sensor_data["lidar"]["points"]
        annos = sensor_data["lidar"]["annotations"]
        gt_boxes = annos["boxes"]

        names = annos["names"]
        group_dict = {}
        group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            filepath = database_save_path / filename
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes[i, :3]
            with open(filepath, 'w') as f:
                gt_points.tofile(f)
            if (used_classes is None) or names[i] in used_classes:
                if relative_path:
                    db_path = str(database_save_path.stem + "/" + filename)
                else:
                    db_path = str(filepath)
                db_info = {
                    "name": names[i],
                    "path": db_path,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    # "group_id": -1,
                    # "bbox": bboxes[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)

def create_groundtruth_database_with_sweep_info(dataset_class_name,
                                                data_path,
                                                info_path=None,
                                                used_classes=None,
                                                database_save_path=None,
                                                db_info_save_path=None,
                                                relative_path=True,
                                                add_rgb=False,
                                                lidar_only=False,
                                                bev_only=False,
                                                coors_range=None):
    dataset = get_dataset_class(dataset_class_name)(
        info_path=info_path,
        root_path=data_path,
    )
    root_path = Path(data_path)
    if database_save_path is None:
        database_save_path = root_path / 'gt_database_with_sweep_info'
    else:
        database_save_path = Path(database_save_path)
    if db_info_save_path is None:
        db_info_save_path = root_path / "dbinfos_train_with_sweep_info.pkl"
    database_save_path.mkdir(parents=True, exist_ok=True)
    all_db_infos = {}

    group_counter = 0
    for j in prog_bar(list(range(len(dataset)))):
        image_idx = j
        sensor_data = dataset.get_sensor_data(j)
        if "image_idx" in sensor_data["metadata"]:
            image_idx = sensor_data["metadata"]["image_idx"]

        assert("sweep_points_list" in sensor_data["lidar"])
        sweep_points_list = sensor_data["lidar"]["sweep_points_list"]

        assert("sweep_annotations" in sensor_data["lidar"])
        sweep_gt_boxes_list = sensor_data["lidar"]["sweep_annotations"]["boxes_list"]
        sweep_gt_tokens_list = sensor_data["lidar"]["sweep_annotations"]["tokens_list"]
        sweep_gt_names_list = sensor_data["lidar"]["sweep_annotations"]["names_list"]

        # we focus on the objects in the first frame
        # and find the bounding box index of the same object in every frame
        points = sensor_data["lidar"]["points"]
        annos = sensor_data["lidar"]["annotations"]
        names = annos["names"]
        gt_boxes = annos["boxes"]
        attrs = annos["attrs"]
        tokens = sweep_gt_tokens_list[0]

        # sanity check with redundancy
        assert(len(sweep_gt_boxes_list) == len(sweep_gt_tokens_list) == len(sweep_gt_names_list))
        assert(len(gt_boxes) == len(attrs) == len(tokens))
        assert(np.all(names == sweep_gt_names_list[0]))
        assert(np.all(gt_boxes == sweep_gt_boxes_list[0]))

        # on every frame, we mask points inside each bounding box
        # but we are not looking at every bounding box 
        sweep_point_indices_list = []
        for sweep_points, sweep_gt_boxes in zip(sweep_points_list, sweep_gt_boxes_list):
            sweep_point_indices_list.append(box_np_ops.points_in_rbbox(sweep_points, sweep_gt_boxes))

        # crop point cloud based on boxes in the current frame
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)

        # 
        group_dict = {}
        group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)

        #
        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes.shape[0]
        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            filepath = database_save_path / filename
            
            # only use non-key frame boxes when the object is moving
            if attrs[i] in ['vehicle.moving', 'cycle.with_rider', 'pedestrian.moving']:
                gt_points_list = []
                for t in range(len(sweep_points_list)):
                    # fast pass for most frames
                    if i < len(sweep_gt_tokens_list[t]) and sweep_gt_tokens_list[t][i] == tokens[i]:
                        box_idx = i
                    else:
                        I = np.flatnonzero(tokens[i] == sweep_gt_tokens_list[t])
                        if len(I) == 0: continue 
                        elif len(I) == 1: box_idx = I[0]
                        else: raise ValueError('Identical object tokens')
                    gt_points_list.append(sweep_points_list[t][sweep_point_indices_list[t][:, box_idx]])
                gt_points = np.concatenate(gt_points_list, axis=0)[:, [0, 1, 2, 4]]
            else: 
                gt_points = points[point_indices[:, i]]

            # offset points based on the bounding box in the current frame
            gt_points[:, :3] -= gt_boxes[i, :3]

            with open(filepath, 'w') as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                if relative_path: 
                    db_path = str(database_save_path.stem + "/" + filename)
                else: 
                    db_path = str(filepath)
                db_info = {
                    "name": names[i],
                    "path": db_path, 
                    "image_idx": image_idx, 
                    "gt_idx": i, 
                    "box3d_lidar": gt_boxes[i], 
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i]
                }
                local_group_id = group_ids[i]
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)

