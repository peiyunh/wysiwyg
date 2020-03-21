import sys
import pickle
from second.data.kitti_dataset import KittiDataset

split = "val"

root_path = '/data/kitti/object'
info_path = f'{root_path}/kitti_infos_{split}.pkl'

model_name = "pointpillars_car_vfn_oc_xyres_16"
eval_step = 296960

dataset = KittiDataset(root_path, info_path, class_names=["Car"])
model_dir = f'models/kitti/{model_name}'
result_dir = f'{model_dir}/{split}_results/step_{eval_step}'
result_path = f'{result_dir}/result.pkl'

with open(result_path, 'rb') as f:
    detections = pickle.load(f)
    
dataset.evaluation(detections, result_dir)