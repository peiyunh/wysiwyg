# This function needs to be called in SECOND's environment

import sys
import pickle
from second.data.nuscenes_dataset import NuScenesDataset

split = 'val' if len(sys.argv) <= 3 else sys.argv[3]

model_name = sys.argv[1]
if len(sys.argv) > 2:
    eval_step = int(sys.argv[2])
else:
    eval_step = 58650

root_path = '/data/nuscenes'
if split == 'val':
    info_path = f'{root_path}/infos_val.pkl'
else:
    info_path = f'{root_path}/infos_test.pkl'

# THIS ORDER IS IMPORTANT AS IT TRANSLATES ANCHOR INDEX TO CLASS LABELS
# IT IS TIED WITH THE ORDER CLASSES ARE DEFINED INSIDE THE CONFIG FILE
class_names = ['bus', 'car', 'construction_vehicle', 'trailer', 'truck',
                'barrier', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone']

dataset = NuScenesDataset(root_path, info_path, class_names=class_names)
model_dir = f'models/nuscenes/{model_name}'
if split == 'val':
    result_dir = f'{model_dir}/results/step_{eval_step}'
else:
    result_dir = f'{model_dir}/eval_results/step_{eval_step}'

result_path = f'{result_dir}/result.pkl'
print(f'loading results from {result_path}')
with open(result_path, 'rb') as f:
    detections = pickle.load(f)

dataset.evaluation_nusc(detections, result_dir, clean_after=False)
