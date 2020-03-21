import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='nuscenes')
parser.add_argument('--step', type=int, default='-1')
parser.add_argument('--metric', type=str, default='mean_dist_aps')
parser.add_argument('--thresh', type=str, default="")
args = parser.parse_args()

classes = [
    'car', 'pedestrian', 'barrier', 'traffic_cone', 'truck', 'bus', 'trailer', 'construction_vehicle', 'motorcycle', 'bicycle'
]

name = "freespace"
res_file = f"utils/test_results.json"
if os.path.exists(res_file):
    with open(res_file, 'r') as f:
        summary = json.load(f)

print(summary)

# delim = '\t'
delim = ' & '
metric = args.metric

print('{:24}'.format(f'mAP[{args.thresh}]'), end=delim)
for cls in classes:
    print('{:5}'.format(cls[:5]), end=delim)
print('{:5}'.format('avg'))

print('{:24}'.format(name), end=delim)
APs = []
for cls in classes:
    n = summary[metric][cls]
    if args.thresh in n:
        AP = n[args.thresh]
    else:
        AP = sum(n.values())/len(n)
    APs.append(AP)
    print('{:.3f}'.format(AP), end=delim)
mAP = sum(APs)/len(APs)
print('{:.3f}'.format(mAP))
