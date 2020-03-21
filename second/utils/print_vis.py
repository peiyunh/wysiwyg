import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='nuscenes')
parser.add_argument('--step', type=int, default='-1')
parser.add_argument('--metric', type=str, default='label_aps')
parser.add_argument('--thresh', type=str, default="")
args = parser.parse_args()

dataset = args.dataset
base_dir = f'models/{dataset}'

classes = [
    # # official class order
    # 'car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
    # # sorted by percentage
    # 'car', 'pedestrian', 'barrier', 'traffic_cone', 'truck', 'bus', 'trailer', 'construction_vehicle', 'motorcycle', 'bicycle'
    # 'car', 'car[vehicle.moving]', 'car[vehicle.stopped]', 'car[vehicle.parked]',
    # 'pedestrian', 'pedestrian[pedestrian.moving]', 'pedestrian[pedestrian.sitting_lying_down]', 'pedestrian[pedestrian.standing]',
    # 'barrier',
    # 'traffic_cone',
    # 'truck', 'truck[vehicle.moving]', 'truck[vehicle.stopped]', 'truck[vehicle.parked]',
    # 'bus', 'bus[vehicle.moving]', 'bus[vehicle.stopped]', 'bus[vehicle.parked]',
    # 'trailer', 'trailer[vehicle.moving]', 'trailer[vehicle.stopped]', 'trailer[vehicle.parked]',
    # 'construction_vehicle', 'construction_vehicle[vehicle.moving]', 'construction_vehicle[vehicle.stopped]', 'construction_vehicle[vehicle.parked]',
    # 'motorcycle', 'motorcycle[cycle.with_rider]', 'motorcycle[cycle.without_rider]',
    # 'bicycle', 'bicycle[cycle.with_rider]', 'bicycle[cycle.without_rider]'
    'car[v0-40]',
    # 'car[v40-60]',
    # 'car[v60-80]',
    'car[v80-100]',
    'truck[v0-40]',
    # 'truck[v40-60]',
    # 'truck[v60-80]',
    'truck[v80-100]',
    'bus[v0-40]',
    # 'bus[v40-60]',
    # 'bus[v60-80]',
    'bus[v80-100]',
    'trailer[v0-40]',
    # 'trailer[v40-60]',
    # 'trailer[v60-80]',
    'trailer[v80-100]',
    'construction_vehicle[v0-40]',
    # 'construction_vehicle[v40-60]',
    # 'construction_vehicle[v60-80]',
    'construction_vehicle[v80-100]',
    'pedestrian[v0-40]',
    # 'pedestrian[v40-60]',
    # 'pedestrian[v60-80]',
    'pedestrian[v80-100]',
    'motorcycle[v0-40]',
    # 'motorcycle[v40-60]',
    # 'motorcycle[v60-80]',
    'motorcycle[v80-100]',
    'bicycle[v0-40]',
    # 'bicycle[v40-60]',
    # 'bicycle[v60-80]',
    'bicycle[v80-100]',
    'traffic_cone[v0-40]',
    # 'traffic_cone[v40-60]',
    # 'traffic_cone[v60-80]',
    'traffic_cone[v80-100]',
    'barrier[v0-40]',
    # 'barrier[v40-60]',
    # 'barrier[v60-80]',
    'barrier[v80-100]',
]

if args.dataset == 'nuscenes':
    methods = [
        "pointpillars",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_ta_learn",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_ta_ots",
        "all_pp_mhead_d1_ep20_ev2_pp_oa_ta",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_ta_ots_early_fusion",
    ]
    names = [
        "pp(official)",
        "vp_pp_oa_ta_learn",
        "vp_pp_oa_ta_ots",
        "pp_oa_ta",
        "vp_pp_oa_ta_ots_EF",
    ]

cache = {}

assert(len(methods) == len(names))

for method in methods:
    res_dir = f'{base_dir}/{method}/results'
    if not os.path.exists(res_dir):
         continue

    if args.step == -1:  # use the final checkpoint
        all_steps = []
        for d in os.listdir(res_dir):
            if os.path.exists(os.path.join(res_dir, d, "metrics_summary.json")):
                all_steps.append(int(d.split("_")[1]))
        if len(all_steps) == 0:
            continue
        step = max(all_steps)
    else:
        step = args.step
    print(method, step)

    res_file = f'{res_dir}/step_{step}/metrics_summary.json'
    if os.path.exists(res_file):
        with open(res_file, 'r') as f:
            summary = json.load(f)

        cache[method] = summary


delim = '\t'
# delim = ' & '
metric = args.metric

print('{:24}'.format(f'mAP[{args.thresh}]'), end=delim)
for cls in classes:
    print('{:5}'.format(cls[:5]), end=delim)
print('{:5}'.format('avg'))

for name, method in zip(names, methods):
    if method not in cache:
        continue

    print('{:24}'.format(name), end=delim)
    APs = []
    for cls in classes:
        n = cache[method][metric][cls]
        if args.thresh in n:
            AP = n[args.thresh]
        else:
            AP = sum(n.values())/len(n)
        APs.append(AP)
        print('{:.3f}'.format(AP), end=delim)
    mAP = sum(APs)/len(APs)
    print('{:.3f}'.format(mAP))
