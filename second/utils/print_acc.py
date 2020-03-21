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
    'car', 'pedestrian', 'barrier', 'traffic_cone', 'truck', 'bus', 'trailer', 'construction_vehicle', 'motorcycle', 'bicycle'
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
]

if args.dataset == 'nuscenes':
    methods = [
        "pointpillars",
        # "all_pp_mhead_vpn_swpdb_d1_ep50_ev5_vp_pp_oa_ta_learn",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_ta_learn",
        "all_pp_mhead_vpn_swpdb_d1_ep50_ev5_vp_pp_oa_ta_ots",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_ta_ots",
        "all_pp_mhead_vpn_nodbs_d1_ep20_ev2_vp_pp_ta_ots",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_naive_nomask_ta_ots",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_cull_nomask_ta_ots",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_drill_nomask_ta_ots",
        "all_pp_mhead_vpn_vpnonly_swp0_swpdb_d1_ep20_ev2_vp_oa_drill",
        "all_pp_mhead_vpn_vpnonly_swp0_swpdb_d1_ep20_ev2_vp_oa_cull",
        "all_pp_mhead_d1_ep20_ev2_pp_oa_ta",
        "all_pp_mhead_nodbs_d1_ep20_ev2_pp_ta",
        "all_pp_mhead_swp0_d1_ep20_ev2_pp_oa",
        "all_pp_mhead_vpn_d1_ep20_ev2_vp_pp_ta_ots",
        "all_pp_mhead_vpn_rpnonly_swp0_nodbs_d1_ep20_ev2_pp",
        "all_pp_mhead_vpn_swp0_nodbs_d1_ep20_ev2_vp_pp",
        "all_pp_mhead_vpn_swp0_swpdb_d1_ep20_ev2_vp_pp_oa_cull",
        "all_pp_mhead_vpn_swp0_swpdb_d1_ep20_ev2_vp_pp_oa_double_drill",
        "all_pp_mhead_vpn_swp0_swpdb_d1_ep20_ev2_vp_pp_oa_drill",
        "all_pp_mhead_vpn_swp0_swpdb_d1_ep20_ev2_vp_pp_oa_naive",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_cull_ta_ots",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_drill_ta_ots",
        # "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_ta_learn",
        # "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_ta_ots",
        "all_pp_mhead_vpn_swpdb_d1_ep50_ev5_vp_pp_oa_ots",
        "all_pp_mhead_vpn_vpnonly_nodbs_d1_ep20_ev2_vp_ta_ots",
        "all_pp_mhead_vpn_vpnonly_swp0_nodbs_d1_ep20_ev2_vp",
        "all_pp_mhead_vpn_vpnonly_swp0_swpdb_d1_ep20_ev2_vp_oa_cull_nomask",
        "all_pp_mhead_vpn_vpnonly_swp0_swpdb_d1_ep20_ev2_vp_oa_drill_nomask",
        "all_pp_mhead_vpn_vpnonly_swp0_swpdb_d1_ep20_ev2_vp_oa_naive",
        "all_pp_mhead_vpn_vpnonly_swpdb_d1_ep20_ev2_vp_ta_ots_oa_cull",
        "all_pp_mhead_vpn_vpnonly_swpdb_d1_ep20_ev2_vp_ta_ots_oa_drill",
        "all_pp_mhead_vpn_vpnonly_swpdb_d1_ep20_ev2_vp_ta_ots_oa_naive",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_ta_ots_sfvis",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_ta_ots_early_fusion",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_ta_ots_early_fusion_run_1",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_ta_ots_early_fusion_run_2",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_ta_ots_early_fusion_run_3",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_ta_ots_early_fusion_run_4",
        "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_ta_ots_early_fusion_run_5",
    ]
    names = [
        "pp(official)",
        # "vp_pp_oa_ta_learn",
        "vp_pp_oa_ta_learn",
        "vp_pp_oa_ta_ots(50ep)",
        "vp_pp_oa_ta_ots",
        "vp_pp_ta_ots",
        "vp_pp_oa_naive_nomask_ta_ots",
        "vp_pp_oa_cull_nomask_ta_ots",
        "vp_pp_oa_drill_nomask_ta_ots",
        "vp_oa_drill",
        "vp_oa_cull",
        "pp_oa_ta",
        "pp_ta",
        "pp_oa",
        "vp_pp_ta_ots",
        "pp",
        "vp_pp",
        "vp_pp_oa_cull",
        "vp_pp_oa_double_drill",
        "vp_pp_oa_drill",
        "vp_pp_oa_naive",
        "vp_pp_oa_cull_ta_ots",
        "vp_pp_oa_drill_ta_ots",
        # "vp_pp_oa_ta_learn",
        # "vp_pp_oa_ta_ots",
        "vp_pp_oa_ta_ots(50ep)",
        "vp_ta_ots",
        "vp",
        "vp_oa_cull_nomask",
        "vp_oa_drill_nomask",
        "vp_oa_naive",
        "vp_ta_ots_oa_cull",
        "vp_ta_ots_oa_drill",
        "vp_ta_ots_oa_naive",
        "vp_pp_oa_drill_ta_ots_sfvis",
        "vp_pp_oa_ta_ots_early_fusion",
        "vp_pp_oa_ta_ots_early_fusion(run1)",
        "vp_pp_oa_ta_ots_early_fusion(run2)",
        "vp_pp_oa_ta_ots_early_fusion(run3)",
        "vp_pp_oa_ta_ots_early_fusion(run4)",
        "vp_pp_oa_ta_ots_early_fusion(run5)",
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
        AP = AP * 100
        APs.append(AP)
        print('{:.1f}'.format(AP), end=delim)
    mAP = sum(APs)/len(APs)
    print('{:.1f}'.format(mAP))
