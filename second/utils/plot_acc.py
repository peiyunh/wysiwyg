import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='nuscenes')
parser.add_argument('--step', type=int, default='-1')
parser.add_argument('--metric', type=str, default='label_aps')
parser.add_argument('--detailed', action='store_true')
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
        # 'all_pp_mhead_nodbs_cont_d8_ep50_ev5',
        # 'all_pp_mhead_nodbs_swp0_d8_ep50_ev5',
        # 'all_pp_mhead_nodbs_swp2_d8_ep50_ev5',
        # 'all_pp_mhead_nodbs_swp4_d8_ep50_ev5',
        # 'all_pp_mhead_nodbs_swp6_d8_ep50_ev5',
        # 'all_pp_mhead_nodbs_swp8_d8_ep50_ev5',
        # 'all_pp_mhead_nodbs_d8_ep50_ev5',
        # 'all_pp_mhead_nodbs_swp0_nots_d8_ep50_ev5',
        # 'all_pp_mhead_nodbs_swp2_nots_d8_ep50_ev5',
        # 'all_pp_mhead_nodbs_swp4_nots_d8_ep50_ev5',
        # 'all_pp_mhead_nodbs_swp6_nots_d8_ep50_ev5',
        # 'all_pp_mhead_nodbs_swp8_nots_d8_ep50_ev5',
        # 'all_pp_mhead_nodbs_nots_d8_ep50_ev5',
        'all_pp_mhead_d8_ep50_ev5',
        'all_pp_mhead_d2',
        'all_pp_mhead_swpdb_d8_ep50_ev5.bak',
        'all_pp_mhead_swpdb_d8_ep50_ev5',
        'all_pp_mhead_opn_d8_ep50_ev5',
        'all_pp_mhead_opn_d2_ep50_ev5',
        'all_pp_mhead_vpn_swpdb_d8_ep50_ev5.bak',
        'all_pp_mhead_vpn_swpdb_d8_ep50_ev5',
        'all_pp_mhead_vpn_swpdb_d2_ep50_ev5',
        'all_pp_mhead_vpn_swpdb_raystop_d8_ep50_ev5',
        'all_pp_mhead_vpn_swpdb_cthruwall_d8_ep50_ev5',
        'all_pp_mhead_vpn_swpdb_learning_d8_ep50_ev5',
        'all_pp_mhead_vpn_swpdb_learning_d2_ep50_ev5',
        'all_pp_mhead_vpn_swpdb_learning_d1_ep50_ev5',
        # 'all_pp_mhead_swpdb_d2_ep50_ev5.bak',
        # 'all_pp_mhead_d2',
        # 'all_pp_mhead_occ_nodbs_d8_ep50_ev5',
        # 'all_pp_mhead_vis_nodbs_d8_ep50_ev5',
        # 'all_pp_mhead_opn_nodbs_d8_ep50_ev5',
        # 'all_pp_mhead_aug_nodbs_d8_ep50_ev5',
        # 'all_pp_mhead_vfn_nodbs_d8_ep50_ev5',
        # 'all_pp_mhead_vfn_vpnonly_nodbs_d8_ep50_ev5',
        # 'all_pp_mhead_nodbs_swp0_d8_ep50_ev5',
        # 'all_pp_mhead_vfn_vpnonly_nodbs_swp0_d8_ep50_ev5',
        # 'all_pp_mhead_moredbs_d8_ep50_ev5',
        # 'all_pp_mhead_moredbs_swpdb_d8_ep50_ev5.bak',
        # 'all_pp_mhead_moredbs_swpdb_d8_ep50_ev5',
        # 'all_pp_mhead_vpn_moredbs_swpdb_d8_ep50_ev5.bak',
        # 'all_pp_mhead_vpn_moredbs_swpdb_d8_ep50_ev5',
        # 'all_pp_mhead_vpn_moredbs_swpdb_d2_ep50_ev5',
        # 'all_pp_mhead_vpn_moredbs_swpdb_raystop_d8_ep50_ev5',
        # 'all_pp_mhead_moredbs_swpdb_d2_ep50_ev5.bak',
        # 'all_pp_mhead_moredbs_d2_ep50_ev5',
        # 'all_fhd', 'all_pp_lowa', 'all_pp_mida', 'all_pp_largea',
        # 'all_megvii_d8_ep20_ev2', 'all_megvii_d8_ep50_ev5', 'all_megvii_d2_ep20_ev2'
        # 'megvii',
        # 'pointpillars',
        # 'mapillary',
    ]
    names = [
        # 'pp+10-contT',
        # 'pp+0',
        # 'pp+2',
        # 'pp+4',
        # 'pp+6',
        # 'pp+8',
        # 'pp+10',
        # 'pp+0-NoT',
        # 'pp+2-NoT',
        # 'pp+4-NoT',
        # 'pp+6-NoT',
        # 'pp+8-NoT',
        # 'pp+10-NoT',
        'pp w/ aug',
        'pp w/ aug d2',
        'pp w/ swpaug old',
        'pp w/ swpaug',
        'pp w/ opn aug',
        'pp w/ opn aug d2',
        'pp w/ vpn swpaug old',
        'pp w/ vpn swpaug',
        'pp w/ vpn swpaug d2',
        'pp w/ vpn swpaug raystop',
        'pp w/ vpn swpaug cthruwall',
        'pp w/ vpn swpaug learning',
        'pp w/ vpn swpaug learning d2',
        'pp w/ vpn swpaug learning d1',
        # 'pp w/ swpaug d2 old',
        # 'pp w/ aug d2',
        # 'pp + early',
        # 'pp + late',
        # 'pp + late-opn',
        # 'pp + occ-aug',
        # 'pp + late-vfn',
        # 'pp + vfn-vonly',
        # 'pp w/ more aug',
        # 'pp w/ more swpaug old',
        # 'pp w/ more swpaug',
        # 'pp w/ vpn more swpaug old',
        # 'pp w/ vpn more swpaug',
        # 'pp w/ vpn more swpaug d2',
        # 'pp w/ vpn more swpaug raystop',
        # 'pp w/ more swpaug d2 old',
        # 'pp (1f)',
        # 'pp + vfn-vonly (1f)',
        # 'megvii',
        # 'pointpillars',
        # 'mapillary',
    ]
elif args.dataset == 'kitti':
    methods = [
        'all8_fhd', 'all8_fhd_gt_fgm'
    ]

cache = {}
# for method in os.listdir(base_dir):
for method in methods:
    res_dir = f'{base_dir}/{method}/results'

    if args.step == -1:  # use the final checkpoint
        all_steps = [
            int(d.split('_')[1]) for d in os.listdir(res_dir)
        ]
        step = max(all_steps)
    else:
        step = args.step
    print(res_dir, step)

    res_file = f'{res_dir}/step_{step}/metrics_summary.json'
    if os.path.exists(res_file):
        with open(res_file, 'r') as f:
            summary = json.load(f)

        cache[method] = summary


metric = args.metric
if args.detailed:
    for cls in classes:
        print('{:16}\t{:5}\t{:5}\t{:5}\t{:5}\t{:5}'.format(
            cls, 0.5, 1.0, 2.0, 4.0, avg
        ))
        # for method in cache:
        for method in methods:
            n = cache[method][metric][cls]
            if metric == 'label_aps':
                print('{:16}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
                    method, n['0.5'], n['1.0'], n['2.0'], n['4.0'], sum(n.values())/len(n)
                ))

delim = ''

print('{:16}\t'.format('method'), end=delim)
for cls in classes:
    print('{:5}\t'.format(cls[:5]), end=delim)
print('{:5}'.format('avg'))

for name, method in zip(names, methods):
    print('{:16}\t'.format(name), end=delim)
    APs = []
    for cls in classes:
        n = cache[method][metric][cls]
        AP = sum(n.values())/len(n)
        APs.append(AP)
        print('{:.3f}\t'.format(AP), end=delim)
    mAP = sum(APs)/len(APs)
    print('{:.3f}'.format(mAP))
