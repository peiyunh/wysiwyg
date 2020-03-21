import ast
import numpy as np

# rootdir = "/data/nuscenes_models/second"
rootdir = "models/nuscenes"

models = [
    "all_pp_mhead_d2", 
    "all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_ta_ots_early_fusion", 
]

for model in models:
    path = f"{rootdir}/{model}/log.json.lst"
    print(path)
    
    step_times = []
    voxel_times = []
    prep_times = []
    with open(path) as f:
        for line in f:
            if line.startswith(r'{"runtime":'): 
                data = ast.literal_eval(line)
                step_times.append(data["runtime"]["steptime"])
                voxel_times.append(data["runtime"]["voxel_gene_time"])
                prep_times.append(data["runtime"]["prep_time"])
    
    step_times = np.array(step_times)
    voxel_times = np.array(voxel_times)
    prep_times = np.array(prep_times)

    # I = (step_times <= 1)
    I = np.arange(len(step_times))
    print(
        "total:", step_times[I].mean(),
        "voxel:", voxel_times[I].mean(),
        "prep:", prep_times[I].mean()
    )