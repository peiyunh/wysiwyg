import os
import ast
import json
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

root_dir = "models/kitti/"

ignored_names = [
    "backup",
    "all8_fhd(run0)",
    "all8_fhd_gt_fgm",
    # "all_fhd",
    # "all8_fhd",
    # "car_fhd",
    # "car_fhd_onestage",
    # "people_fhd",
    "pointpillars_pp_pretrain",
    # "pointpillars_car_xyres_16",
    # "pointpillars_ped_cycle_xyres_16",
    # "pointpillars_car_vfn_xyres_16",
    # "pointpillars_ped_cycle_vfn_xyres_16",
]

ignored_labels = [
    # "oc",
]

metrics = {
    "Car": "3d@0.70",
    "Pedestrian": "3d@0.50",
    "Cyclist": "3d@0.50",
}

numbers = {
    "Car": {},
    "Pedestrian": {},
    "Cyclist": {},
}

difficulties = {
    0: "Easy",
    1: "Medium",
    2: "Hard"
}

for name in os.listdir(root_dir):
    if name in ignored_names:
        continue

    if any(label in name for label in ignored_labels):
        continue

    last_step = -1
    stats_list = []
    path = os.path.join(root_dir, name, "log.json.lst")
    with open(path) as f:
        for line in f:
            data = ast.literal_eval(line)
            if "runtime" in data:
                last_step = data["runtime"]["step"]
            elif "eval.kitti" in data:
                mAPs_dict = data["eval.kitti"]["official"]
                stats_list.append((last_step, mAPs_dict))

    for last_step, mAPs_dict in stats_list:
        for cls in metrics:
            metric = metrics[cls]
            if cls in mAPs_dict:
                mAPs = (mAPs_dict[cls][metric])
                if name not in numbers[cls]:
                    numbers[cls][name] = []
                numbers[cls][name].append([last_step]+mAPs)

plt.figure(figsize=(32,18))
i = 0
for cls in numbers:
    for name in numbers[cls]:
        data = np.array(numbers[cls][name])
        steps, mAPs = data[:,0], data[:, 1:]
        # pick the best checkpoint based on the performance on the medium subset
        k = np.argmax(mAPs[:, 1])
        for j in difficulties:
            plt.subplot(len(numbers), 3, i*3+j+1)
            plt.plot(steps, mAPs[:,j], '-.')
            plt.plot(steps, np.full_like(steps, mAPs[k,j]), label=name)
    for j in difficulties:
        plt.subplot(len(numbers), 3, i*3+j+1)
        plt.title(cls+" - "+difficulties[j])
        plt.legend()
    i += 1
plt.savefig(f"results/kitti.png", dpi=250, bbox_inches="tight")
