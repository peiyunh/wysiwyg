import os
import ast
import json
import numpy as np

root_dir = "models/kitti/"

ignored_names = [
    "backup",
    "all8_fhd(run0)",
    "all8_fhd_gt_fgm"
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

step_set = [
    296950,
]

for name in os.listdir(root_dir):
    if name in ignored_names:
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
        if len(step_set) > 0 and last_step not in step_set:
            continue
        for cls in metrics:
            metric = metrics[cls]
            if cls in mAPs_dict:
                mAPs = (mAPs_dict[cls][metric])
                if name not in numbers[cls]:
                    numbers[cls][name] = []
                numbers[cls][name].append([last_step]+mAPs)


# model selection across checkpoints
best_steps = {}
best_numbers = {}
for cls in numbers:
    if cls not in best_numbers: best_numbers[cls] = {}
    if cls not in best_steps: best_steps[cls] = {}

    for name in numbers[cls]:
        if name not in best_numbers[cls]: best_numbers[cls][name] = []

        data = np.array(numbers[cls][name])
        steps, mAPs = data[:,0], data[:, 1:]
        # pick the best checkpoint based on the performance on the medium subset
        k = np.argmax(mAPs[:, 1])
        best_numbers[cls][name] = mAPs[k, :]
        best_steps[cls][name] = int(steps[k])


# diff_order = [1, 0, 2]
diff_order = [0, 1, 2]
for cls in best_numbers:
    print(f"{cls:>48}", end='\t')
    print(f"{'step':>8}", end='\t')
    for i in diff_order:
        print(f"{difficulties[i]:<4}", end='\t')
    print(f"{'Avg':<4}")

    ordered_names = sorted(best_numbers[cls].keys(), key=lambda x: -best_numbers[cls][x][1])
    for name in ordered_names:
        print(f"{name:>48}", end='\t')
        print(f"{best_steps[cls][name]:>8}", end='\t')
        for i in diff_order:
            print(f"{best_numbers[cls][name][i]:.2f}", end='\t')
        print(f"{np.mean(best_numbers[cls][name]):.2f}")
    print()
