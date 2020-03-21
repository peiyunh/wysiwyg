from pathlib import Path

"""
# branch: pps-kitti
# car
CUDA_VISIBLE_DEVICES=0 python script.py test_kitti --model_name="pointpillars_car_oc_xyres_16" --eval_step=296960 &
# ped
CUDA_VISIBLE_DEVICES=1 python script.py test_kitti --model_name="pointpillars_ped_cycle_oc_lb_xyres_16" --eval_step=278400 &
# cyc
CUDA_VISIBLE_DEVICES=2 python script.py test_kitti --model_name="pointpillars_ped_cycle_oc_lb_xyres_16" --eval_step=139200 &

# branch: pps-kitti-vp-oa-early-fusion
# car
CUDA_VISIBLE_DEVICES=0 python script.py test_kitti --model_name="pointpillars_car_vfn_oc_xyres_16" --eval_step=296960 &
# ped
CUDA_VISIBLE_DEVICES=1 python script.py test_kitti --model_name="pointpillars_ped_cycle_vfn_oc_lb_xyres_16" --eval_step=204160 &
# cyc
CUDA_VISIBLE_DEVICES=2 python script.py test_kitti --model_name="pointpillars_ped_cycle_vfn_oc_lb_xyres_16" --eval_step=83520 &
"""
# split = "test"
split = "val"
kitti_dir = "/data/kitti/object"
model_dir = "models/kitti"

"""
submission = {
    "name": "WYSIWYG",
    "dirs": {
        "Car": f"pointpillars_car_vfn_oc_xyres_16/{split}_results/step_296960/kitti", 
        "Pedestrian": f"pointpillars_ped_cycle_vfn_oc_lb_xyres_16/{split}_results/step_204160/kitti", 
        "Cyclist": f"pointpillars_ped_cycle_vfn_oc_lb_xyres_16/{split}_results/step_83520/kitti",
    } 
}
"""
submission = {
    "name": "WYSIWYG",
    "dirs": {
        # "Car": f"pointpillars_car_vfn_oc_xyres_16/{split}_results/step_296960/kitti", 
        "Car": f"trainval_pointpillars_car_vpn_xyres_16/{split}_results/step_296960/kitti", 
        "Pedestrian": f"trainval_pointpillars_ped_cycle_vpn_lb_xyres_16/{split}_results/step_296960/kitti", 
        "Cyclist": f"trainval_pointpillars_ped_cycle_vpn_lb_xyres_16/{split}_results/step_296960/kitti", 
    } 
}

# score_thr = 0.5
score_thr = 0.0
# score_thr = 0.3
if score_thr > 0:
    output_dir = Path(f"results/kitti/{split}/{submission['name']}_above_{score_thr}/data")
else:
    output_dir = Path(f"results/kitti/{split}/{submission['name']}/data")

output_dir.mkdir(parents=True, exist_ok=True)

with open(f"{kitti_dir}/{split}.txt") as f:
    I = [int(line.strip()) for line in f]

for i in I: 
    output_file = output_dir / ("%06d.txt" % i)
    output_handle = open(output_file, "a")
    for cls in submission["dirs"]:
        input_file = Path(model_dir) / submission["dirs"][cls] / ("%06d.txt" % i)
        input_handle = open(input_file, "r")
        for line in input_handle:
            if line.startswith(cls):
                elements = line.split()
                label, trunc, occl, alpha, x1, y1, x2, y2, h, w, l, x, y, z, r, score = elements
                if float(score) < score_thr: continue
                if not line.endswith('\n'): line += '\n'
                output_handle.write(line)
        input_handle.close()
    output_handle.close()
