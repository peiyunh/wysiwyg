import numpy as np
from pathlib import Path
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.patches as patches

split = "test"
kitti_dir = Path("/data/kitti/object")
image_dir = kitti_dir / ("testing" if split=="test" else "training") / "image_2"

model_names = [
    # "PointPillars", 
    "WYSIWYG", 
]

res_dirs = [
    Path(f"results/kitti/{split}/{model_name}/data") for model_name in model_names
]
if split == "val":
    model_names = ["GT"] + model_names
    res_dirs = [Path("/data/kitti/object/training/label_2")] + res_dirs

    
with open(f"{kitti_dir}/{split}.txt") as f:
    I = [int(line.strip()) for line in f]

CLASSES = ["Car", "Pedestrian", "Cyclist", "Van", "Truck", "Misc", "Tram", "Person_sitting"]
colors = {
    cls : (0.7 * np.random.rand(3) + 0.3).tolist() for cls in CLASSES
}

def sigmoid(x):
    return 5 / (1 + np.exp(-x))

# np.random.shuffle(I)
for i in I: 
    plt.clf()

    image_file = image_dir / ("%06d.png" % i)
    image = imread(image_file)
    for m, res_dir in enumerate(res_dirs):
        plt.subplot(len(res_dirs), 1, m+1)
        plt.title(model_names[m])
    
        plt.imshow(image)
        res_file = res_dir / ("%06d.txt" % i)
        with open(res_file) as f: 
            for line in f:
                elements = line.split()
                if len(elements) == 15:
                    label, trunc, occl, alpha, x1, y1, x2, y2, h, w, l, x, y, z, r = elements
                    score = 1
                else:
                    label, trunc, occl, alpha, x1, y1, x2, y2, h, w, l, x, y, z, r, score = elements
                
                x1, x2, y1, y2, score = float(x1), float(x2), float(y1), float(y2), float(score)
                if label=="DontCare" or score < 0.5: continue

                print(model_names[m], ":", line.strip())
                rect = patches.Rectangle((x1, y1), x2-x1+1, y2-y1+1, linewidth=sigmoid(score), edgecolor=colors[label], facecolor="none")
                ax = plt.gca()
                ax.add_patch(rect)
    plt.show()