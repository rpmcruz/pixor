import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--datadir', default='/data')
parser.add_argument('--threshold', default=0.5, type=float)
args = parser.parse_args()

import torch
import matplotlib.pyplot as plt
import objdetect as od
import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

########################## DATA ##########################

transforms = [
    data.DiscretizeBEV((800, 700, 35), ((-40, 40), (0, 70), (-2.5, 1)), 10),
]
to_grid = data.ToGrid((800, 700), (200, 175), 200/800)
ds = data.KITTI(args.datadir, transforms)

########################## MODEL ##########################

model = torch.load(args.model, map_location=device)

########################## TRAIN ##########################

model.eval()
batch_features = torch.stack([torch.tensor(to_grid(*ds[i])[0]) for i in range(16)]).to(device)
with torch.no_grad():
    list_scores, list_bboxes = model(batch_features, args.threshold)

for i, (pred_locs, pred_dims, pred_angles) in enumerate(list_bboxes):
    plt.subplot(4, 4, i+1)
    features, gt_locs, gt_dims, gt_angles = ds[i]
    data.draw_topview(features, gt_locs, gt_dims, gt_angles)
    data.draw_topview(None, pred_locs, pred_dims, pred_angles, 'g')
    plt.title(f'{i} predicted: {len(pred_locs)}, truth: {len(gt_locs)}')
plt.show()
