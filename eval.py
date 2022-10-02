import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--datadir', default='/data')
args = parser.parse_args()

import torch
import matplotlib.pyplot as plt
import objdetect as od
import mydata

device = 'cuda' if torch.cuda.is_available() else 'cpu'

########################## DATA ##########################

transforms = [
    mydata.DiscretizeBEV((800, 700, 35), ((-40, 40), (0, 70), (-2.5, 1)), 10),
    mydata.ToGrid((800, 700), (200, 175), 200/800),
]
ds = mydata.KITTI(args.datadir, transforms)

########################## MODEL ##########################

model = torch.load(args.model, map_location=device)

########################## TRAIN ##########################

model.eval()
batch_features = torch.stack([ds[i][0] for i in range(16)]).to(device)
with torch.no_grad():
    list_scores, list_bboxes = model(features)

for i, (features, (locations, dimensions, angles)) in enumerate(zip(
        batch_features, list_bboxes)):
    plt.subplot(4, 4, i+1)
    mydata.draw_topview(features, locations, dimensions, angles)
plt.show()
