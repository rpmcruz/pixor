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

ds = mydata.DiscretizeTopView(mydata.KITTI(args.datadir), (512, 512))
tr = torch.utils.data.DataLoader(ds, 16, True, collate_fn=od.utils.collate_fn)

########################## MODEL ##########################

model = torch.load(args.model, map_location=device)

########################## TRAIN ##########################

model.eval()
images = torch.stack([ds[i]['image'] for i in range(16)])
batch_bboxes, batch_angles = model(imgs.to(device))

for i, (image, bboxes, angles) in enumerate(zip(images, batch_bboxes, batch_angles)):
    plt.subplot(4, 4, i+1)
    mydata.plot_topview(image, bboxes, angles)
plt.show()
