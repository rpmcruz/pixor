import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('--datadir', default='/data')
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()

import torch
import torchvision
from time import time
from tqdm import tqdm
import mydata, mymodels

device = 'cuda' if torch.cuda.is_available() else 'cpu'

########################## DATA ##########################

transforms = [
    mydata.RandomYRotation(-5, 5),
    mydata.RandomXFlip(),
    mydata.DiscretizeBEV((800, 700, 35), ((-40, 40), (0, 70), (-2.5, 1)), 10),
    mydata.ToGrid((800, 700), (200, 175), 800/200),
]
ds = mydata.KITTI(args.datadir, transforms)
tr = torch.utils.data.DataLoader(ds, 8, True, num_workers=4, pin_memory=True)

########################## MODEL ##########################

model = mymodels.Pixor(200/800).to(device)
cls_loss = torchvision.ops.sigmoid_focal_loss
reg_loss = torch.nn.SmoothL1Loss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

########################## TRAIN ##########################

model.train()
for epoch in range(args.epochs):
    tic = time()
    avg_loss = 0
    for features, grid_scores, grid_bboxes in tr: #tqdm(tr):
        features = features.to(device)
        grid_scores = grid_scores.to(device)
        grid_bboxes = grid_bboxes.to(device)
        preds_scores, preds_bboxes = model(features)
        loss_value = \
            cls_loss(preds_scores, grid_scores).mean() + \
            (grid_scores * reg_loss(preds_bboxes, grid_bboxes)).mean()
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        avg_loss += float(loss_value) / len(tr)
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Avg loss: {avg_loss}')

torch.save(model.cpu(), args.output)
