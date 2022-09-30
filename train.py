import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()

import torch
from tqdm import tqdm
import objdetect as od
import mydata, mymodels

device = 'cuda' if torch.cuda.is_available() else 'cpu'

########################## DATA ##########################

grid_size = (14, 14)
ds = mydata.DiscretizeTopView(mydata.KITTI('/data'), (512, 512))
tr = torch.utils.data.DataLoader(ds, 16, True, collate_fn=od.utils.collate_fn)

########################## MODEL ##########################

model = mymodels.Pixor().to(device)
scores_loss = torch.nn.BCEWithLogitsLoss()
bboxes_loss = torch.nn.MSELoss(reduction='none')
angles_loss = torch.nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters())

########################## TRAIN ##########################

model.train()
for epoch in range(args.epochs):
    avg_loss = 0
    for imgs, targets in tqdm(tr):
        imgs = imgs.to(device)[:, None]
        preds_scores, preds_bboxes, preds_angles = model(imgs)

        slices = od.grid.slices_center_locations(*grid_size, targets['bboxes'])
        scores = od.grid.scores(*grid_size, slices, device=device)
        bboxes = od.grid.offset_logsize_bboxes(*grid_size, slices, targets['bboxes'], device=device)
        angles = mymodels.grid_angles(*grid_size, slices, targets['angles'], device=device)

        loss_value = \
            scores_loss(preds_scores, scores) + \
            (scores * bboxes_loss(preds_bboxes, bboxes)).mean() + \
            (scores * angles_loss(preds_angles, angles)).mean()
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        avg_loss += float(loss_value) / len(tr)
    print(f'Epoch {epoch+1}/{args.epochs} - Avg loss: {avg_loss}')

torch.save(model.cpu(), args.output)
