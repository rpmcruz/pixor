import torch
import torchvision
import objdetect as od

# objdetect does not have methods to handle angles. these utility functions are
# to handle angles grids.

def grid_angles(h, w, batch_slices, batch_angles, device=None):
    n = len(batch_slices)
    grid = torch.zeros((n, 2, h, w), dtype=torch.float32, device=device)
    for i, (slices, angles) in enumerate(zip(batch_slices, batch_angles)):
        for (yy, xx), angle in zip(slices, angles):
            grid[i, 0, yy, xx] = torch.cos(angle)
            grid[i, 1, yy, xx] = torch.sin(angle)
    return grid

def inv_grid_angles(hasobjs, angles):
    angles = torch.atan2(angles[:, [0]], angles[:, [1]])
    return [aa[h[0]] for h, aa in zip(hasobjs, angles)]

# our hero
# https://arxiv.org/abs/1902.06326

class Pixor(torch.nn.Module):
    def __init__(self):
        # FIXME: this architecture is not exactly like PIXOR (it's easy to change though; consider that your homework ;-))
        super().__init__()
        resnet = torchvision.models.resnet50()
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.header = torch.nn.Sequential(torch.nn.Conv2d(2048, 96, 3), torch.nn.ReLU())
        self.scores = torch.nn.Conv2d(96, 1, 1)
        self.bboxes = torch.nn.Conv2d(96, 4, 1)
        self.angles = torch.nn.Conv2d(96, 2, 1)

    def forward(self, x, threshold=0.5):
        x = x.repeat(1, 3, 1, 1)  # FIXME: this hack is due to the backbone I used is for for 3-channel images
        x = self.backbone(x)
        x = self.header(x)
        scores = self.scores(x)
        bboxes = self.bboxes(x)
        angles = self.angles(x)
        if not self.training:
            # when in evaluation mode, convert the output grid into a list of bboxes/classes
            scores = torch.sigmoid(scores)
            hasobjs = scores >= threshold
            scores = od.grid.inv_scores(hasobjs, scores)
            bboxes = od.grid.inv_offset_logsize_bboxes(hasobjs, bboxes)
            angles = inv_grid_angles(hasobjs, angles)
            #bboxes = od.post.NMS(scores, bboxes)
            return bboxes, angles
        return scores, bboxes, angles
