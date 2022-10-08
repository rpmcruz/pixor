import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class KITTI(torch.utils.data.Dataset):
    # Only loads cars, loads velodyne and labels in camera coordinates

    def __init__(self, root, transforms=[]):
        self.root = os.path.join(root, 'kitti', 'object', 'training')
        self.fnames = [f[:-4] for f in sorted(os.listdir(os.path.join(self.root, 'label_2')))]
        self.transforms = transforms

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        fname = self.fnames[i]
        # labels file
        # dimensions  3D object dimensions: height, width, length (in meters)
        # location    3D object location x,y,z in camera coordinates (in meters)
        # rotation_y  Rotation ry around Y-axis in camera coordinates [-pi..pi]
        labels_fname = os.path.join(self.root, 'label_2', f'{fname}.txt')
        labels = pd.read_csv(labels_fname, sep=' ', header=None)
        ix = labels.iloc[:, 0] == 'Car'  # filter only cars
        dimensions = labels.loc[ix, 8:10].to_numpy()
        locations = labels.loc[ix, 11:13].to_numpy()
        angles = labels.loc[ix, 14].to_numpy()
        # camera calibration
        # we will need this to convert velodyne points -> camera coordinates
        calib_fname = os.path.join(self.root, 'calib', f'{fname}.txt')
        with open(calib_fname) as f:
            calib = [np.array([float(v) for v in l.split()[1:]]) for l in f.readlines()]
        P2 = calib[2].reshape((3, 4))
        R0 = np.c_[np.r_[calib[4].reshape((3, 3)), np.zeros((1, 3))], np.zeros((4, 1))]
        R0[-1, -1] = 1
        Tvelo_cam = np.r_[calib[5].reshape((3, 4)), np.zeros((1, 4))]
        Tvelo_cam[-1, -1] = 1
        # velodyne points (convert them to camera coordinates)
        points_fname = os.path.join(self.root, 'velodyne', f'{fname}.bin')
        points = np.fromfile(points_fname, dtype=np.float32).reshape(-1, 4)
        points = (R0 @ Tvelo_cam @ points.T).T  # project velodyne -> camera coordinates
        radiance = points[:, 3]
        points = points[:, :3]
        # in the data, the axis order seems to be (X,Z,Y). let's fix that.
        points = points[:, [0, 2, 1]]
        locations = locations[:, [0, 2, 1]]
        dimensions = dimensions[:, [2, 0, 1]]  # length=X, width=Y, height=Z
        angles = -angles  # it seems the angle goes clockwise (fix it)
        # transformations
        output = (points, radiance, locations, dimensions, angles)
        for t in self.transforms:
            output = t(*output)
        return output

def RandomYRotation(rot_min_deg, rot_max_deg):
    def f(points, radiance, locations, dimensions, angles):
        rot = np.random.rand()*(rot_max_deg-rot_min_deg) + rot_min_deg
        rot = rot*np.pi/180  # to radians
        cos_angle = np.cos(rot)
        sin_angle = np.sin(rot)
        R = np.array(((cos_angle, -sin_angle, 0),
            (sin_angle, cos_angle, 0), (0, 0, 1)), np.float32)
        points = np.dot(R, points.T).T
        locations = np.dot(R, locations.T).T
        angles -= rot
        return points, radiance, locations, dimensions, angles
    return f

def RandomXFlip():
    def f(points, radiance, locations, dimensions, angles):
        if np.random.rand() < 0.5:
            points[:, 0] *= -1
            locations[:, 0] *= -1
            angles = np.pi - angles
        return points, radiance, locations, dimensions, angles
    return f

def DiscretizeBEV(out_shape, limits, ratio_meters2pixels):
    dL, dW, dH = out_shape
    xlimits, ylimits, zlimits = limits
    def f(points, radiance, locations, dimensions, angles):
        # truncate points outside range
        # notice that we do not truncate the Z axis because we want to use those
        # to "add two additional channels to occupancy feature maps to cover
        # out-of-range points."
        ix = (points[:, 0] >= xlimits[0]) & (points[:, 0] < xlimits[1]) & \
            (points[:, 1] >= ylimits[0]) & (points[:, 1] < ylimits[1])
        points = points[ix]
        # map poins to indices: (1) normalize them, (2) convert to indices
        xx = (points[:, 0]-xlimits[0]) / (xlimits[1]-xlimits[0])
        xx = (xx * dL).astype(int)
        yy = (points[:, 1]-ylimits[0]) / (ylimits[1]-ylimits[0])
        yy = (yy * dW).astype(int)
        zz = (points[:, 2]-zlimits[0]) / (zlimits[1]-zlimits[0])
        zz = (zz * dH).astype(int)
        # the following line is to agglomerate out-of-range points at indices 0
        # and -1 (last).
        zz = 1 + np.minimum(dH, np.maximum(-1, zz))
        # features map: the PIXOR paper is unclear how "occupancy" and
        # "intensity/radiance" is computed, but...
        # for intensity: they cite a paper that says "The intensity feature is
        # the reflectance value of the point which has the maximum height in each
        # cell."
        # for occupancy: they have a follow-up paper (HDNET) that says "We then
        # compute *binary* occupancy maps" It seems like they just put 0/1 where
        # there is a lidar. A unofficial github implementation also did just that.
        # It's a little confusing because other literature uses ray-tracing and
        # another algorithm to crease dense occupancy maps.
        # Also notice that the shape of our features is dH,dW,dL instead of
        # dL,dW,dH. This is because for matplotib (x=cols, y=rows) and for
        # pytorch channels come first.
        height_map = -np.inf * np.ones((dW, dL), np.float32)
        features = np.zeros((dH+3, dW, dL), np.float32)
        for x, y, z, r in zip(xx, yy, zz, radiance):
            features[z, y, x] += 1  # occupancy
            if height_map[y, x] < z:
                features[-1, y, x] = r  # radiance/intensity
                height_map[y, x] = z
        # convert locations/dimensions: meters => pixels
        # in the case of locations, translation necessary (e.g. -40,40 => 0,80 => 0,800)
        locations[:, 0] = (locations[:, 0]-xlimits[0]) * ratio_meters2pixels
        locations[:, 1] = (locations[:, 1]-ylimits[0]) * ratio_meters2pixels
        locations[:, 2] = (locations[:, 2]-zlimits[0]) * ratio_meters2pixels
        dimensions *= ratio_meters2pixels
        # ignore z-axis from the labels
        locations = locations[:, :2]
        dimensions = dimensions[:, :2]
        # filter labels outside view
        ix = np.all(np.logical_and(locations >= 0, locations < np.array([[dL, dW]])), 1)
        locations = locations[ix]
        dimensions = dimensions[ix]
        angles = angles[ix]
        return features, locations, dimensions, angles
    return f

def ToGrid(feature_shape, grid_shape, ratio_feature2grid):
    dL, dW = grid_shape
    cell_L = feature_shape[0]/dL
    cell_W = feature_shape[1]/dW
    def f(features, locations, dimensions, angles):
        grid_scores = np.zeros((1, dW, dL), np.float32)
        grid_bboxes = np.zeros((6, dW, dL), np.float32)
        if len(locations) == 0:
            return features, grid_scores, grid_bboxes
        yc = (locations[:, 1]*ratio_feature2grid).astype(int)
        xc = (locations[:, 0]*ratio_feature2grid).astype(int)
        # a minor difference is that our dx/dy offset is relative to the cell
        # top/left corner, not the center.
        # the paper also says "[t]he learning target [bboxes] [...] is normalized
        # before-hand over the training set to have zero mean and unit variance."
        # the values are already small, so I don't normalize.
        grid_scores[:, yc, xc] = 1
        grid_bboxes[0, yc, xc] = np.cos(angles)
        grid_bboxes[1, yc, xc] = np.sin(angles)
        grid_bboxes[2, yc, xc] = (locations[:, 0] % cell_L) / cell_L
        grid_bboxes[3, yc, xc] = (locations[:, 1] % cell_W) / cell_W
        grid_bboxes[4, yc, xc] = np.log(dimensions[:, 0])
        grid_bboxes[5, yc, xc] = np.log(dimensions[:, 1])
        return features, grid_scores, grid_bboxes
    return f

def ToGrid_Debug(feature_shape, grid_shape, ratio_feature2grid):
    g = ToGrid(feature_shape, grid_shape, ratio_feature2grid)
    def f(features, locations, dimensions, angles):
        return locations, *g(features, locations, dimensions, angles)
    return f

def inv_scores(scores, threshold):
    hasobjs = scores >= threshold
    return scores[hasobjs]

def inv_bboxes(scores, threshold, bboxes, ratio_grid2feature):
    _, h, w = scores.shape
    xx = np.arange(0, w, dtype=np.float32)[None, :]
    yy = np.arange(0, h, dtype=np.float32)[:, None]
    angles = np.arctan2(bboxes[1], bboxes[0])
    xc = (xx + bboxes[2]) * ratio_grid2feature
    yc = (yy + bboxes[3]) * ratio_grid2feature
    locations = np.stack((xc, yc), -1)
    dimensions = np.moveaxis(np.exp(bboxes[4:6]), 0, 2)
    # filter those with objects
    hasobjs = (scores >= threshold)[0]
    angles = angles[hasobjs]
    locations = locations[hasobjs]
    dimensions = dimensions[hasobjs]
    return locations, dimensions, angles

def InvGrid_Debug(ratio_grid2feature):
    def f(locations, features, grid_scores, grid_bboxes):
        return features, *inv_bboxes(grid_scores, 0.5, grid_bboxes, ratio_grid2feature)
    return f

# DEBUG

def draw_raw(points, radiance, locations, dimensions, angles):
    plt.scatter(points[:, 0], points[:, 1], s=1, c='k')
    plt.scatter(locations[:, 0], locations[:, 1], s=10, c='g')
    plt.xlabel('X')
    plt.ylabel('Y')
    for loc, dim, angle in zip(locations, dimensions, angles):
        angle_deg = angle*180/np.pi
        bx, by = loc[0]-dim[0]/2, loc[1]-dim[1]/2
        plt.gca().add_patch(Rectangle((bx, by), dim[0], dim[1],
            angle=angle_deg, rotation_point='center', linewidth=1,
            edgecolor='r', facecolor='none'))
        plt.text(loc[0], loc[1], str(int(angle_deg)), c='b')

def draw_topview(features, locations, dimensions, angles, bc='r'):
    if features is not None:
        image = np.any(features[:35+2] >= 0.5, 0)
        plt.imshow(image, cmap='gray_r', origin='lower', vmin=0, vmax=1)
        plt.xlabel('X')
        plt.ylabel('Y')
    for loc, dim, angle in zip(locations, dimensions, angles):
        bx, by = loc[0]-dim[0]/2, loc[1]-dim[1]/2
        angle_deg = angle*180/np.pi
        plt.gca().add_patch(Rectangle((bx, by), dim[0], dim[1],
            angle=angle_deg, rotation_point='center', linewidth=1,
            edgecolor=bc, facecolor='none'))

def draw_grid(locations, features, grid_scores, grid_bboxes):
    image = np.any(features[:35+2], 0)
    plt.imshow(image, cmap='gray_r', origin='lower', vmin=0, vmax=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    ih, iw = image.shape
    _, gh, gw = grid_scores.shape
    plt.vlines(np.linspace(0, iw, gw+1), 0, ih, color='gray', lw=1, alpha=0.25)
    plt.hlines(np.linspace(0, ih, gh+1), 0, iw, color='gray', lw=1, alpha=0.25)
    plt.scatter(locations[:, 0], locations[:, 1], s=8, c='g')
    for i in range(gw):
        for j in range(gh):
            if grid_scores[0, j, i] >= 0.5:
                _, _, ox, oy, _, _ = grid_bboxes[:, j, i]
                plt.text(i*iw/gw, j*ih/gh, f'{ox:.2f},{oy:.2f}', c='b')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('steps', type=int)
    args = parser.parse_args()
    transforms = [
        RandomYRotation(-5, 5),
        RandomXFlip(),
        DiscretizeBEV((800, 700, 35), ((-40, 40), (0, 70), (-2.5, 1)), 10),
        ToGrid_Debug((800, 700), (200, 175), 200/800),
        InvGrid_Debug(800/200),
    ]
    draw = [draw_raw, draw_raw, draw_raw, draw_topview, draw_grid, draw_topview]
    assert args.steps < len(transforms), f'steps = [0,{len(transforms)-1}]'
    transforms = transforms[:args.steps]
    draw = draw[args.steps]
    ds = KITTI('/data', transforms)
    for i, d in enumerate(ds):
        if i >= 8: break
        plt.subplot(2, 4, i+1)
        draw(*d)
    plt.show()
