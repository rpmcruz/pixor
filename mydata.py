import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def truncate_points(points, mins, maxs):
    ix = np.logical_and(
        np.all(points >= mins, 0),
        np.all(points <= maxs, 0))
    return points[:, ix]

def project_velodyne_to_camera(points, R0, Tvelo_cam):
    return R0 @ Tvelo_cam @ points

def plot_topview(image, bboxes, angles):
    h, w = image.shape
    plt.imshow(image, vmin=0, vmax=1, cmap='gray_r', origin='lower')
    for bbox, angle in zip(bboxes, angles):
        plt.gca().add_patch(matplotlib.patches.Rectangle(
            (bbox[0]*w, bbox[1]*h), (bbox[2]-bbox[0])*w, (bbox[3]-bbox[1])*h,
            angle=(3*np.pi/2-angle)*180/np.pi, rotation_point='center',
            linewidth=1, edgecolor='r', facecolor='none'))

class KITTI(torch.utils.data.Dataset):
    # Only loads cars, loads velodyne and labels in camera coordinates

    mins = np.array([[-44.1], [-2.2], [-0.2]])
    maxs = np.array([[39.9], [4.1], [86.2]])

    def __init__(self, root):
        self.root = os.path.join(root, 'kitti', 'object', 'training')
        self.fnames = [f[:-4] for f in sorted(os.listdir(os.path.join(self.root, 'label_2')))]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        fname = self.fnames[i]
        # labels variables
        # dimensions  3D object dimensions: height, width, length (in meters)
        # location    3D object location x,y,z in camera coordinates (in meters)
        # rotation_y  Rotation ry around Y-axis in camera coordinates [-pi..pi]
        labels_fname = os.path.join(self.root, 'label_2', f'{fname}.txt')
        labels = pd.read_csv(labels_fname, sep=' ', header=None)
        ix = labels.iloc[:, 0] == 'Car'  # filter only cars
        bboxes = labels.loc[ix, 4:7].to_numpy()
        dimensions = labels.loc[ix, 8:10].to_numpy().T
        locations = labels.loc[ix, 11:13].to_numpy().T
        rotations_y = labels.loc[ix, 14].to_numpy()
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
        points = points.T  # (N,4) => (4,N)
        points = project_velodyne_to_camera(points, R0, Tvelo_cam)
        points = points[:3]  # exclude luminance
        points = truncate_points(points, self.mins, self.maxs)
        return points, locations, dimensions, rotations_y

class DiscretizeTopView(torch.utils.data.Dataset):
    def __init__(self, ds, img_shape, dict_transform=None):
        self.ds = ds
        self.img_shape = img_shape
        self.dict_transform = dict_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        points, locations, dimensions, rotations_y = self.ds[i]
        # locations/dimensions => bboxes x1y1x2y2
        bboxes = np.concatenate((
            locations[[0, 2]] - dimensions[[0, 2]]/2,
            locations[[0, 2]] + dimensions[[0, 2]]/2), 0)
        # normalize all
        points = (points-self.ds.mins) / (self.ds.maxs-self.ds.mins)
        bboxes = (bboxes-self.ds.mins[[0, 2, 0, 2]]) / (self.ds.maxs[[0, 2, 0, 2]]-self.ds.mins[[0, 2, 0, 2]])
        # points discretization
        image = np.zeros(self.img_shape, np.float32)
        image[(self.img_shape[1]*points[2]).astype(int),
             (self.img_shape[0]*points[0]).astype(int)] = 1
        d = {'image': image, 'bboxes': bboxes.T, 'angles': rotations_y}
        if self.dict_transform:
            d = self.dict_transform(**d)
        d['image'] = torch.tensor(d['image'])
        return d

# DEBUG

def debug_plot_raw(ds):
    for i, (points, locations, dimensions, rotations_y) in enumerate(ds):
        if i >= 20: break
        points, locations, dimensions, rotations_y = ds[i]
        plt.clf()
        plt.scatter(points[0], points[2], s=1, c='k')
        plt.scatter(locations[0], locations[2], s=10, c='g')
        for loc, size, roty in zip(locations.T, dimensions.T, rotations_y):
            rect = matplotlib.patches.Rectangle((loc[0]-size[0]/2, loc[2]-size[2]/2), size[0], size[2], angle=(np.pi/2-roty)*180/np.pi, rotation_point='center', linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
        plt.title(f'#objects: {locations.shape[1]}')
        plt.savefig(f'raw-{i:02d}.png')

def debug_plot_topview(ds):
    for i, d in enumerate(ds):
        if i >= 20: break
        plt.clf()
        plot_topview(d['image'], d['bboxes'], d['angles'])
        plt.title(f'#objects: {d["bboxes"].shape[0]}')
        plt.savefig(f'topview-{i:02d}.png')

if __name__ == '__main__':
    import matplotlib
    plt.rcParams['figure.figsize'] = (20, 20)
    matplotlib.use('Agg')
    ds = KITTI('/data')
    debug_plot_raw(ds)
    ds = DiscretizeTopView(ds, (512, 512))
    debug_plot_topview(ds)
