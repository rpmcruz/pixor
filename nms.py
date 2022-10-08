import numpy as np
from shapely.geometry import Polygon

def polygon(loc, dim, angle):
    print('angle:', angle)
    M = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    print('pre dim:', dim)
    dim = dim[::-1]
    dim = M @ (dim/2)
    print('rot dim:', dim)
    corners = [loc + dim*np.array(s) for s in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
    return Polygon(corners)

def IoU(bbox1, bbox2):
    p1 = polygon(*bbox1)
    p2 = polygon(*bbox2)
    intersection = p1.intersection(p2).area
    union = p1.area + p2.area - intersection
    return intersection / union

def NMS(scores, bboxes, lambda_nms=0.5):
    ix = [
        not any(  # discard if all conditions met
            i != j and
            scores[j] > scores[i] and
            IoU(bboxes[i], bboxes[j]) >= lambda_nms
            for j in range(len(bboxes)))
        for i in range(len(bboxes))]
    return scores[ix], bboxes[ix]

if __name__ == '__main__':  # debug our polygon conversion function
    import matplotlib.pyplot as plt
    import mydata
    transforms = [mydata.DiscretizeBEV((800, 700, 35), ((-40, 40), (0, 70), (-2.5, 1)), 10)]
    ds = mydata.KITTI('/data', transforms)
    features, locations, dimensions, angles = ds[6]
    mydata.draw_topview(features, locations, dimensions, angles)
    for loc, dim, angle in zip(locations, dimensions, angles):
        p = polygon(loc, dim, angle)
        plt.plot(*p.exterior.xy, c='green')
    plt.show()
