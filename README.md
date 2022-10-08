# My PIXOR

This is an unofficial PIXOR implementation. [PIXOR](https://arxiv.org/abs/1902.06326) is a neural network for object detection in LiDAR data. It works by discretizing the point-cloud onto an image (2D topview) and then applying a one-stage object detection based on YOLO (but simpler, without anchors). The innovation over YOLO is that it also outputs angles.

From visual inspection and a lot of debugging, everything seems to be working fine. Results may vary from the paper because a lot of details are missing from the paper, namely the optimizer strategy they have used. Other than that, pretty much everything is implemented except for metrics (mAP). You can use the evaluation code provided by the [KITTI bird's eye view evaluation page](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=bev) or you could easily adapt [this evaluation code of mine](https://github.com/rpmcruz/objdetect/blob/main/src/objdetect/metrics.py) (just replace that IoU function with the one provided in this code since bounding boxes here are angular).

Again, some things are unclear from the paper. I comment what these are in the code. When unclear, I went with my best guess based on the cited papers, including the subsequent paper [HDNET](http://proceedings.mlr.press/v87/yang18b/yang18b.pdf) from the same authors. In particular, the diagram in the paper showing the architecture is unclear to me (`model.py`); when undecided on the number of neurons and so on, I went with the smallest choice. Let me know if you make any improvements.

## Usage

* `train.py FILENAME` to train the model (saves to the given filename)
* `eval.py FILENAME` to evaluate the model (loads from a given filename)

The two most important parts of the implementation are: (1) the input transformation (`data.py`), and (2) the model itself (`model.py`). I have used KITTI (like the paper), but it should be fairly easy to extend to other datasets (just copy and edit the `KITTI` class).

## Related Repositories

If you like this code, you might want to check out these two repositories of mine. Some code here was borrowed from them.

* [objdetect](https://github.com/rpmcruz/objdetect): package to help create one-stage object detection models.
* [pnets](https://github.com/rpmcruz/pnets): package to help with point-clouds.

![](picture.png)

-- Ricardo Cruz <rpcruz@fe.up.pt>
