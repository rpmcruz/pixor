import torch
import torchvision
from torch.nn.functional import relu
import mydata

# Our hero: PIXOR https://arxiv.org/abs/1902.06326
# There are slight variations on resblocks. I followed the citation from PIXOR
# "Identity Mappings in Deep Residual Networks". Three things are unclear:
# (1) PIXOR says "[t]he first convolution of each residual block has a stride of
# 2", but this messes up with the residual addition. I decided to apply 3 convs,
# the first with stride=2, the two others like the previous paper.
# (2) The final res convolution must have the same channels has the input (for
# the pixelwise sum to be possible). When the diagram shows 24-24-96, I always
# use 24-24-24.
# (3) PIXOR says "second to fifth blocks are composed of residual layers (with
# number of layers equals to 3, 6, 6, 3 respectively)", but the diagram always
# shows three sets of weights per block, never 6. I decided to go always with
# 3 layers, per the diagram.

class ResBlock(torch.nn.Module):
    def __init__(self, *channels):
        super().__init__()
        assert len(channels) == 4
        self.downconv = torch.nn.Conv2d(channels[0], channels[1], 3, stride=2,
            padding=1)
        self.F = torch.nn.Sequential(
            torch.nn.BatchNorm2d(channels[1]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels[1], channels[2], 3, padding=1),
            torch.nn.BatchNorm2d(channels[2]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels[2], channels[3], 3, padding=1),
        )

    def forward(self, x):
        x = self.downconv(x)
        return x + self.F(x)

class Upsample(torch.nn.Module):
    def __init__(self, left_channels, above_channels, out_channels, fix_odd):
        super().__init__()
        self.conv = torch.nn.Conv2d(left_channels, out_channels, 1)
        self.deconv = torch.nn.ConvTranspose2d(above_channels, out_channels, 3,
            stride=2, padding=1, output_padding=1)
        self.fix_odd = fix_odd

    def forward(self, left, above):
        left = self.conv(left)
        above = self.deconv(above)
        # PIXOR activation maps are not always divisible by 2 (i.e. it
        # downsamples 175->88, so we need to truncate deconv (88->175)
        if self.fix_odd:
            above = above[:, :, :-1]
        # the paper does "pixel-wise summation"
        return left + above

class Pixor(torch.nn.Module):
    def __init__(self, ratio_grid2feature):
        super().__init__()
        self.ratio_grid2feature = ratio_grid2feature
        self.block1 = torch.nn.Sequential(
            # PIXOR paper has a typo in the diagram (the input has 36 channels,
            # but it's clear from the text that the input channels is 38).
            torch.nn.Conv2d(38, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.ReLU(),
        )
        # see explanations at the beginning for why the output res channels is
        # different than the diagram
        self.resblock2 = ResBlock(32, 24, 24, 24)  # 96
        self.resblock3 = ResBlock(24, 48, 48, 48)  # 192
        self.resblock4 = ResBlock(48, 64, 64, 64)  # 256
        self.resblock5 = ResBlock(64, 96, 96, 96)  # 384
        # for the following three layers, I apply relu in the forward pass
        self.conv_resblock5 = torch.nn.Conv2d(96, 196, 1)
        self.upsample6 = Upsample(64, 196, 128, False)
        self.upsample7 = Upsample(48, 128, 96, True)
        self.header = torch.nn.Sequential(
            torch.nn.Conv2d(96, 96, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(96, 96, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(96, 96, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(96, 96, 3, padding=1),
        )
        self.scores = torch.nn.Conv2d(96, 1, 3, padding=1)
        self.bboxes = torch.nn.Conv2d(96, 6, 3, padding=1)

    def forward(self, x, threshold=0.5):
        x = self.block1(x)
        x = self.resblock2(x)
        x3 = self.resblock3(x)
        x4 = self.resblock4(x3)
        x5 = self.resblock5(x4)
        x = relu(self.conv_resblock5(x5))
        x = relu(self.upsample6(x4, x))
        x = relu(self.upsample7(x3, x))
        x = self.header(x)
        scores = self.scores(x)
        bboxes = self.bboxes(x)
        if not self.training:
            # when in evaluation mode, convert the output grid back into list
            scores = scores.numpy()
            bboxes = bboxes.numpy()
            scores = [inv_scores(ss, threshold) for ss in scores]
            bboxes = [inv_bboxes(ss, threshold, bb, self.ratio_grid2feature)
                for ss, bb in zip(scores, bboxes)]
        return scores, bboxes
