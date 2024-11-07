from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import exp


class CONV(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, padding=0, stride=1, dilation=1
               , groups=1, act=True):
        super().__init__()
        if act:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize, stride=stride,
                          padding=padding, dilation=dilation, groups=groups),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize, stride=stride,
                          padding=padding, dilation=dilation, groups=groups),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.conv(x)


class up_sample:
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                             align_corners=self.align_corners)


class reorg_layer(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.stride = s

    def forward(self, x):
        B, C, H, W = x.size()
        out_C = C * (self.stride ** 2)
        out_H = H // self.stride
        out_W = W // self.stride
        x = x.view(B, C, out_H, self.stride, out_W, self.stride)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, out_C, out_H, out_W)
        return x

class spp(nn.Module):
    def __init(self):
        super().__init()

    def forward(self, x):
        x_1 = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        x_2 = F.max_pool2d(x, kernel_size=9, stride=1, padding=4)
        x_3 = F.max_pool2d(x, kernel_size=13, stride=1, padding=6)
        x = torch.cat((x, x_1, x_2, x_3), dim=1)
        return x


class EMA:
    def __init__(self, model, decay=0.999, updates=0):
        self.model = deepcopy(model).eval()
        self.decay = decay
        self.updates = updates
        self.decay = lambda x: decay * (1 - exp(-x / 2000.))
        for param in self.model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.state_dict()
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()
