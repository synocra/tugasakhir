# Ultralytics YOLO11 - Coordinate Attention Module
# Paper: "Coordinate Attention for Efficient Mobile Network Design" (CVPR 2021)
# Compatible with dynamic channel scaling in YOLO11

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CoordAtt']


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """
    Coordinate Attention that automatically adapts to input channel count.
    No need to hardcode 'inp' or 'oup' — it uses the actual input shape.
    This makes it fully compatible with YOLO11's width scaling (n, s, m, ...).
    """
    def __init__(self, reduction=32):
        super().__init__()
        self.reduction = reduction
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.act = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Pooling
        x_h = self.pool_h(x)  # [N, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [N, C, W, 1]

        # Concat along height dimension
        y = torch.cat([x_h, x_w], dim=2)  # [N, C, H+W, 1]

        # Channel reduction
        mip = max(8, c // self.reduction)
        conv1 = nn.Conv2d(c, mip, kernel_size=1, stride=1, padding=0, device=x.device, dtype=x.dtype)
        bn1 = nn.BatchNorm2d(mip, device=x.device, dtype=x.dtype)
        y = self.act(bn1(conv1(y)))

        # Split back
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [N, C, 1, W]

        # Generate attention maps
        conv_h = nn.Conv2d(mip, c, kernel_size=1, stride=1, padding=0, device=x.device, dtype=x.dtype)
        conv_w = nn.Conv2d(mip, c, kernel_size=1, stride=1, padding=0, device=x.device, dtype=x.dtype)

        a_h = torch.sigmoid(conv_h(x_h))
        a_w = torch.sigmoid(conv_w(x_w))

        return identity * a_h * a_w