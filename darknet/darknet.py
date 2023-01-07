# -*- coding: utf-8 -*-

"""
@date: 2023/1/7 下午12:26
@file: darknet.py
@author: zj
@description: 
"""

import torch
from torch import nn


class ConvBNAct(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int):
        super().__init__()
        pad = (kernel_size - 1) // 2
        # H_out = floor((H_in + 2 * Pad - Dilate * (Kernel - 1) - 1) / Stride + 1)
        #       = floor((H_in + 2 * (Kernel - 1) // 2 - Dilate * (Kernel - 1) - 1) / Stride + 1)

        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=(kernel_size, kernel_size),
                              stride=(stride, stride),
                              padding=pad,
                              bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class ResBlock(nn.Module):

    def __init__(self, ch, num_blocks=1, shortcut=True):

        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(num_blocks):
            self.module_list.append(nn.Sequential(
                # 1x1卷积，通道数减半，不改变空间尺寸
                ConvBNAct(ch, ch // 2, 1, 1),
                # 3x3卷积，通道数倍增，恢复原始大小，不改变空间尺寸
                ConvBNAct(ch // 2, ch, 3, 1)
            ))

    def forward(self, x):
        for module in self.module_list:
            h = x
            h = module(h)
            x = x + h if self.shortcut else h
        return x


class DownSample(nn.Module):

    def __init__(self, in_ch=32, out_ch=64, kernel_size=3, stride=2, num_blocks=1, shortcut=True):
        super(DownSample, self).__init__()
        self.conv = ConvBNAct(in_ch=in_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride)
        self.res_block = ResBlock(ch=out_ch, num_blocks=num_blocks, shortcut=shortcut)

    def forward(self, x):
        x = self.conv(x)
        x = self.res_block(x)

        return x


class Backbone(nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()
        self.stem = ConvBNAct(in_ch=3, out_ch=32, kernel_size=3, stride=1)

        self.stage1 = DownSample(in_ch=32, out_ch=64, kernel_size=3, stride=2, num_blocks=1)
        self.stage2 = DownSample(in_ch=64, out_ch=128, kernel_size=3, stride=2, num_blocks=2)
        self.stage3 = DownSample(in_ch=128, out_ch=256, kernel_size=3, stride=2, num_blocks=8)
        self.stage4 = DownSample(in_ch=256, out_ch=512, kernel_size=3, stride=2, num_blocks=8)
        self.stage5 = DownSample(in_ch=512, out_ch=1024, kernel_size=3, stride=2, num_blocks=4)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x


class Darknet53(nn.Module):

    def __init__(self, num_classes=1000):
        super(Darknet53, self).__init__()
        self.backbone = Backbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.reshape(x.shape[:2])

        x = self.classifier(x)
        return x


if __name__ == '__main__':
    m = Darknet53()
    data = torch.randn(1, 3, 224, 224)

    output = m(data)
    print(output.shape)
