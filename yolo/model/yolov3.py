# -*- coding: utf-8 -*-

"""
@date: 2023/1/3 上午11:20
@file: darknet.py
@author: zj
@description: 
"""

import os

import torch
from torch import nn

from darknet.darknet import ConvBNAct, ResBlock, Darknet53, FastDarknet53
from .yololayer import YOLOLayer

from yolo.util import logging

logger = logging.get_logger(__name__)


class DarknetBackbone(nn.Module):

    def __init__(self, arch='Darknet53', pretrained=None):
        super().__init__()
        self.arch = arch
        self.pretrained = pretrained

        if 'Darknet53' == arch:
            self.darknet = Darknet53(in_channel=3, num_classes=1000)
        elif 'FastDarknet53' == arch:
            self.darknet = FastDarknet53(in_channel=3, num_classes=1000)
        else:
            raise ValueError(f"{arch} doesn't supports")

        self._init_weights(pretrained=pretrained)

    def _init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)

        if pretrained is not None and pretrained != '':
            assert os.path.isfile(pretrained), pretrained
            logger.info(f'Loading pretrained {self.arch}: {pretrained}')

            state_dict = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # strip the names

            self.darknet.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        x = self.darknet.backbone.stem(x)

        x = self.darknet.backbone.stage1(x)
        x = self.darknet.backbone.stage2(x)
        x3 = self.darknet.backbone.stage3(x)
        x4 = self.darknet.backbone.stage4(x3)
        x5 = self.darknet.backbone.stage5(x4)

        return x3, x4, x5


class FPNNeck(nn.Module):

    def __init__(self):
        super(FPNNeck, self).__init__()

        self.module1 = nn.Sequential(
            ResBlock(ch=1024, num_blocks=2, shortcut=False),
            ConvBNAct(in_ch=1024, out_ch=512, kernel_size=1, stride=1)
        )

        self.module2 = nn.Sequential(
            ConvBNAct(in_ch=512, out_ch=256, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        self.module3 = nn.Sequential(
            ConvBNAct(in_ch=768, out_ch=256, kernel_size=1, stride=1),
            ConvBNAct(in_ch=256, out_ch=512, kernel_size=3, stride=1),
            ResBlock(ch=512, num_blocks=1, shortcut=False),
            ConvBNAct(in_ch=512, out_ch=256, kernel_size=1, stride=1)
        )

        self.module4 = nn.Sequential(
            ConvBNAct(in_ch=256, out_ch=128, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )

        self.module5 = ConvBNAct(in_ch=384, out_ch=128, kernel_size=1, stride=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x3, x4, x5):
        f1 = self.module1(x5)

        f2 = self.module2(f1)
        assert f2.shape[2:] == x4.shape[2:]
        f2 = torch.cat((f2, x4), 1)
        f2 = self.module3(f2)

        f3 = self.module4(f2)
        assert f3.shape[2:] == x3.shape[2:]
        f3 = torch.cat((f3, x3), 1)
        f3 = self.module5(f3)

        return f1, f2, f3


class YOLOv3Head(nn.Module):
    strides = [32, 16, 8]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    def __init__(self, anchors, num_classes=80):
        super(YOLOv3Head, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes

        output_ch = (4 + 1 + self.num_classes) * 3

        self.yolo1 = nn.Sequential(
            ConvBNAct(in_ch=512, out_ch=1024, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=1024,
                      out_channels=output_ch,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            YOLOLayer(self.anchors[self.anchor_mask[0]], self.strides[0], num_classes=self.num_classes)
        )

        self.yolo2 = nn.Sequential(
            ConvBNAct(in_ch=256, out_ch=512, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=512,
                      out_channels=output_ch,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            YOLOLayer(self.anchors[self.anchor_mask[1]], self.strides[1], num_classes=self.num_classes)
        )

        self.yolo3 = nn.Sequential(
            ConvBNAct(in_ch=128, out_ch=256, kernel_size=3, stride=1),
            ResBlock(ch=256, num_blocks=2, shortcut=False),
            nn.Conv2d(in_channels=256,
                      out_channels=output_ch,
                      kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            YOLOLayer(self.anchors[self.anchor_mask[2]], self.strides[2], num_classes=self.num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, f1, f2, f3):
        assert f1.shape[1] == 512
        o1 = self.yolo1(f1)

        assert f2.shape[1] == 256
        o2 = self.yolo2(f2)

        assert f3.shape[1] == 128
        o3 = self.yolo3(f3)

        return o1, o2, o3


class YOLOv3(nn.Module):

    def __init__(self, anchors, num_classes=20, arch='Darknet53', pretrained=None):
        super(YOLOv3, self).__init__()

        self.backbone = DarknetBackbone(arch=arch, pretrained=pretrained)
        self.neck = FPNNeck()
        self.head = YOLOv3Head(anchors, num_classes=num_classes)

    def forward(self, x):
        # x: [B, 3, H, W]
        x3, x4, x5 = self.backbone(x)
        # x3: [B, 256, H/8, W/8]
        # x4: [B, 512, H/16, W/16]
        # x5: [B, 1024, H/32, W/32]
        f1, f2, f3 = self.neck(x3, x4, x5)
        # f1: [B, 512, H/32, W/32]
        # f2: [B, 256, H/16, W/16]
        # f3: [B, 128, H/8, W/8]
        o1, o2, o3 = self.head(f1, f2, f3)

        if self.training:
            # o1: [B, num_anchors*(5+num_classes), H/32, W/32]
            # o2: [B, num_anchors*(5+num_classes), H/16, W/16]
            # o3: [B, num_anchors*(5+num_classes), H/8, W/8]
            return [o1, o2, o3]
        else:
            # res: [B, (H*W + 2H*2W + 4H*4W) / (32*32) * 3, 85] = [B, H*W*63 / (32*32), 85]
            return torch.cat((o1, o2, o3), dim=1)
