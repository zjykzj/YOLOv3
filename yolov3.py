# -*- coding: utf-8 -*-

"""
@date: 2023/1/3 上午11:20
@file: darknet.py
@author: zj
@description: 
"""

import math
import torch
from torch import nn

from yolo.model.yolo_layer import YOLOLayer


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
        x3 = self.stage3(x)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        return x3, x4, x5


class Neck(nn.Module):

    def __init__(self):
        super(Neck, self).__init__()

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

    def forward(self, x, last_x1, last_x2):
        x = self.module1(x)

        x1 = self.module2(x)
        assert x1.shape[2:] == last_x1.shape[2:]
        x1 = torch.cat((x1, last_x1), 1)
        x1 = self.module3(x1)

        x2 = self.module4(x1)
        assert x2.shape[2:] == last_x2.shape[2:]
        x2 = torch.cat((x2, last_x2), 1)
        x2 = self.module5(x2)

        return x, x1, x2


class Head(nn.Module):

    def __init__(self, config_model, ignore_thresh=0.70):
        super(Head, self).__init__()
        self.yolo1 = nn.Sequential(
            ConvBNAct(in_ch=512, out_ch=1024, kernel_size=3, stride=1),
            YOLOLayer(config_model, layer_no=0, in_ch=1024, ignore_thre=ignore_thresh)
        )
        self.yolo2 = nn.Sequential(
            ConvBNAct(in_ch=256, out_ch=512, kernel_size=3, stride=1),
            YOLOLayer(config_model, layer_no=1, in_ch=512, ignore_thre=ignore_thresh)
        )
        self.yolo3 = nn.Sequential(
            ConvBNAct(in_ch=128, out_ch=256, kernel_size=3, stride=1),
            ResBlock(ch=256, num_blocks=2, shortcut=False),
            YOLOLayer(config_model, layer_no=2, in_ch=256, ignore_thre=ignore_thresh)
        )

    def forward(self, x1, x2, x3):
        assert x1.shape[1] == 512
        x1 = self.yolo1(x1)

        assert x2.shape[1] == 256
        x2 = self.yolo2(x2)

        assert x3.shape[1] == 128
        x3 = self.yolo3(x3)

        return x1, x2, x3


class YOLOv3(nn.Module):

    def __init__(self, config_model, ignore_thresh=0.7):
        super(YOLOv3, self).__init__()
        assert config_model['TYPE'] == 'YOLOv3'

        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head(config_model, ignore_thresh=ignore_thresh)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, 3, H, W]
        x1, x2, x3 = self.backbone(x)
        # x1: [B, 256, H/8, W/8]
        # x2: [B, 512, H/16, W/16]
        # x3: [B, 1024, H/32, W/32]
        print(x1.reshape(-1)[:5])
        print(x2.reshape(-1)[:5])
        print(x3.reshape(-1)[:5])
        x1, x2, x3 = self.neck(x3, x2, x1)
        # x1: [B, 512, H/32, W/32]
        # x2: [B, 256, H/16, W/16]
        # x3: [B, 128, H/8, W/8]
        x1, x2, x3 = self.head(x1, x2, x3)
        # x1: [B, H/32 * W/32 * 3, 85]
        # x2: [B, H/16 * W/16 * 3, 85]
        # x3: [B, H/8 * W/8 * 3, 85]

        # res: [B, (H*W + 2H*2W + 4H*4W) / (32*32) * 3, 85]
        #     =[B, H*W*63 / (32*32), 85
        return torch.cat((x1, x2, x3), 1)


if __name__ == '__main__':
    cfg_file = 'config/yolov3_default.cfg'
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    # 创建YOLOv3
    model = YOLOv3(cfg['MODEL'])
    # model = ConvBNAct(in_ch=3, out_ch=3, kernel_size=1, stride=1)
    print(model)
    model.eval()

    import random
    import numpy as np

    seed = 1  # seed必须是int，可以自行设置
    random.seed(seed)
    np.random.seed(seed)  # numpy产生的随机数一致
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
        torch.cuda.manual_seed_all(seed)  # 多卡模式下，让所有显卡生成的随机数一致？这个待验证
        # CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
        # 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
        torch.backends.cudnn.deterministic = True

        # 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
        # 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
        torch.backends.cudnn.benchmark = False

    # data = torch.randn((1, 3, 416, 416))
    # data = torch.randn((1, 3, 608, 608))
    data = torch.randn((1, 3, 640, 640))
    print(data.reshape(-1)[:20])
    outputs = model(data)
    print(outputs.shape)

    print(outputs.reshape(-1)[:20])
