# -*- coding: utf-8 -*-

"""
@date: 2023/5/4 下午3:02
@file: model.py
@author: zj
@description: 
"""

import numpy as np

import torch

from yolo.model.yolov3 import YOLOv3, DarknetBackbone, FPNNeck, YOLOv3Head


def init():
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


def test_backbone():
    data = torch.randn(1, 3, 608, 608)

    print("=> Darknet53 Train")
    model = DarknetBackbone(arch='Darknet53')
    model.train()

    x3, x4, x5 = model(data)
    print(x3.shape, x4.shape, x5.shape)

    print("=> Darknet53 Eval")
    model.eval()

    x3, x4, x5 = model(data)
    print(x3.shape, x4.shape, x5.shape)

    print("=> FastDarknet53 Train")
    model = DarknetBackbone(arch='FastDarknet53')
    model.train()

    x3, x4, x5 = model(data)
    print(x3.shape, x4.shape, x5.shape)

    print("=> FastDarknet53 Eval")
    model.eval()

    x3, x4, x5 = model(data)
    print(x3.shape, x4.shape, x5.shape)


def test_neck():
    x3 = torch.randn(1, 256, 76, 76)
    x4 = torch.randn(1, 512, 38, 38)
    x5 = torch.randn(1, 1024, 19, 19)

    print("=> FPNNeck Train")
    model = FPNNeck()
    model.train()

    f1, f2, f3 = model(x3, x4, x5)
    print(f1.shape, f2.shape, f3.shape)

    print("=> FPNNeck Eval")
    model.eval()

    f1, f2, f3 = model(x3, x4, x5)
    print(f1.shape, f2.shape, f3.shape)


def test_head():
    f1 = torch.randn(1, 512, 19, 19)
    f2 = torch.randn(1, 256, 38, 38)
    f3 = torch.randn(1, 128, 76, 76)

    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    anchors = torch.tensor(anchors, dtype=torch.float)

    print("=> YOLOv3Head Train")
    model = YOLOv3Head(anchors, num_classes=80)
    model.train()

    o1, o2, o3 = model(f1, f2, f3)
    print(o1.shape, o2.shape, o3.shape)

    print("=> YOLOv3Head Eval")
    model.eval()

    o1, o2, o3 = model(f1, f2, f3)
    print(o1.shape, o2.shape, o3.shape)


def test_yolov3():
    cfg_file = 'config/yolov3_default.cfg'
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    # data = torch.randn((1, 3, 416, 416))
    # data = torch.randn((1, 3, 608, 608))
    data = torch.randn((1, 3, 640, 640))

    anchors = cfg['MODEL']['ANCHORS']
    anchors = torch.tensor(anchors, dtype=torch.float)
    num_classes = cfg['MODEL']['N_CLASSES']
    arch = cfg['MODEL']['BACKBONE']
    model = YOLOv3(anchors, num_classes=num_classes, arch=arch)
    print(model)

    print("=> YOLOv3 Train")
    model.train()

    o1, o2, o3 = model(data)
    print(o1.shape, o2.shape, o3.shape)

    print("=> YOLOv3 Eval")
    model.eval()

    # data = torch.randn((1, 3, 416, 416))
    # data = torch.randn((1, 3, 608, 608))
    data = torch.randn((1, 3, 640, 640))
    outputs = model(data)
    print(outputs.shape)


if __name__ == '__main__':
    # test_backbone()
    # test_neck()
    # test_head()
    test_yolov3()
