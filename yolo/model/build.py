# -*- coding: utf-8 -*-

"""
@date: 2022/12/14 上午11:00
@file: build.py
@author: zj
@description: 
"""

import torch

from .yolov3 import YOLOv3


def build_model(args, cfg):
    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    # 预测正样本框阈值
    ignore_thre = cfg['TRAIN']['IGNORETHRE']
    model = YOLOv3(cfg['MODEL'], ignore_thre=ignore_thre)

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    # 默认在GPU环境下训练，差别在于是否进行分布式训练
    model = model.cuda().to(memory_format=memory_format)

    return model
