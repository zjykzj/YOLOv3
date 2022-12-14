# -*- coding: utf-8 -*-

"""
@date: 2022/12/14 上午11:19
@file: build.py
@author: zj
@description: 
"""

import torch

from yolo.data.cocodataset import COCODataset
from yolo.utils.cocoapi_evaluator import COCOAPIEvaluator


def build_data(args, cfg):
    """
    针对数据集，需要设置
    1. 数据集路径 args.data
    2. 图像大小　cfg['TRAIN']['IMGSIZE']
    3. 数据增强策略　cfg['AUGMENTATION']
　
    针对数据加载器，需要设置
    1. 批量大小：当前最大支持16 cfg['TRAIN']['BATCHSIZE']
    2. 线程数：默认为4 args.workers
    """
    # YOLO使用的数据集，对于测试集，采用了COCO提供的评估器
    imgsize = cfg['TRAIN']['IMGSIZE']
    train_dataset = COCODataset(model_type=cfg['MODEL']['TYPE'],
                                # data_dir='COCO/',
                                data_dir=args.data,
                                img_size=imgsize,
                                augmentation=cfg['AUGMENTATION'])

    train_sampler = None
    # val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # collate_fn = lambda b: fast_collate(b, memory_format)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'], shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    return train_sampler, train_loader


def build_evaluator(args, cfg):
    evaluator = COCOAPIEvaluator(model_type=cfg['MODEL']['TYPE'],
                                 # data_dir='COCO/',
                                 data_dir=args.data,
                                 img_size=cfg['TEST']['IMGSIZE'],
                                 confthre=cfg['TEST']['CONFTHRE'],
                                 nmsthre=cfg['TEST']['NMSTHRE'])





    return evaluator
