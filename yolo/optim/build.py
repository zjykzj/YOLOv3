# -*- coding: utf-8 -*-

"""
@date: 2022/12/14 上午10:58
@file: build.py
@author: zj
@description: 
"""

import torch
from torch import optim


def build_scheduler(cfg, optimizer):
    burn_in = cfg['TRAIN']['BURN_IN']
    steps = eval(cfg['TRAIN']['STEPS'])

    # Learning rate setup
    def burnin_schedule(i):
        if i < burn_in:
            # 在warmup阶段，使用线性学习率进行递增
            factor = pow(i / burn_in, 4)
        elif i < steps[0]:
            factor = 1.0
        elif i < steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)
    return scheduler


def build_optimizer(args, model):
    # Scale learning rate based on global batch size
    args.lr = args.lr * float(args.batch_size * args.world_size) / 256.
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    return optimizer
