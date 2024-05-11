# -*- coding: utf-8 -*-

"""
@Time    : 2024/5/10 22:15
@File    : metric.py
@Author  : zj
@Description: 
"""

import torch

from utils.metrics import bbox_iou, box_iou

if __name__ == '__main__':
    boxes1 = torch.tensor([[18.51009, 3.91383, 2.97982, 7.82766],
                           ])
    boxes2 = torch.tensor(([[18.00000, 3.00000, 0.40601, 0.48120],
                            [18.00000, 3.00000, 0.80957, 1.17773],
                            [18.00000, 3.00000, 1.94336, 1.96680],
                            [18.00000, 3.00000, 3.35352, 4.62891],
                            [18.00000, 3.00000, 9.17188, 8.81250], ]))

    iou = bbox_iou(boxes1, boxes2, xywh=True, GIoU=False)
    print(f"iou: {iou}\n- shape: {iou.shape}")

    print(iou.squeeze(), iou.squeeze().shape)
