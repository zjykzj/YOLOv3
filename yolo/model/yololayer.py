# -*- coding: utf-8 -*-

"""
@date: 2023/1/3 下午4:39
@file: yololayer.py
@author: zj
@description: 
"""

import numpy as np

import torch
from torch import nn
from torch import Tensor

from yolo.util.box_utils import xywh2xyxy


class YOLOLayer(nn.Module):

    def __init__(self, masked_anchors, stride, num_classes=80):
        super(YOLOLayer, self).__init__()
        assert isinstance(masked_anchors, Tensor)
        self.masked_anchors = masked_anchors
        self.stride = stride
        self.num_classes = num_classes

        self.masked_anchors /= stride
        self.num_anchors = len(self.masked_anchors)

    def forward(self, outputs):
        if self.training:
            return outputs

        B, C, H, W = outputs.shape[:4]
        n_ch = 5 + self.num_classes
        assert C == (self.num_anchors * n_ch)

        dtype = outputs.dtype
        device = outputs.device

        # grid coordinate
        # [W] -> [1, 1, W, 1] -> [B, H, W, num_anchors]
        x_shift = torch.broadcast_to(torch.arange(W).reshape(1, 1, W, 1),
                                     (B, H, W, self.num_anchors)).to(dtype=dtype, device=device)
        # [H] -> [1, H, 1, 1] -> [B, H, W, num_anchors]
        y_shift = torch.broadcast_to(torch.arange(H).reshape(1, H, 1, 1),
                                     (B, H, W, self.num_anchors)).to(dtype=dtype, device=device)

        # broadcast anchors to all grids
        # [num_anchors] -> [1, 1, 1, num_anchors] -> [B, H, W, num_anchors]
        w_anchors = torch.broadcast_to(self.masked_anchors[:, 0].reshape(1, 1, 1, self.num_anchors),
                                       (B, H, W, self.num_anchors)).to(dtype=dtype, device=device)
        h_anchors = torch.broadcast_to(self.masked_anchors[:, 1].reshape(1, 1, 1, self.num_anchors),
                                       (B, H, W, self.num_anchors)).to(dtype=dtype, device=device)

        # Reshape
        # [B, num_anchors * (4+1+num_classes), H, W] ->
        # [B, H, W, num_anchors * (4+1+num_classes)] ->
        # [B, H, W, num_anchors, 4+1+num_classes]
        outputs = outputs.permute(0, 2, 3, 1).reshape(B, H, W, self.num_anchors, n_ch)

        # logistic activation
        # xy + obj_pred + cls_pred
        # outputs[..., np.r_[:2, 4:5]] = torch.sigmoid(outputs[..., np.r_[:2, 4:5]])
        outputs[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(outputs[..., np.r_[:2, 4:n_ch]])

        # b_x = sigmoid(t_x) + c_x
        # b_y = sigmoid(t_y) + c_y
        # b_w = exp(t_w) * p_w
        # b_h = exp(t_h) * p_h
        #
        # 预测框坐标x0加上每个网格的左上角坐标x
        outputs[..., 0] += x_shift
        # 预测框坐标y0加上每个网格的左上角坐标y
        outputs[..., 1] += y_shift
        # 计算预测框长/宽的实际长度
        outputs[..., 2] = torch.exp(outputs[..., 2]) * w_anchors
        outputs[..., 3] = torch.exp(outputs[..., 3]) * h_anchors

        # 分类概率压缩
        # outputs[..., 5:] = torch.softmax(outputs[..., 5:], dim=-1)

        # 推理阶段，不计算损失. 将预测框坐标按比例返回到原图大小
        outputs[..., :4] *= self.stride
        # [xc, yc, w, h] -> [x1, y1, x2, y2]
        outputs[..., :4] = xywh2xyxy(outputs[..., :4], is_center=True)
        # [B, H, W, n_anchors, n_ch] -> [B, H*W*n_anchors, n_ch]
        return outputs.reshape(B, -1, n_ch)
