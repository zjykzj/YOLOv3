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


class YOLOLayer(nn.Module):

    def __init__(self, anchors, stride, num_classes=80):
        super(YOLOLayer, self).__init__()
        assert isinstance(anchors, Tensor)
        self.anchors = anchors
        self.stride = stride
        self.num_classes = num_classes

        self.num_anchors = len(self.anchors)

    def forward(self, outputs):
        if self.training:
            return outputs

        B, C, H, W = outputs.shape[:4]
        n_ch = 5 + self.num_classes
        assert C == (self.num_anchors * n_ch)

        dtype = outputs.dtype
        device = outputs.device

        # grid coordinate
        # [F_size] -> [B, num_anchors, H, W]
        x_shift = torch.broadcast_to(torch.arange(W),
                                     (B, self.num_anchors, H, W)).to(dtype=dtype, device=device)
        # [F_size] -> [f_size, 1] -> [B, num_anchors, H, W]
        y_shift = torch.broadcast_to(torch.arange(H).reshape(H, 1),
                                     (B, self.num_anchors, H, W)).to(dtype=dtype, device=device)

        # broadcast anchors to all grids
        # [num_anchors] -> [1, num_anchors, 1, 1] -> [B, num_anchors, H, W]
        w_anchors = torch.broadcast_to(self.anchors[:, 0].reshape(1, self.num_anchors, 1, 1),
                                       [B, self.num_anchors, H, W]).to(dtype=dtype, device=device)
        h_anchors = torch.broadcast_to(self.anchors[:, 1].reshape(1, self.num_anchors, 1, 1),
                                       [B, self.num_anchors, H, W]).to(dtype=dtype, device=device)

        # Reshape
        # [B, num_anchors * (4+1+num_classes), H, W] ->
        # [B, num_anchors, 4+1+num_classes, H, W] ->
        # [B, num_anchors, H, W, 4+1+num_classes]
        output = outputs.reshape(B, self.num_anchors, n_ch, H, W).permute(0, 1, 3, 4, 2)

        # logistic activation
        # xy + obj_pred + cls_pred
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

        pred = output
        # b_x = sigmoid(t_x) + c_x
        # b_y = sigmoid(t_y) + c_y
        # b_w = exp(t_w) * p_w
        # b_h = exp(t_h) * p_h
        #
        # 预测框坐标x0加上每个网格的左上角坐标x
        pred[..., 0] += x_shift
        # 预测框坐标y0加上每个网格的左上角坐标y
        pred[..., 1] += y_shift
        # 计算预测框长/宽的实际长度
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors

        # 推理阶段，不计算损失. 将预测框坐标按比例返回到原图大小
        pred[..., :4] *= self.stride
        # [B, n_anchors, F_H, F_W, n_ch] -> [B, n_anchors * F_H * F_W, n_ch]
        return pred.reshape(B, -1, n_ch)
