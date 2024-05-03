# -*- coding: utf-8 -*-

"""
@Time    : 2024/4/5 22:34
@File    : components.py
@Author  : zj
@Description: 
"""

import numpy as np

import torch
import torch.nn as nn

from models.common import Conv


class Reorg(nn.Module):

    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        # [1, 64, 26, 26]
        N, C, H, W = x.shape[:4]
        ws = self.stride
        hs = self.stride

        # [N, C, H, W] -> [N, C, H/S, S, W/S, S] -> [N, C, H/S, W/S, S, S]
        # [1, 64, 26, 26] -> [1, 64, 13, 2, 13, 2] -> [1, 64, 13, 13, 2, 2]
        x = x.view(N, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        # [N, C, H/S, W/S, S, S] -> [N, C, H/S * W/S, S * S] -> [N, C, S * S, H/S * W/S]
        # [1, 64, 13, 13, 2, 2] -> [1, 64, 13 * 13, 2 * 2] -> [1, 64, 2 * 2, 13 * 13]
        x = x.view(N, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        # [N, C, S * S, H/S * W/S] -> [N, C, S * S, H/S, W/S] -> [N, S * S, C, H/S, W/S]
        # [1, 64, 2 * 2, 13 * 13] -> [1, 64, 2*2, 13, 13] -> [1, 2*2, 64, 13, 13]
        x = x.view(N, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        # [N, S * S, C, H/S, W/S] -> [N, S * S * C, H/S, W/S]
        # [1, 2*2, 64, 13, 13] -> [1, 2*2*64, 13, 13]
        x = x.view(N, hs * ws * C, int(H / hs), int(W / ws)).contiguous()
        # [1, 256, 13, 13]
        return x


class YOLOv2Detect(nn.Module):
    # YOLOv2 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # print(f"m.anchors: {self.anchors}")
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            # YOLOv2 use 5 anchors
            bs, _, ny, nx = x[i].shape  # x(bs,425,20,20) to x(bs,5,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[-2:] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(bs, nx, ny, i)

                # b_x = sigmoid(t_x) + c_x
                # b_y = sigmoid(t_y) + c_y
                # b_w = p_w * e^t_w
                # b_h = p_h * e^t_h
                #
                # x/y/conf compress to [0,1]
                # [bs, 5, 20, 20, 3]
                xy_conf = torch.sigmoid(x[i][..., np.r_[:2, 4:5]])
                xy_conf[..., 0] += self.grid[i][0]
                xy_conf[..., 1] += self.grid[i][1]
                # exp()
                # [bs, 5, 20, 20, 2]
                wh = torch.exp(x[i][..., 2:4])
                wh[..., 0] *= self.anchor_grid[i][0]
                wh[..., 1] *= self.anchor_grid[i][1]
                # calculate classification probability
                # [bs, 5, 20, 20, 80]
                probs = torch.softmax(x[i][..., 5:], dim=-1)

                # [xcyc, wh, conf, probs]
                # [bs, 5, 20, 20, 85]
                y = torch.cat((xy_conf[..., :2], wh, xy_conf[..., 2:], probs), dim=4)
                # Scale relative to image width/height
                y[..., :4] *= self.stride[i]

                z.append(y.view(bs, self.na * ny * nx, self.no))

        # 训练阶段，返回特征层数据（List[Tensor]） [1, 5, 8, 8, 85]
        # 1: 批量大小
        # 5: 锚点个数
        # 8: 特征数据高
        # 9: 特征数据宽
        # 85: 预测框坐标（xc/yc/box_w/box_h）+预测框置信度+类别数
        # 推理阶段，返回特征层数据+推理结果（Tuple(Tensor, Tensor)）
        #         如果导出，仅返回推理结果（bs, 每个特征层输出的预测锚点个数, 每个锚点输出维度）
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, bs, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype

        # print(f"anchors: {self.anchors}")
        # print(f"stride: {self.stride}")
        # print(f"nc: {self.nc}")

        # grid coordinate
        # [F] -> [B, n_anchors, F_H, F_W]
        x_shift = torch.broadcast_to(torch.arange(nx), (bs, self.na, ny, nx)).to(dtype=t, device=d)
        # [F] -> [F, 1] -> [B, n_anchors, F_H, F_W]
        y_shift = torch.broadcast_to(torch.arange(ny).reshape(ny, 1), (bs, self.na, ny, nx)).to(dtype=t, device=d)

        # broadcast anchors to all grids
        # [n_anchors] -> [1, n_anchors, 1, 1] -> [B, n_anchors, F_H, F_W]
        w_anchors = torch.broadcast_to(self.anchors[i][:, 0].reshape(1, self.na, 1, 1),
                                       [bs, self.na, ny, nx]).to(dtype=t, device=d)
        h_anchors = torch.broadcast_to(self.anchors[i][:, 1].reshape(1, self.na, 1, 1),
                                       [bs, self.na, ny, nx]).to(dtype=t, device=d)

        return torch.stack([x_shift, y_shift]), torch.stack([w_anchors, h_anchors])


class YOLOv3Detect(YOLOv2Detect):
    # YOLOv3 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            # YOLOv2 use 5 anchors
            bs, _, ny, nx = x[i].shape  # x(bs,425,20,20) to x(bs,5,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[-2:] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(bs, nx, ny, i)

                # b_x = sigmoid(t_x) + c_x
                # b_y = sigmoid(t_y) + c_y
                # b_w = p_w * e^t_w
                # b_h = p_h * e^t_h
                #
                # x/y/conf/probs compress to [0,1]
                # [bs, 5, 20, 20, 2+1+80]
                xy_conf_probs = torch.sigmoid(x[i][..., np.r_[0:2, 4:self.no]])
                xy_conf_probs[..., 0] += self.grid[i][0]
                xy_conf_probs[..., 1] += self.grid[i][1]
                # exp()
                # [bs, 5, 20, 20, 2]
                wh = torch.exp(x[i][..., 2:4])
                wh[..., 0] *= self.anchor_grid[i][0]
                wh[..., 1] *= self.anchor_grid[i][1]

                # [xcyc, wh, conf, probs]
                # [bs, 5, 20, 20, 85]
                y = torch.cat((xy_conf_probs[..., :2], wh, xy_conf_probs[..., 2:]), dim=4)
                # Scale relative to image width/height
                y[..., :4] *= self.stride[i]

                z.append(y.view(bs, self.na * ny * nx, self.no))

        # 训练阶段，返回特征层数据（List[Tensor]） [1, 5, 8, 8, 85]
        # 1: 批量大小
        # 5: 锚点个数
        # 8: 特征数据高
        # 9: 特征数据宽
        # 85: 预测框坐标（xc/yc/box_w/box_h）+预测框置信度+类别数
        # 推理阶段，返回特征层数据+推理结果（Tuple(Tensor, Tensor)）
        #         如果导出，仅返回推理结果（bs, 每个特征层输出的预测锚点个数, 每个锚点输出维度）
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, bs, nx=20, ny=20, i=0):
        return super()._make_grid(bs, nx, ny, i)


class ResBlock(nn.Module):

    def __init__(self, ch, num_blocks=1, shortcut=True):

        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(num_blocks):
            self.module_list.append(nn.Sequential(
                # 1x1卷积，通道数减半，不改变空间尺寸
                Conv(ch, ch // 2, 1, 1, 0),
                # 3x3卷积，通道数倍增，恢复原始大小，不改变空间尺寸
                Conv(ch // 2, ch, 3, 1, 1)
            ))

    def forward(self, x):
        for module in self.module_list:
            h = x
            h = module(h)
            x = x + h if self.shortcut else h
        return x
