# -*- coding: utf-8 -*-

"""
@date: 2024/6/2 下午10:31
@file: iouloss.py
@author: zj
@description: 
"""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from utils.torch_utils import de_parallel
from utils.metrics import bbox_iou


class IOULoss(nn.Module):

    def __init__(self, model, ignore_thresh=0.7,
                 coord_scale=1.0, noobj_scale=1.0, obj_scale=5.0, class_scale=1.0,
                 GIoU=False, DIoU=False, CIoU=False):
        super(IOULoss, self).__init__()
        device = next(model.parameters()).device  # get model device

        m = de_parallel(model).model[-1]  # Detect() module
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        assert self.nl == 3, "IOULoss supports three-layer feature loss calculation, starting from YOLOv3Loss, it supports multi-layer feature loss"
        self.anchors = m.anchors
        self.device = device

        self.no = self.nc + 5  # number of outputs per anchor
        self.ignore_thresh = ignore_thresh

        self.noobj_scale = noobj_scale
        self.obj_scale = obj_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale

        self.giou = GIoU
        self.diou = DIoU
        self.ciou = CIoU

        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.ref_anchors = torch.zeros(self.nl * self.na, 4).to(dtype=self.anchors[0].dtype, device=self.device)
        self.anchor_masks = torch.arange(self.nl * self.na).reshape(self.nl, self.na)

    def forward(self, p: List[Tensor], targets: Tensor):
        """
        Perform forward pass of the network.

        Args:
            p (List[Tensor]): Predictions made by the network.
            Each item is a tensor with format [bs, n_anchors, feat_h, feat_w, (cxcy, wh, conf, cls_probs)].
            targets (Tensor): Ground truth targets. Each item format is [image_id, class_id, xc, yc, box_w, box_h].

        Returns:
            tensor: Loss value computed based on predictions and targets.
        """
        assert len(p) == self.nl, "The number of feature layers and prediction layers should be equal."

        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        lcls = torch.zeros(1, device=self.device)  # class loss

        if torch.sum(self.ref_anchors) == 0:
            for i in range(self.nl):
                bs, _, ny, nx, _ = p[i].shape  # pi(bs,5,20,20,85)
                self.ref_anchors[i * self.na:(i + 1) * self.na, 2] = self.anchors[i][..., 0] / nx
                self.ref_anchors[i * self.na:(i + 1) * self.na, 3] = self.anchors[i][..., 1] / ny

        for i in range(self.nl):
            box_target, box_mask, iou_target, iou_mask, class_target, class_mask = \
                self.build_targets(p[i].detach().clone(), targets.clone(), i)

            bs, _, ny, nx, _ = p[i].shape  # pi(bs,5,20,20,85)
            # pi = p[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            conf = p[i][..., 4]
            probs = p[i][..., 5:self.no]

            box_loss = torch.zeros(1, device=self.device)  # box loss
            obj_iou_loss = torch.zeros(1, device=self.device)  # object loss
            noobj_iou_loss = torch.zeros(1, device=self.device)  # object loss
            class_loss = torch.zeros(1, device=self.device)  # class loss
            # --------------------------------------
            # box loss
            # [bi, n_anchors, f_h*f_w, 1] -> [bs*n_anchors*f_h*f_w]
            box_mask = box_mask.reshape(-1)
            if torch.sum(box_mask > 0) > 0:
                box_target = box_target.reshape(-1)[box_mask > 0]
                box_loss = torch.sum(1 - box_target)

            # --------------------------------------
            # iou loss
            # [bi, n_anchors, f_h*f_w, 1] -> [bs*n_anchors*f_h*f_w]
            iou_mask = iou_mask.reshape(-1)

            if torch.sum(iou_mask == 2) > 0:
                obj_iou_target = iou_target.reshape(-1)[iou_mask == 2]
                obj_iou_pred = conf.reshape(-1)[iou_mask == 2]
                obj_iou_loss = F.binary_cross_entropy_with_logits(obj_iou_pred, obj_iou_target, reduction='sum')

            if torch.sum(iou_mask == 1) > 0:
                noobj_iou_target = iou_target.reshape(-1)[iou_mask == 1]
                noobj_iou_pred = conf.reshape(-1)[iou_mask == 1]
                noobj_iou_loss = F.binary_cross_entropy_with_logits(noobj_iou_pred, noobj_iou_target, reduction='sum')

            # --------------------------------------
            # class loss
            # [bi, n_anchors, f_h*f_w, 1] -> [bs*n_anchors*f_h*f_w]
            class_mask = class_mask.reshape(-1)
            if torch.sum(class_mask > 0) > 0:
                class_target = class_target.reshape(-1, self.nc)[class_mask > 0].reshape(-1)
                class_pred = probs.reshape(-1, self.nc)[class_mask > 0].reshape(-1)
                class_loss = F.binary_cross_entropy_with_logits(class_pred, class_target, reduction='sum')

            # calculate the loss, normalized by batch size.
            lbox += box_loss * self.coord_scale
            lobj += obj_iou_loss * self.obj_scale + noobj_iou_loss * self.noobj_scale
            lcls += class_loss * self.class_scale

        lbox /= bs
        lobj /= bs
        lcls /= bs
        return (lbox + lobj + lcls), torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, pi, targets, i=0):
        bs, _, ny, nx, _ = pi.shape  # pi(bs,5,20,20,85)
        pred_boxes = self._make_pred(pi, i)
        box_target, box_mask, iou_target, iou_mask, class_target, class_mask = \
            self._build_mask(bs, nx, ny, i)

        for bi in range(bs):
            num_obj = torch.sum(targets[..., 0] == bi)
            if num_obj == 0:
                # 这幅图像没有目标，那么仅计算负样本的目标置信度损失
                continue
            # 获取该幅图像的标注框和对应类别下标
            gt_targets = targets[targets[..., 0] == bi]
            gt_boxes = gt_targets[..., 2:6]

            truth_boxes = gt_boxes.clone()
            truth_boxes[..., :2] = 0

            gt_boxes[..., 0::2] *= nx
            gt_boxes[..., 1::2] *= ny
            gt_cls_ids = gt_targets[..., 1].long()

            # 逐个标注框进行计算
            for ni in range(num_obj):
                # 获取第ni个标注框
                # [4]: [xc, yc, w, h]
                gt_box = gt_boxes[ni]
                # 获取第ni个标注框的类别标签
                gt_class = gt_cls_ids[ni]
                # [4]: [0, 0, w, h]
                truth_box = truth_boxes[ni]

                cell_idx_x, cell_idx_y = torch.floor(gt_box[:2]).long()
                cell_idx = cell_idx_y * nx + cell_idx_x

                # 计算该标注框和所有预测框之间的IoU
                # [4] -> [1, 4]
                # [B, n_anchors, f_h*f_w, 4] -> [n_anchors*f_h*f_w, 4]
                # [n_anchors, f_h*f_w]
                pred_ious = bbox_iou(gt_box.unsqueeze(0), pred_boxes[bi].reshape(-1, 4), xywh=True,
                                     GIoU=self.giou, DIoU=self.diou, CIoU=self.ciou).squeeze().reshape(self.na, ny * nx)
                # 对于IoU大于阈值的负样本，不参与训练。
                # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
                # 对于iou大于阈值的负样本, 不参与计算
                n_pos = torch.nonzero(pred_ious.view(-1) > self.ignore_thresh).numel()
                if n_pos > 0:
                    # 如果存在, 那么不参与置信度损失计算
                    iou_mask[bi][pred_ious > self.ignore_thresh] = 0

                # [4] -> [1, 4]
                # [nl*na, 4]
                # [nl*na]
                anchor_ious = bbox_iou(truth_box.unsqueeze(0), self.ref_anchors, xywh=False,
                                       GIoU=self.giou, DIoU=self.diou, CIoU=self.ciou).squeeze()
                argmax_anchor_idx = torch.max(anchor_ious)
                if argmax_anchor_idx not in self.anchor_masks[i]:
                    # 该预测框最佳匹配的锚点框不是这一层
                    continue
                argmax_anchor_idx = argmax_anchor_idx % self.na

                # update box_target, box_mask
                box_target[bi, argmax_anchor_idx, cell_idx, :] = pred_ious[argmax_anchor_idx, cell_idx]
                box_mask[bi, argmax_anchor_idx, cell_idx, :] = 1

                # update iou target and iou mask
                iou_target[bi, argmax_anchor_idx, cell_idx, :] = 1
                iou_mask[bi, argmax_anchor_idx, cell_idx, :] = 2

                # update cls_target, cls_mask
                class_target[bi, argmax_anchor_idx, cell_idx, gt_class] = 1
                class_mask[bi, argmax_anchor_idx, cell_idx, :] = 1

        return box_target, box_mask, iou_target, iou_mask, class_target, class_mask

    def _make_grid(self, bs, nx=20, ny=20, i=0):
        d = self.device
        t = self.anchors[i].dtype

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

        # [2, B, n_anchors, F_H, F_W], [2, B, n_anchors, F_H, F_W]
        return torch.stack([x_shift, y_shift]), torch.stack([w_anchors, h_anchors])

    def _make_pred(self, pi, i=0):
        bs, _, ny, nx, _ = pi.shape  # pi(bs,5,85,20,20)
        if self.grid[i].shape[-2:] != pi.shape[2:4] or self.grid[i].shape[1] != bs:
            self.grid[i], self.anchor_grid[i] = self._make_grid(bs, nx, ny, i)

        # b_x = sigmoid(t_x) + c_x
        # b_y = sigmoid(t_y) + c_y
        # b_w = p_w * e^t_w
        # b_h = p_h * e^t_h
        #
        # x/y compress to [0,1]
        # [bs, 5, 20, 20, 2]
        xy = torch.sigmoid(pi[..., :2])
        xy[..., 0] += self.grid[i][0]
        xy[..., 1] += self.grid[i][1]
        # exp([bs, 5, 20, 20, 2])
        wh = torch.exp(pi[..., 2:4])
        wh[..., 0] *= self.anchor_grid[i][0]
        wh[..., 1] *= self.anchor_grid[i][1]

        # [bs, n_anchors, f_h, f_w, 4] -> [bs, n_anchors, f_h*f_w, 4]
        pred_boxes = torch.cat((xy, wh), dim=4).reshape(bs, self.na, -1, 4)

        return pred_boxes

    def _build_mask(self, bs, nx=20, ny=20, i=0):
        d = self.device
        t = self.anchors[i].dtype

        # [B, n_anchors, f_h*f_w, 1]
        box_target = torch.zeros((bs, self.na, ny * nx, 1)).to(dtype=t, device=d)
        box_mask = torch.zeros((bs, self.na, ny * nx, 1)).to(dtype=t, device=d)

        # [B, n_anchors, f_h*f_w, 1]
        iou_target = torch.zeros((bs, self.na, ny * nx, 1)).to(dtype=t, device=d)
        iou_mask = torch.ones((bs, self.na, ny * nx, 1)).to(dtype=t, device=d)

        # [B, n_anchors, f_h*f_w, n_classes]
        class_target = torch.zeros((bs, self.na, ny * nx, self.nc)).to(dtype=t, device=d)
        class_mask = torch.zeros((bs, self.na, ny * nx, 1)).to(dtype=t, device=d)

        return box_target, box_mask, iou_target, iou_mask, class_target, class_mask
