# -*- coding: utf-8 -*-

"""
@Time    : 2024/5/1 16:25
@File    : yolov3loss.py
@Author  : zj
@Description:

标注框是和锚点框匹配的，所以每个特征层的锚点框确定之后，就可以计算得出标注框最匹配的是哪一层的锚点框，也就是说，该标注框位于哪一个特征层进行预测。

"""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from utils.torch_utils import de_parallel
from .yolov2loss import bboxes_iou


class YOLOv3Loss(nn.Module):

    def __init__(self, model, ignore_thresh=0.7,
                 coord_scale=1.0, noobj_scale=1.0, obj_scale=5.0, class_scale=1.0):
        super(YOLOv3Loss, self).__init__()
        device = next(model.parameters()).device  # get model device

        m = de_parallel(model).model[-1]  # Detect() module
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        assert self.nl == 3, "YOLOv3Loss supports three-layer feature loss calculation, starting from YOLOv3Loss, it supports multi-layer feature loss"
        self.anchors = m.anchors
        self.device = device

        self.no = self.nc + 5  # number of outputs per anchor
        self.ignore_thresh = ignore_thresh

        self.noobj_scale = noobj_scale
        self.obj_scale = obj_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale

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
            box_target, box_mask, box_scale, iou_target, iou_mask, class_target, class_mask = \
                self.build_targets(p[i].detach().clone(), targets.clone(), i)

            bs, _, ny, nx, _ = p[i].shape  # pi(bs,5,20,20,85)
            # pi = p[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # x/y compress to [0,1]
            xy = torch.sigmoid(p[i][..., 0:2])
            # exp()
            wh = torch.exp(p[i][..., 2:4])
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
            # [bi, n_anchors, f_h*f_w, 4] -> [bs*n_anchors*f_h*f_w, 4]
            if torch.sum(box_mask > 0) > 0:
                box_target = box_target.reshape(-1, 4)[box_mask > 0]
                box_scale = box_scale.reshape(-1)[box_mask > 0].reshape(-1, 1)

                xy_pred = xy.reshape(-1, 2)[box_mask > 0]
                xy_loss = F.mse_loss(xy_pred, box_target[..., :2], reduction='none')

                wh_pred = wh.reshape(-1, 2)[box_mask > 0]
                wh_loss = F.mse_loss(wh_pred, box_target[..., 2:], reduction='none')
                wh_loss *= box_scale

                box_loss = xy_loss.sum() + wh_loss.sum()

            # --------------------------------------
            # iou loss
            # [bi, n_anchors, f_h*f_w, 1] -> [bs*n_anchors*f_h*f_w]
            iou_mask = iou_mask.reshape(-1)

            if torch.sum(iou_mask == 2) > 0:
                obj_iou_target = iou_target.reshape(-1)[iou_mask == 2]
                obj_iou_pred = conf.reshape(-1)[iou_mask == 2]
                # obj_iou_loss = F.mse_loss(obj_iou_pred, obj_iou_target, reduction='sum')
                obj_iou_loss = F.binary_cross_entropy_with_logits(obj_iou_pred, obj_iou_target, reduction='sum')

            if torch.sum(iou_mask == 1) > 0:
                noobj_iou_target = iou_target.reshape(-1)[iou_mask == 1]
                noobj_iou_pred = conf.reshape(-1)[iou_mask == 1]
                noobj_iou_loss = F.mse_loss(torch.sigmoid(noobj_iou_pred), noobj_iou_target, reduction='sum')

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
        box_target, box_mask, box_scale, iou_target, iou_mask, class_target, class_mask = \
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

            # 预测框和标注框的IoU，如果大于忽略阈值，那么不计算目标置信损失
            # ([n_anchors*f_h*f_w, 4], [num_obj, 4]) -> [n_anchors*f_h*f_w, num_obj] -> [n_anchors, f_h*f_w, num_obj]
            ious = bboxes_iou(pred_boxes[bi].reshape(-1, 4), gt_boxes, xyxy=False).reshape(self.na, -1, num_obj)

            # 计算每个网格中每个预测框计算得到的最大IoU
            # shape: (H * W, num_anchors, 1)
            max_iou, _ = torch.max(ious, dim=-1, keepdim=True)
            # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
            # 对于正样本(iou大于阈值), 不参与计算
            n_pos = torch.nonzero(max_iou.view(-1) > self.ignore_thresh).numel()
            if n_pos > 0:
                # 如果存在, 那么不参与损失计算
                iou_mask[bi][max_iou > self.ignore_thresh] = 0

            # print(f"gt_boxes: {gt_boxes}")
            # 计算标注框和所有锚点框的IoU，如果标注框最匹配的锚点框位于该层，计算损失
            truth_anchor_ious = bboxes_iou(truth_boxes, self.ref_anchors, xyxy=True)
            # 每个标注框最匹配的锚点框下标
            best_n_all = torch.argmax(truth_anchor_ious, dim=-1)
            best_n_mask = ((best_n_all == self.anchor_masks[i][0]) |
                           (best_n_all == self.anchor_masks[i][1]) | (best_n_all == self.anchor_masks[i][2]))
            # 如果没有标注框和该层锚点框匹配，跳过
            if torch.sum(best_n_mask) <= 0:
                continue

            for ni in range(num_obj):
                # 该标注框对应的锚点框位于该层
                if best_n_mask[ni] > 0:
                    # compute the center of each gt box to determine which cell it falls on
                    # assign it to a specific anchor by choosing max IoU
                    # 首先计算锚点框的中心点位于哪个网格, 然后选择其中IoU最大的锚点框参与训练
                    # [4]: [xc, yc, w, h]
                    gt_box = gt_boxes[ni]
                    gt_class = gt_cls_ids[ni]
                    cell_idx_x, cell_idx_y = torch.floor(gt_box[:2]).long()
                    cell_idx = cell_idx_y * nx + cell_idx_x

                    argmax_anchor_idx = best_n_all[ni] % self.na

                    # update box_target, box_mask
                    box_target[bi, argmax_anchor_idx, cell_idx, 0] = gt_box[0] - gt_box[0].to(torch.int16)
                    box_target[bi, argmax_anchor_idx, cell_idx, 1] = gt_box[1] - gt_box[1].to(torch.int16)
                    box_target[bi, argmax_anchor_idx, cell_idx, 2] = gt_box[2] / self.anchors[i][argmax_anchor_idx, 0]
                    box_target[bi, argmax_anchor_idx, cell_idx, 3] = gt_box[3] / self.anchors[i][argmax_anchor_idx, 1]

                    box_mask[bi, argmax_anchor_idx, cell_idx, :] = 1

                    pred_box = pred_boxes[bi, argmax_anchor_idx, cell_idx]
                    scale = 2 - (pred_box[2] / nx) * (pred_box[3] / ny)
                    if scale < 1 or scale > 2:
                        box_scale[bi, argmax_anchor_idx, cell_idx, :] = 0
                    else:
                        box_scale[bi, argmax_anchor_idx, cell_idx, :] = scale

                    # update iou target and iou mask
                    iou_target[bi, argmax_anchor_idx, cell_idx, :] = 1
                    iou_mask[bi, argmax_anchor_idx, cell_idx, :] = 2

                    # update cls_target, cls_mask
                    class_target[bi, argmax_anchor_idx, cell_idx, gt_class] = 1
                    class_mask[bi, argmax_anchor_idx, cell_idx, :] = 1

        return box_target, box_mask, box_scale, iou_target, iou_mask, class_target, class_mask

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
        # pred_boxes[..., 0::2] = torch.clamp(pred_boxes[..., 0::2], 0, nx)
        # pred_boxes[..., 1::2] = torch.clamp(pred_boxes[..., 1::2], 0, ny)

        return pred_boxes

    def _build_mask(self, bs, nx=20, ny=20, i=0):
        d = self.device
        t = self.anchors[i].dtype

        # [B, n_anchors, F_H*F_W, 4]
        box_target = torch.zeros((bs, self.na, ny * nx, 4)).to(dtype=t, device=d)
        box_mask = torch.zeros((bs, self.na, ny * nx, 1)).to(dtype=t, device=d)
        box_scale = torch.zeros((bs, self.na, ny * nx, 1)).to(dtype=t, device=d)

        # [B, n_anchors, F_H*F_W, 1]
        iou_target = torch.zeros((bs, self.na, ny * nx, 1)).to(dtype=t, device=d)
        iou_mask = torch.ones((bs, self.na, ny * nx, 1)).to(dtype=t, device=d)

        # [B, n_anchors, F_H*F_W, N_classes]
        class_target = torch.zeros((bs, self.na, ny * nx, self.nc)).to(dtype=t, device=d)
        class_mask = torch.zeros((bs, self.na, ny * nx, 1)).to(dtype=t, device=d)

        return box_target, box_mask, box_scale, iou_target, iou_mask, class_target, class_mask
