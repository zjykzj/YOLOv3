# -*- coding: utf-8 -*-

"""
@date: 2023/5/4 下午9:21
@file: yololoss.py
@author: zj
@description:
"""
from typing import List
import numpy as np

import torch
from torch import nn
from torch import Tensor

import torch.nn.functional as F

from yolo.util.box_utils import bboxes_iou


def make_deltas(box1: Tensor, box2: Tensor) -> Tensor:
    """
    Calculate the delta values σ(t_x), σ(t_y), exp(t_w), exp(t_h) used for transforming box1 to  box2
    sigmoid(t_x) = b_x - c_x
    sigmoid(t_y) = b_y - c_y
    e^t_w = b_w / p_w
    e^t_h = b_h / p_h

    Arguments:
    box1 -- tensor of shape (N, 4) first set of boxes (c_x, c_y, w, h)
    box2 -- tensor of shape (N, 4) second set of boxes (c_x, c_y, w, h)

    Returns:
    deltas -- tensor of shape (N, 4) delta values (t_x, t_y, t_w, t_h)
                   used for transforming boxes to reference boxes
    """
    assert len(box1.shape) == len(box2.shape) == 2
    # [N, 4] -> [N]
    t_x = box2[:, 0] - box1[:, 0]
    t_y = box2[:, 1] - box1[:, 1]
    t_w = box2[:, 2] / box1[:, 2]
    t_h = box2[:, 3] / box1[:, 3]

    t_x = t_x.view(-1, 1)
    t_y = t_y.view(-1, 1)
    t_w = t_w.view(-1, 1)
    t_h = t_h.view(-1, 1)

    # σ(t_x), σ(t_y), exp(t_w), exp(t_h)
    deltas = torch.cat([t_x, t_y, t_w, t_h], dim=1)
    return deltas


def build_mask(B, H, W, num_anchors=3, num_classes=20, dtype=torch.float, device=torch.device('cpu')):
    # [B, H*W, num_anchors, 1]
    iou_target = torch.zeros((B, H * W, num_anchors, 1)).to(dtype=dtype, device=device)
    # [B, H*W, num_anchors, 1]
    iou_mask = torch.ones((B, H * W, num_anchors, 1)).to(dtype=dtype, device=device)

    # [B, H*W, num_anchors, 4]
    box_target = torch.zeros((B, H * W, num_anchors, 4)).to(dtype=dtype, device=device)
    # [B, H*W, num_anchors, 1]
    box_mask = torch.zeros((B, H * W, num_anchors, 1)).to(dtype=dtype, device=device)
    # [B, H*W, num_anchors, 1]
    box_scale = torch.zeros((B, H * W, num_anchors, 1)).to(dtype=dtype, device=device)

    # [B, H*W, num_anchors, num_classes]
    class_target = torch.zeros((B, H * W, num_anchors, num_classes)).to(dtype=dtype, device=device)
    # [B, H*W, num_anchors, 1]
    # class_target = torch.zeros((B, H * W, num_anchors, 1)).to(dtype=dtype, device=device)
    # [B, H*W, num_anchors, 1]
    class_mask = torch.zeros((B, H * W, num_anchors, 1)).to(dtype=dtype, device=device)

    return iou_target, iou_mask, box_target, box_mask, box_scale, class_target, class_mask


class YOLOv3Loss(nn.Module):
    strides = [32, 16, 8]
    anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    def __init__(self, anchors, num_classes=80, ignore_thresh=0.5,
                 coord_scale=1.0, noobj_scale=1.0, obj_scale=1.0, class_scale=1.0):
        super(YOLOv3Loss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh

        self.noobj_scale = noobj_scale
        self.obj_scale = obj_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale

        self.ref_anchors = anchors

    def make_pred_boxes(self, outputs: Tensor, masked_anchors: Tensor, ref_anchors: Tensor):
        dtype = outputs.dtype
        device = outputs.device

        num_anchors = len(masked_anchors)
        num_ref_anchors = len(ref_anchors)

        B, C, H, W = outputs.shape[:4]
        outputs = outputs.reshape(B, self.num_anchors, 5 + self.num_classes, H, W).permute(0, 1, 3, 4, 2)

        # grid coordinate
        x_shift = torch.broadcast_to(torch.arange(W).reshape(1, 1, W),
                                     (num_anchors, H, W)).to(dtype=dtype, device=device)
        y_shift = torch.broadcast_to(torch.arange(H).reshape(1, H, 1),
                                     (num_anchors, H, W)).to(dtype=dtype, device=device)

        # broadcast anchors to all grids
        w_anchors = torch.broadcast_to(masked_anchors[:, 0].reshape(num_anchors, 1, 1),
                                       (num_anchors, H, W)).to(dtype=dtype, device=device)
        h_anchors = torch.broadcast_to(masked_anchors[:, 1].reshape(num_anchors, 1, 1),
                                       (num_anchors, H, W)).to(dtype=dtype, device=device)

        # b_x = sigmoid(t_x) + c_x
        # b_y = sigmoid(t_y) + c_y
        # b_w = p_w * e^t_w
        # b_h = p_h * e^t_h
        #
        pred_boxes = outputs[..., :4]
        # x/y compress to [0,1]
        pred_boxes[..., :2] = torch.sigmoid(pred_boxes[..., :2])
        pred_boxes[..., 0] += x_shift.expand(B, num_anchors, H, W)
        pred_boxes[..., 1] += y_shift.expand(B, num_anchors, H, W)
        # exp()
        pred_boxes[..., 2:4] = torch.exp(pred_boxes[..., 2:4])
        pred_boxes[..., 2] *= w_anchors.expand(B, num_anchors, H, W)
        pred_boxes[..., 3] *= h_anchors.expand(B, num_anchors, H, W)

        pred_boxes = pred_boxes.reshape(B, num_anchors, H * W, 4)

        # [x_c, y_c, w, h]
        masked_anchors_x1y1 = torch.stack([x_shift, y_shift, w_anchors, h_anchors]).permute(1, 2, 3, 0)
        masked_anchors_x1y1 = masked_anchors_x1y1.reshape(num_anchors, H * W, -1)

        # grid coordinate
        x_shift = torch.broadcast_to(torch.arange(W).reshape(1, 1, W),
                                     (num_ref_anchors, H, W)).to(dtype=dtype, device=device)
        y_shift = torch.broadcast_to(torch.arange(H).reshape(1, H, 1),
                                     (num_ref_anchors, H, W)).to(dtype=dtype, device=device)

        # broadcast anchors to all grids
        ref_w_anchors = torch.broadcast_to(ref_anchors[:, 0].reshape(num_ref_anchors, 1, 1),
                                           (num_ref_anchors, H, W)).to(dtype=dtype, device=device)
        ref_h_anchors = torch.broadcast_to(ref_anchors[:, 1].reshape(num_ref_anchors, 1, 1),
                                           (num_ref_anchors, H, W)).to(dtype=dtype, device=device)
        # [x_c, y_c, w, h]
        ref_anchors_x1y1 = torch.stack([x_shift, y_shift, ref_w_anchors, ref_h_anchors]).permute(1, 2, 3, 0)
        ref_anchors_x1y1 = ref_anchors_x1y1.reshape(num_ref_anchors, H * W, 4)

        return pred_boxes, masked_anchors_x1y1, ref_anchors_x1y1

    def build_targets(self, outputs: Tensor, targets: Tensor,
                      anchor_mask: List, masked_anchors: Tensor, ref_anchors: Tensor):
        num_anchors = len(masked_anchors)
        num_ref_anchors = len(ref_anchors)

        B, C, H, W = outputs.shape[:4]
        assert C == num_anchors * (5 + self.num_classes)

        dtype = outputs.dtype
        device = outputs.device

        # [4] = [x_c, y_c, w, h] 坐标相对于网格大小
        all_pred_boxes, masked_anchors_x1y1, ref_anchors_x1y1 = self.make_pred_boxes(outputs, masked_anchors,
                                                                                     ref_anchors)
        ref_anchors_xcyc = ref_anchors_x1y1.clone()
        ref_anchors_xcyc[..., :2] += 0.5

        # [B, num_max_det, 5] -> [B, num_max_det] -> [B]
        gt_num_objs = (targets.sum(dim=2) > 0).sum(dim=1)

        iou_target, iou_mask, box_target, box_mask, box_scale, class_target, class_mask = \
            build_mask(B, H, W, num_anchors, self.num_classes, dtype, device)
        # 逐图像操作
        for bi in range(B):
            num_obj = gt_num_objs[bi]
            if num_obj == 0:
                # 对于没有标注框的图像，不参与损失计算
                iou_mask[bi, ...] = 0
                continue
            # [num_obj, 4]: [x_c, y_c, w, h]
            gt_boxes = targets[bi][:num_obj][..., 1:]
            gt_boxes[..., 0::2] *= W
            gt_boxes[..., 1::2] *= H
            # [num_obj]
            gt_cls_ids = targets[bi][:num_obj][..., 0]

            # pred_box: [x_c, y_c, w, h]
            pred_boxes = all_pred_boxes[bi][..., :4].reshape(-1, 4)

            # 首先计算预测框与标注框的IoU，忽略正样本的置信度损失计算
            ious = bboxes_iou(pred_boxes, gt_boxes, xyxy=False)
            ious = ious.reshape(num_anchors, -1, num_obj)
            # 计算每个网格中每个预测框计算得到的最大IoU
            max_iou, _ = torch.max(ious, dim=-1, keepdim=True)

            # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
            # 对于正样本(iou大于阈值), 不参与计算
            n_pos = torch.nonzero(max_iou.view(-1) > self.ignore_thresh).numel()
            if n_pos > 0:
                # 如果存在, 那么不参与损失计算
                iou_mask[bi][max_iou >= self.ignore_thresh] = 0

            # 然后计算锚点框与标注框的IoU，保证每个标注框对应一个锚点框
            # 注意：响应的锚点框有可能位于不同的特征层
            ref_anchors_ious = bboxes_iou(ref_anchors_xcyc.reshape(num_ref_anchors * H * W, 4), gt_boxes, xyxy=False)
            ref_anchors_ious = ref_anchors_ious.reshape(num_ref_anchors, H * W, num_obj)

            # iterate over all objects
            # 每个标注框选择一个锚点框进行训练
            for ni in range(num_obj):
                # compute the center of each gt box to determine which cell it falls on
                # assign it to a specific anchor by choosing max IoU
                # 首先计算锚点框的中心点位于哪个网格, 然后选择其中IoU最大的锚点框参与训练
                # 注意：响应锚点框位于不同的特征层中

                # 第t个标注框
                # [4]: [xc, yc, w, h]
                gt_box = gt_boxes[ni]
                # 对应的类别下标
                gt_class = gt_cls_ids[ni]
                # 对应网格下标
                cell_idx_x, cell_idx_y = torch.floor(gt_box[:2])
                # 网格列表下标
                cell_idx = cell_idx_y * W + cell_idx_x
                cell_idx = cell_idx.long()

                # update box_target, box_mask
                # 获取该标注框在对应网格上与所有锚点框的IoU
                # [num_ref_anchors, H*W, num_obj] -> [num_ref_anchors]
                overlaps_in_cell = ref_anchors_ious[:, cell_idx, ni]
                # 选择IoU最大的锚点框下标
                argmax_anchor_idx = torch.argmax(overlaps_in_cell)

                # 验证该锚点框是否作用于该特征层，如果不是，跳过；如果是，设置响应框的conf/box/cls以及不响应框的conf
                if int(argmax_anchor_idx) not in anchor_mask:
                    continue
                masked_anchor_idx = argmax_anchor_idx % 3

                # [H*W, Num_anchors, 4] -> [4]: 获取对应网格中指定锚点框的坐标 [x1, y1, w, h]
                response_anchor_x1y1 = masked_anchors_x1y1[masked_anchor_idx, cell_idx, :]
                target_delta = make_deltas(response_anchor_x1y1.unsqueeze(0), gt_box.unsqueeze(0)).squeeze(0)

                box_target[bi, masked_anchor_idx, cell_idx, :] = target_delta
                box_mask[bi, masked_anchor_idx, cell_idx, :] = 1
                box_scale[bi, masked_anchor_idx, cell_idx, :] = torch.sqrt(2 - gt_box[2] * gt_box[3] / W / H)

                # update cls_target, cls_mask
                # 赋值对应类别下标, 对应掩码设置为1
                class_target[bi, masked_anchor_idx, cell_idx, int(gt_class)] = 1.
                # class_target[bi, cell_idx, masked_anchor_idx, :] = gt_class
                class_mask[bi, masked_anchor_idx, cell_idx, :] = 1

                # update iou target and iou mask
                iou_target[bi, masked_anchor_idx, cell_idx, :] = max_iou[masked_anchor_idx, cell_idx, :]
                iou_mask[bi, masked_anchor_idx, cell_idx, :] = 2

        # [B, num_anchors, H*W, 1] -> [B*num_anchors*H*W]
        iou_target = iou_target.reshape(-1)
        iou_mask = iou_mask.reshape(-1)
        # [B, num_anchors, H*W, 4] -> [B*num_anchors*H*W, 4]
        box_target = box_target.reshape(-1, 4)
        box_mask = box_mask.reshape(-1)
        # [B, num_anchors, H*W, 1] -> [B*num_anchors*H*W]
        box_scale = box_scale.reshape(-1, 1)
        # [B, num_anchors, H*W, num_classes] -> [B*num_anchors*H*W, num_classes]
        class_target = class_target.reshape(-1, self.num_classes)
        # class_target = class_target.reshape(-1).long()
        class_mask = class_mask.reshape(-1)

        return iou_target, iou_mask, box_target, box_mask, box_scale, class_target, class_mask

    def _forward(self, outputs: Tensor, targets: Tensor,
                 anchor_mask: List, masked_anchors: Tensor, ref_anchors: Tensor):
        """
        计算损失需要得到
        1. 标注框坐标和锚点框坐标之间的delta（作为target）
        2. 输出卷积特征生成的预测框delta（作为预测结果）
        """
        iou_target, iou_mask, box_target, box_mask, box_scale, class_target, class_mask = \
            self.build_targets(outputs.detach().clone(), targets, anchor_mask, masked_anchors, ref_anchors)

        num_anchors = len(masked_anchors)

        B, C, H, W = outputs.shape[:4]
        n_ch = 5 + self.num_classes
        assert C == num_anchors * n_ch

        # [B, C, H, W] -> [B, num_anchors, 5+num_classes, H, W] -> [B, num_anchors, H, W, 5+num_classes]
        outputs = outputs.reshape(B, self.num_anchors, 5 + self.num_classes, H, W).permute(0, 1, 3, 4, 2)

        # [B, num_anchors, H, W,  5+num_classes] -> [B, num_anchors*H*W, 5+num_classes]
        outputs = outputs.reshape(B, -1, n_ch)
        # x/y/conf/class compress to [0,1]
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4])

        # [B, num_anchors*H*W, 5+num_classes] -> [B, num_anchors*H*W, 4] -> [B*num_anchors*H*W, 4]
        pred_deltas = outputs[..., :4].reshape(-1, 4)
        # [B, num_anchors*H*W, 5+num_classes] -> [B, num_anchors*H*W] -> [B*num_anchors*H*W]
        pred_confs = outputs[..., 4:5].reshape(-1)
        # [B, num_anchors*H*W, 5+num_classes] -> [B, num_anchors*H*W, num_classes] -> [num_anchors*B*H*W, num_classes]
        pred_probs = outputs[..., 5:].reshape(-1, self.num_classes)

        # --------------------------------------
        # box loss
        pred_deltas = pred_deltas[box_mask > 0]
        box_target = box_target[box_mask > 0]
        assert pred_deltas.shape == box_target.shape
        box_scale = box_scale[box_mask > 0]

        box_xy_loss = F.binary_cross_entropy_with_logits(pred_deltas[..., :2], box_target[..., :2],
                                                         weight=box_scale * box_scale, reduction='sum')
        box_wh_loss = F.mse_loss(pred_deltas[..., 2:4] * box_scale, box_target[..., 2:4] * box_scale,
                                 reduction='sum') / 2.0

        # --------------------------------------
        # iou loss
        obj_pred_confs = pred_confs[iou_mask == 2]
        obj_iou_target = iou_target[iou_mask == 2]
        assert obj_pred_confs.shape == obj_iou_target.shape
        obj_iou_loss = F.binary_cross_entropy_with_logits(obj_pred_confs, obj_iou_target, reduction='sum')

        noobj_pred_confs = pred_confs[iou_mask == 1]
        noobj_iou_target = iou_target[iou_mask == 1]
        assert noobj_pred_confs.shape == noobj_iou_target.shape
        noobj_iou_loss = F.binary_cross_entropy_with_logits(noobj_pred_confs, noobj_iou_target, reduction='sum')

        # --------------------------------------
        # class loss
        # ignore the gradient of noobject's target
        pred_probs = pred_probs[class_mask > 0]
        class_target = class_target[class_mask > 0]
        assert pred_probs.shape == class_target.shape
        class_loss = F.binary_cross_entropy_with_logits(pred_probs, class_target, reduction='sum')

        # calculate the loss, normalized by batch size.
        loss = (box_xy_loss + box_wh_loss) * self.coord_scale + \
               obj_iou_loss * self.obj_scale + noobj_iou_loss * self.noobj_scale + \
               class_loss * self.class_scale
        return loss / B

    def forward(self, outputs, targets):
        loss_list = list()
        for idx, output in enumerate(outputs):
            assert len(output) == len(targets)

            stride = self.strides[idx]
            anchor_mask = self.anchor_masks[idx]

            ref_anchors = self.anchors / stride
            masked_anchors = ref_anchors[anchor_mask]
            loss_list.append(self._forward(output, targets.clone(), anchor_mask, masked_anchors, ref_anchors))

        return torch.stack(loss_list).sum()
