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
    # [B, H*W, num_anchors, 2]
    box_scale = torch.zeros((B, H * W, num_anchors, 2)).to(dtype=dtype, device=device)

    # [B, H*W, num_anchors, num_classes]
    class_target = torch.zeros((B, H * W, num_anchors, num_classes)).to(dtype=dtype, device=device)
    # [B, H*W, num_anchors, 1]
    class_mask = torch.zeros((B, H * W, num_anchors, 1)).to(dtype=dtype, device=device)

    return iou_target, iou_mask, box_target, box_mask, box_scale, class_target, class_mask


class YOLOv3Loss(nn.Module):
    strides = [32, 16, 8]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    def __init__(self, anchors, num_classes=80, ignore_thresh=0.70):
        super(YOLOv3Loss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh

        self.ref_anchors = anchors

    def make_pred_boxes(self, outputs, masked_anchors, ref_anchors):
        dtype = outputs.dtype
        device = outputs.device

        num_anchors = len(masked_anchors)

        B, C, H, W = outputs.shape[:4]
        # [B, num_anchors * (5+num_classes), H, W] ->
        # [B, num_anchors, 5+num_classes, H, W] ->
        # [B, num_anchors, H, W, 5+num_classes]
        outputs = outputs.reshape(B, num_anchors, 5 + self.num_classes, H, W) \
            .permute(0, 1, 3, 4, 2)

        # grid coordinate
        # [F_size] -> [num_anchors, H, W]
        x_shift = torch.broadcast_to(torch.arange(W),
                                     (num_anchors, H, W)).to(dtype=dtype, device=device)
        # [F_size] -> [f_size, 1] -> [num_anchors, H, W]
        y_shift = torch.broadcast_to(torch.arange(H).reshape(H, 1),
                                     (num_anchors, H, W)).to(dtype=dtype, device=device)

        # broadcast anchors to all grids
        # [num_anchors] -> [num_anchors, 1, 1] -> [num_anchors, H, W]
        w_anchors = torch.broadcast_to(masked_anchors[:, 0].reshape(num_anchors, 1, 1),
                                       [num_anchors, H, W]).to(dtype=dtype, device=device)
        h_anchors = torch.broadcast_to(masked_anchors[:, 1].reshape(num_anchors, 1, 1),
                                       [num_anchors, H, W]).to(dtype=dtype, device=device)

        # b_x = sigmoid(t_x) + c_x
        # b_y = sigmoid(t_y) + c_y
        # b_w = p_w * e^t_w
        # b_h = p_h * e^t_h
        #
        # [B, num_anchors, H, W, 4]
        pred_boxes = outputs[..., :4]
        # x/y compress to [0,1]
        pred_boxes[..., :2] = torch.sigmoid(pred_boxes[..., :2])
        pred_boxes[..., 0] += x_shift.expand(B, num_anchors, H, W)
        pred_boxes[..., 1] += y_shift.expand(B, num_anchors, H, W)
        # exp()
        pred_boxes[..., 2:4] = torch.exp(pred_boxes[..., 2:4])
        pred_boxes[..., 2] *= w_anchors.expand(B, num_anchors, H, W)
        pred_boxes[..., 3] *= h_anchors.expand(B, num_anchors, H, W)

        # [B, num_anchors, H, W, 4] -> [B, H, W, num_anchors, 4] -> [B, H*W, num_anchors, 4]
        pred_boxes = pred_boxes.permute(0, 2, 3, 1, 4).reshape(B, H * W, num_anchors, 4)

        # [4, num_anchors, H, W] -> [H, W, num_anchors, 4]
        # [x_c, y_c, w, h]
        masked_anchors_x1y1 = torch.stack([x_shift, y_shift, w_anchors, h_anchors]).permute(2, 3, 1, 0)
        # [H, W, num_anchors, 4] -> [H*W, num_anchors, 4]
        masked_anchors_x1y1 = masked_anchors_x1y1.reshape(H * W, num_anchors, -1)

        # grid coordinate
        # [F_size] -> [num_anchors, H, W]
        x_shift = torch.broadcast_to(torch.arange(W),
                                     (len(ref_anchors), H, W)).to(dtype=dtype, device=device)
        # [F_size] -> [f_size, 1] -> [num_anchors, H, W]
        y_shift = torch.broadcast_to(torch.arange(H).reshape(H, 1),
                                     (len(ref_anchors), H, W)).to(dtype=dtype, device=device)

        # broadcast anchors to all grids
        # [num_anchors*3] -> [num_anchors*3, 1, 1] -> [num_anchors*3, H, W]
        ref_w_anchors = torch.broadcast_to(ref_anchors[:, 0].reshape(len(ref_anchors), 1, 1),
                                           [len(ref_anchors), H, W]).to(dtype=dtype, device=device)
        ref_h_anchors = torch.broadcast_to(ref_anchors[:, 1].reshape(len(ref_anchors), 1, 1),
                                           [len(ref_anchors), H, W]).to(dtype=dtype, device=device)
        # [4, num_anchors*3, H, W] -> [H, W, num_anchors*3, 4]
        # [x_c, y_c, w, h]
        ref_anchors_x1y1 = torch.stack([x_shift, y_shift, ref_w_anchors, ref_h_anchors]).permute(2, 3, 1, 0)
        # [H, W, num_anchors*3, 4] -> [H*W, num_anchors*3, 4]
        ref_anchors_x1y1 = ref_anchors_x1y1.reshape(H * W, num_anchors, -1)

        return pred_boxes, masked_anchors_x1y1, ref_anchors_x1y1

    def build_targets(self, outputs: Tensor, targets: Tensor,
                      masked_anchors: Tensor, anchor_mask: List, ref_anchors: Tensor):
        num_anchors = len(masked_anchors)

        B, C, H, W = outputs.shape[:4]
        assert C == num_anchors * (5 + self.num_classes)

        dtype = outputs.dtype
        device = outputs.device

        # all_pred_boxes: [B, H*W, num_anchors, 4]
        # masked_anchors_x1y1: [H*W, num_anchors, 4]
        # ref_anchors_x1y1: [H*W, num_anchors*3, 4]
        # [4] = [x_c, y_c, w, h] 坐标相对于网格大小
        all_pred_boxes, masked_anchors_x1y1, ref_anchors_x1y1 = \
            self.make_pred_boxes(outputs, masked_anchors, ref_anchors)
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
            # [num_obj, 4]
            # [4]: [x_c, y_c, w, h]
            gt_boxes = targets[bi][:num_obj][..., :4]
            # [num_obj]
            gt_cls_ids = targets[bi][:num_obj][..., 4]

            # 放大到网格大小
            gt_boxes[..., 0::2] *= W
            gt_boxes[..., 1::2] *= H

            # [H*W, num_anchors, 4] -> [H*W*num_anchors, 4]
            # pred_box: [x_c, y_c, w, h]
            pred_boxes = all_pred_boxes[bi][..., :4].reshape(-1, 4)

            # 首先计算预测框与标注框的IoU，忽略正样本的置信度损失计算
            # ious: [H*W*num_anchors, num_obj]
            ious = bboxes_iou(pred_boxes, gt_boxes, xyxy=False)
            # [H*W*num_anchors, num_obj] -> [H*W, num_anchors, num_obj]
            ious = ious.reshape(-1, num_anchors, num_obj)
            # 计算每个网格中每个预测框计算得到的最大IoU
            # shape: (H * W, num_anchors, 1)
            max_iou, _ = torch.max(ious, dim=-1, keepdim=True)

            # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
            # 对于正样本(iou大于阈值), 不参与计算
            # [H*W, Num_anchors, 1] -> [H*W*Num_anchors] -> [n_pos]
            n_pos = torch.nonzero(max_iou.view(-1) > self.ignore_thresh).numel()
            if n_pos > 0:
                # 如果存在, 那么不参与损失计算
                iou_mask[bi][max_iou >= self.ignore_thresh] = 0

            # 然后计算锚点框与标注框的IoU，保证每个标注框对应一个锚点框
            # 注意：响应的锚点框有可能位于不同的特征层
            # overlaps: [H*W*(num_anchors*3), num_obj]
            ref_anchors_ious = bboxes_iou(ref_anchors_xcyc.reshape(-1, 4), gt_boxes, xyxy=False)
            # [H*W*(num_anchors*3), num_obj] -> [H*W, num_anchors*3, num_obj]
            ref_anchors_ious = ref_anchors_ious.reshape(-1, len(ref_anchors), num_obj)

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
                # [H*W, num_anchors*3, num_obj] -> [num_anchors*3]
                # print(cell_idx, ni, overlaps.shape, cell_idx_x, cell_idx_y, H, W)
                overlaps_in_cell = ref_anchors_ious[cell_idx, :, ni]
                # 选择IoU最大的锚点框下标
                argmax_anchor_idx = torch.argmax(overlaps_in_cell)

                # 验证该锚点框是否作用于该特征层，如果不是，跳过；如果是，设置响应框的conf/box/cls以及不响应框的conf
                if argmax_anchor_idx not in anchor_mask:
                    continue

                # [H*W, Num_anchors, 4] -> [4]
                # 获取对应网格中指定锚点框的坐标 [x1, y1, w, h]
                response_anchor_x1y1 = masked_anchors_x1y1[cell_idx, argmax_anchor_idx % 3, :]
                target_delta = make_deltas(response_anchor_x1y1.unsqueeze(0), gt_box.unsqueeze(0)).squeeze(0)

                box_target[bi, cell_idx, argmax_anchor_idx % 3, :] = target_delta
                box_mask[bi, cell_idx, argmax_anchor_idx % 3, :] = 1
                box_scale[bi, cell_idx, argmax_anchor_idx % 3, :] = torch.sqrt(2 - gt_box[2] * gt_box[3] / W / H)

                # update cls_target, cls_mask
                # 赋值对应类别下标, 对应掩码设置为1
                class_target[bi, cell_idx, argmax_anchor_idx % 3, int(gt_class)] = 1.
                class_mask[bi, cell_idx, argmax_anchor_idx % 3, :] = 1

                # update iou target and iou mask
                iou_target[bi, cell_idx, argmax_anchor_idx % 3, :] = max_iou[cell_idx, argmax_anchor_idx % 3, :]
                iou_mask[bi, cell_idx, argmax_anchor_idx % 3, :] = 1

        # [B, H*W, num_anchors, 1] -> [B*H*W*num_anchors]
        iou_target = iou_target.reshape(-1, 1)
        iou_mask = iou_mask.reshape(-1, 1)
        # [B, H*W, num_anchors, 4] -> [B*H*W*num_anchors, 4]
        box_target = box_target.reshape(-1, 4)
        box_mask = box_mask.reshape(-1, 1)
        # [B, H*W, num_anchors, 2] -> [B*H*W*num_anchors, 2]
        box_scale = box_scale.reshape(-1, 2)
        # [B, H*W, num_anchors, num_classes] -> [B*H*W*num_anchors, num_classes]
        class_target = class_target.reshape(-1, self.num_classes)
        class_mask = class_mask.reshape(-1, 1)

        return iou_target, iou_mask, box_target, box_mask, box_scale, class_target, class_mask

    def _forward(self, outputs, targets, masked_anchors, anchor_mask, ref_anchors):
        """
        计算损失需要得到
        1. 标注框坐标和锚点框坐标之间的delta（作为target）
        2. 输出卷积特征生成的预测框delta（作为预测结果）
        """
        iou_target, iou_mask, box_target, box_mask, box_scale, class_target, class_mask = \
            self.build_targets(outputs.detach().clone(), targets, masked_anchors, anchor_mask, ref_anchors)

        num_anchors = len(masked_anchors)
        n_ch = outputs.shape[-1]

        B, _, H, W = outputs.shape[:4]
        # [B, C, H, W] -> [B, num_anchors, 5+num_classes, H, W] -> [B, H, W, num_anchors, 5+num_classes]
        outputs = outputs.reshape(B, num_anchors, 5 + self.num_classes, H, W).permute(0, 3, 4, 1, 2)
        # [B, H, W, num_anchors, 5+num_classes] -> [B, H*W*num_anchors, 5+num_classes]
        outputs = outputs.reshape(B, -1, 5 + self.num_classes)
        # x/y/conf/class compress to [0,1]
        outputs[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(outputs[..., np.r_[:2, 4:n_ch]])
        # outputs[..., :2] = torch.sigmoid(outputs[..., :2])
        # outputs[..., 4:] = torch.sigmoid(outputs[..., 4:])
        # exp()
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4])

        # [B, H*W*num_anchors, 5+num_classes] -> [B, H*W*num_anchors, 4] -> [B*H*W*num_anchors, 4]
        pred_deltas = outputs[..., :4].reshape(-1, 4)
        # print(torch.max(pred_deltas[..., :2]), torch.min(pred_deltas[..., :2]))
        # print(torch.max(pred_deltas[..., 2:4]), torch.min(pred_deltas[..., 2:4]))
        # [B, H*W*num_anchors, 5+num_classes] -> [B, H*W*num_anchors, 1] -> [B*H*W*num_anchors, 1]
        pred_confs = outputs[..., 4:5].reshape(-1, 1)
        # print(f"pred_confs: {pred_confs.shape} {torch.max(pred_confs)} {torch.min(pred_confs)}")
        # print(torch.max(pred_confs), torch.min(pred_confs))
        # [B, H*W*num_anchors, 5+num_classes] -> [B, H*W*num_anchors, num_classes] -> [B*H*W*num_anchors, num_classes]
        pred_probs = outputs[..., 5:].reshape(-1, self.num_classes)
        # print(torch.max(pred_probs), torch.min(pred_probs))

        # --------------------------------------
        # box loss
        pred_deltas = pred_deltas * box_mask
        box_target = box_target * box_mask
        box_scale = box_scale * box_mask

        box_xy_loss = F.binary_cross_entropy(pred_deltas[..., :2], box_target[..., :2],
                                             weight=box_scale * box_scale, reduction='sum')
        # print(box_xy_loss)

        pred_deltas_wh = pred_deltas[..., 2:] * box_scale
        box_target_wh = box_target[..., 2:] * box_scale
        box_wh_loss = F.mse_loss(pred_deltas_wh, box_target_wh, reduction='sum') / 2.
        # print(box_wh_loss)

        # --------------------------------------
        # iou loss
        # print(f"pred_confs: {torch.max(pred_confs)} {torch.min(pred_confs)}")
        # print(f"iou_target: {torch.max(iou_target)} {torch.min(iou_target)}")
        # print(f"iou_mask: {torch.max(iou_mask)} {torch.min(iou_mask)}")
        pred_confs = pred_confs * iou_mask
        iou_target = iou_target * iou_mask
        # print(torch.max(pred_confs), torch.min(pred_confs))
        # print(torch.max(iou_target), torch.min(iou_target))
        iou_loss = F.binary_cross_entropy(pred_confs.reshape(-1), iou_target.reshape(-1), reduction='sum')
        # print(iou_loss)

        # --------------------------------------
        # class loss
        # ignore the gradient of noobject's target
        class_mask = class_mask > 0
        # print(class_mask)
        pred_probs = pred_probs * class_mask
        class_target = class_target * class_mask

        # calculate the loss, normalized by batch size.
        class_loss = F.binary_cross_entropy(pred_probs, class_target, reduction='sum')

        # print(f"box_loss: {box_loss} iou_loss: {iou_loss} class_loss: {class_loss}")
        loss = box_xy_loss + box_wh_loss + iou_loss + class_loss
        return loss

    def forward(self, outputs, targets):
        loss = 0.
        for output in outputs:
            assert len(output) == len(targets)
            for i, (stride, anchor_mask) in enumerate(zip(self.strides, self.anchor_mask)):
                ref_anchors = self.anchors / stride
                masked_anchors = ref_anchors[anchor_mask]
                loss += self._forward(output, targets.clone(), masked_anchors, anchor_mask, ref_anchors)

        return loss
