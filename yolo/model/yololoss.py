# -*- coding: utf-8 -*-

"""
@date: 2023/5/4 下午9:21
@file: yololoss.py
@author: zj
@description:

RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.
Many models use a sigmoid layer right before the binary cross entropy layer.
In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits
or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are
safe to autocast.
"""
from typing import List

import numpy as np

import torch
from torch import Tensor
from torch import nn


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    # bboxes_a: [N_a, 4]
    # bboxes_b: [N_b, 4]
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        # xyxy: x_top_left, y_top_left, x_bottom_right, y_bottom_right
        # 计算交集矩形的左上角坐标
        # torch.max([N_a, 1, 2], [N_b, 2]) -> [N_a, N_b, 2]
        # torch.max: 双重循环
        #   第一重循环 for i in range(N_a)，遍历boxes_a, 获取边界框i，大小为[2]
        #       第二重循环　for j in range(N_b)，遍历bboxes_b，获取边界框j，大小为[2]
        #           分别比较i[0]/j[0]和i[1]/j[1]，获取得到最大的x/y
        #   遍历完成后，获取得到[N_a, N_b, 2]
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        # 计算交集矩形的右下角坐标
        # torch.min([N_a, 1, 2], [N_b, 2]) -> [N_a, N_b, 2]
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # 计算bboxes_a的面积
        # x_bottom_right/y_bottom_right - x_top_left/y_top_left = w/h
        # prod([N, w/h], 1) = [N], 每个item表示边界框的面积w*h
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # x_center/y_center -> x_top_left, y_top_left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        # x_center/y_center -> x_bottom_right/y_bottom_right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # prod([N_a, w/h], 1) = [N_a], 每个item表示边界框的面积w*h
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    # 判断符合条件的结果：x_top_left/y_top_left < x_bottom_right/y_bottom_right
    # [N_a, N_b, 2] < [N_a, N_b, 2] = [N_a, N_b, 2]
    # prod([N_a, N_b, 2], 2) = [N_a, N_b], 数值为1/0
    en = (tl < br).type(tl.type()).prod(dim=2)
    # 首先计算交集w/h: [N_a, N_b, 2] - [N_a, N_b, 2] = [N_a, N_b, 2]
    # 然后计算交集面积：prod([N_a, N_b, 2], 2) = [N_a, N_b]
    # 然后去除不符合条件的交集面积
    # [N_a, N_b] * [N_a, N_b](数值为1/0) = [N_a, N_b]
    # 大小为[N_a, N_b]，表示bboxes_a的每个边界框与bboxes_b的每个边界框之间的IoU
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())

    # 计算IoU
    # 首先计算所有面积
    # area_a[:, None] + area_b - area_i =
    # [N_a, 1] + [N_b] - [N_a, N_b] = [N_a, N_b]
    # 然后交集面积除以所有面积，计算IoU
    # [N_a, N_b] / [N_a, N_b] = [N_a, N_b]
    return area_i / (area_a[:, None] + area_b - area_i)


class YOLOv3Loss(nn.Module):
    strides = [32, 16, 8]
    anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    def __init__(self, anchors, n_classes, ignore_thresh=0.7, device=None):
        super(YOLOv3Loss, self).__init__()
        self.anchors = anchors
        self.n_classes = n_classes
        self.ignore_thresh = ignore_thresh
        self.device = device

        self.l2_loss = nn.MSELoss(reduction="sum").to(device)
        # self.bce_loss = nn.BCELoss(reduction="sum").to(device)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum").to(device)

    def _make_pred(self, outputs: Tensor):
        B, C, H, W = outputs.shape[:4]
        n_ch = 5 + self.n_classes
        assert C == (self.num_anchors * n_ch)

        dtype = outputs.dtype
        device = outputs.device

        # grid coordinate
        x_shift = torch.broadcast_to(torch.arange(W).reshape(1, 1, 1, W),
                                     (B, self.num_anchors, H, W)).to(dtype=dtype, device=device)
        y_shift = torch.broadcast_to(torch.arange(H).reshape(1, 1, H, 1),
                                     (B, self.num_anchors, H, W)).to(dtype=dtype, device=device)

        masked_anchors = torch.Tensor(self.masked_anchors)

        # broadcast anchors to all grids
        w_anchors = torch.broadcast_to(masked_anchors[:, 0].reshape(1, self.num_anchors, 1, 1),
                                       (B, self.num_anchors, H, W)).to(dtype=dtype, device=device)
        h_anchors = torch.broadcast_to(masked_anchors[:, 1].reshape(1, self.num_anchors, 1, 1),
                                       (B, self.num_anchors, H, W)).to(dtype=dtype, device=device)
        # Reshape
        outputs = outputs.reshape(B, self.num_anchors, n_ch, H, W).permute(0, 1, 3, 4, 2)
        preds = outputs[..., :4]

        # logistic activation
        preds[..., :2] = torch.sigmoid(preds[..., :2])

        # b_x = sigmoid(t_x) + c_x
        # b_y = sigmoid(t_y) + c_y
        # b_w = exp(t_w) * p_w
        # b_h = exp(t_h) * p_h
        #
        preds[..., 0] += x_shift
        preds[..., 1] += y_shift
        preds[..., 2] = torch.exp(preds[..., 2]) * w_anchors
        preds[..., 3] = torch.exp(preds[..., 3]) * h_anchors

        return preds

    def _build_target(self, pred: Tensor, labels: Tensor):
        B, _, H, W, _ = pred.shape
        n_ch = 5 + self.n_classes

        # target assignment
        tgt_mask = torch.zeros(B, self.num_anchors, H, W, 4 + self.n_classes).to(dtype=self.dtype, device=self.device)
        obj_mask = torch.ones(B, self.num_anchors, H, W).to(dtype=self.dtype, device=self.device)
        tgt_scale = torch.zeros(B, self.num_anchors, H, W, 2).to(dtype=self.dtype, device=self.device)

        target = torch.zeros(B, self.num_anchors, H, W, n_ch).to(dtype=self.dtype, device=self.device)

        truth_x_all = labels[:, :, 1] * W
        truth_y_all = labels[:, :, 2] * H
        truth_w_all = labels[:, :, 3] * W
        truth_h_all = labels[:, :, 4] * H
        truth_i_all = truth_x_all.to(dtype=torch.int16)
        truth_j_all = truth_y_all.to(dtype=torch.int16)

        num_labels = (labels.sum(dim=2) > 0).sum(dim=1)

        for bi in range(B):
            n = int(num_labels[bi])
            if n == 0:
                continue

            truth_box = torch.zeros((n, 4)).to(dtype=self.dtype, device=self.device)
            truth_box[:n, 2] = truth_w_all[bi, :n]
            truth_box[:n, 3] = truth_h_all[bi, :n]
            truth_i = truth_i_all[bi, :n]
            truth_j = truth_j_all[bi, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box, self.ref_anchors, xyxy=True)
            assert isinstance(anchor_ious_all, torch.Tensor)
            # [n, 9] -> [n]
            best_n_all = torch.argmax(anchor_ious_all, dim=1)
            # [n] -> [n]
            best_n = best_n_all % 3
            assert isinstance(best_n, torch.Tensor)
            # (best_n_all == self.anch_mask[0]): [n] == 第一个锚点框下标
            # (beat_n_all == self.anch_mask[1]): [n] == 第二个锚点框下标
            # (beat_n_all == self.anch_mask[1]): [n] == 第三个锚点框下标
            # [n] | [n] | [n] = [n]
            # 计算每个真值标注框最匹配的锚点框作用在当前层特征数据的掩码
            best_n_mask = ((best_n_all == self.anch_mask[0]) |
                           (best_n_all == self.anch_mask[1]) | (best_n_all == self.anch_mask[2]))
            assert isinstance(best_n_mask, torch.Tensor)

            truth_box[:n, 0] = truth_x_all[bi, :n]
            truth_box[:n, 1] = truth_y_all[bi, :n]

            # 计算预测框和真值边界框的IoU
            # ([num_anchors*F_H*F_W, 4], [n, 4]) -> [B*num_anchors*F_H*F_W, n]
            # 预测框坐标：xc/yc是相对于指定网格的比率计算，w/h是相对于特征图空间尺寸的对数运算
            # 真值标注框：xc/yc是相对于输入模型图像的比率计算，w/h是相对于输入模型图像的比率计算，也就是说，参照物是特征图空间尺寸
            pred_ious = bboxes_iou(pred[bi].reshape(-1, 4), truth_box, xyxy=False)
            assert isinstance(pred_ious, torch.Tensor)

            pred_best_iou = torch.max(pred_ious, dim=1)[0]
            pred_best_iou = (pred_best_iou > self.ignore_thresh)
            pred_best_iou = pred_best_iou.view(pred[bi].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[bi] = ~pred_best_iou

            if sum(best_n_mask) == 0:
                continue

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    a = best_n[ti]
                    i, j = truth_i[ti], truth_j[ti]

                    obj_mask[bi, a, j, i] = 1
                    tgt_mask[bi, a, j, i, :] = 1
                    tgt_scale[bi, a, j, i, :] = torch.sqrt(2 - truth_w_all[bi, ti] * truth_h_all[bi, ti] / H / W)

                    target[bi, a, j, i, 0] = truth_x_all[bi, ti] - \
                                             truth_x_all[bi, ti].to(dtype=torch.int16).to(dtype=self.dtype)
                    target[bi, a, j, i, 1] = truth_y_all[bi, ti] - \
                                             truth_y_all[bi, ti].to(dtype=torch.int16).to(dtype=self.dtype)

                    target[bi, a, j, i, 2] = torch.log(
                        truth_w_all[bi, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 0] + 1e-16)
                    target[bi, a, j, i, 3] = torch.log(
                        truth_h_all[bi, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 1] + 1e-16)

                    target[bi, a, j, i, 4] = 1
                    target[bi, a, j, i, 5 + labels[bi, ti, 0].to(dtype=torch.int16)] = 1

        return target, obj_mask, tgt_mask, tgt_scale

    def _forward(self, layer_no: int, output: Tensor, labels: Tensor):
        assert isinstance(output, Tensor)
        self.dtype = output.dtype

        self.anch_mask = self.anchor_masks[layer_no]
        self.num_anchors = len(self.anch_mask)
        self.stride = self.strides[layer_no]

        # [9, 2] 按比例缩放锚点框长／宽
        self.all_anchors_grid = [(w / self.stride, h / self.stride) for w, h in self.anchors]
        # [3, 2] 采集指定YOLO使用的锚点
        self.masked_anchors = [self.all_anchors_grid[i] for i in self.anch_mask]
        # [9, 4]
        self.ref_anchors = torch.zeros((len(self.all_anchors_grid), 4)).to(dtype=self.dtype, device=self.device)
        # 赋值，锚点框宽／高
        self.ref_anchors[:, 2:] = torch.from_numpy(np.array(self.all_anchors_grid)) \
            .to(dtype=self.dtype, device=self.device)

        # ------------------------------------------------- target
        pred = self._make_pred(output.clone().detach())
        target, obj_mask, tgt_mask, tgt_scale = self._build_target(pred, labels)

        # ------------------------------------------------- output
        B, C, H, W = output.shape[:4]
        n_ch = 5 + self.n_classes
        output = output.reshape(B, self.num_anchors, n_ch, H, W).permute(0, 1, 3, 4, 2)
        # output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

        # ------------------------------------------------- loss
        # loss calculation
        output[..., 4] *= obj_mask
        output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        output[..., 2:4] *= tgt_scale

        target[..., 4] *= obj_mask
        target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        target[..., 2:4] *= tgt_scale

        # bceloss = nn.BCELoss(weight=tgt_scale * tgt_scale, reduction="sum").to(self.device)  # weighted BCEloss
        bceloss = nn.BCEWithLogitsLoss(weight=tgt_scale * tgt_scale, reduction="sum") \
            .to(device=self.device)  # weighted BCEloss
        loss_xy = bceloss(output[..., :2], target[..., :2])
        loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4]) / 2
        loss_obj = self.bce_loss(output[..., 4], target[..., 4])

        mask_cls = tgt_mask[..., 4].reshape(-1)
        output_cls = output[..., 5:].reshape(-1, self.n_classes)[mask_cls > 0]
        target_cls = target[..., 5:].reshape(-1, self.n_classes)[mask_cls > 0]
        loss_cls = self.bce_loss(output_cls, target_cls)

        loss = loss_xy + loss_wh + loss_obj + loss_cls
        return loss

    def forward(self, outputs: List[Tensor], targets: Tensor):
        assert isinstance(outputs, list)
        assert isinstance(targets, Tensor)

        loss_list = []
        for layer_no, output in enumerate(outputs):
            output = output.to(self.device)
            labels = targets.clone().detach().to(self.device)

            loss_list.append(self._forward(layer_no, output, labels))

        return sum(loss_list)
