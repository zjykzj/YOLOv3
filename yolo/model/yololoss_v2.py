# -*- coding: utf-8 -*-

"""
@date: 2023/5/4 下午9:21
@file: yololoss.py
@author: zj
@description:
"""
from typing import Dict
import numpy as np

import torch
from torch import nn
from torch import Tensor


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


class YOLOLoss(nn.Module):
    """
    操作流程：
    """

    strides = [32, 16, 8]
    anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    def __init__(self, anchors, n_classes, ignore_thresh=0.7, device=None):
        super(YOLOLoss, self).__init__()
        self.ignore_thresh = ignore_thresh
        self.device = device

        # 预设的锚点框列表，保存了所有的锚点框长宽
        # [9, 2]
        self.anchors = anchors
        # 数据集类别数
        # COCO: 80
        self.n_classes = n_classes

        # 损失函数，work for ???
        self.l2_loss = nn.MSELoss(reduction="sum").to(device)
        self.bce_loss = nn.BCELoss(reduction="sum").to(device)

    def build_target(self, output, pred, layer_no, labels):
        # 图像批量数目
        batchsize = output.shape[0]
        # 特征数据的空间尺寸
        fsize = output.shape[2]
        # 特征层最终输出的通道维度大小
        n_ch = 5 + self.n_classes
        assert output.shape[-1] == n_ch
        # 数值类型以及对应设备
        dtype = output.dtype

        labels = labels.to(dtype)

        # target assignment
        # 创建掩码，作用母鸡？？？
        # [B, n_anchors, F_H, F_W, 4+n_classes]
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes, dtype=dtype).to(self.device)
        # [B, n_anchors, F_H, F_W]
        # 哪个预测框参与计算
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize, dtype=dtype).to(self.device)
        # [B, n_anchors, F_H, F_W, 2]
        # 这个应该是作用于预测框的w/h
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2, dtype=dtype).to(self.device)

        # [B, n_anchors, F_H, F_W, n_ch]
        # n_ch = 4(xywh) + 1(conf) + n_classes
        # 实际用于损失计算的标签
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch, dtype=dtype).to(self.device)

        # xc: [B, K]
        # B: 批量大小
        # K: 真值框数目
        # xc(x_center): 取值在(0, 1)之间
        # xc * fsize：计算实际坐标
        truth_x_all = labels[:, :, 1] * fsize
        # yc: [B, K]
        truth_y_all = labels[:, :, 2] * fsize
        # w: [B, K]
        truth_w_all = labels[:, :, 3] * fsize
        # h: [B, K]
        truth_h_all = labels[:, :, 4] * fsize
        # xc * fsize：计算实际坐标
        # xc / stride：真值标签框坐标缩放指定倍数，匹配当前特征数据空间尺寸
        # print(labels[:, :, 1])
        # print(labels[:, :, 2])
        #
        # 将真值标签框的坐标映射到缩放后的特征数据中
        # # xc: [B, K]
        # truth_x_all = labels[:, :, 1] / self.stride
        # # yc: [B, K]
        # truth_y_all = labels[:, :, 2] / self.stride
        # # w: [B, K]
        # truth_w_all = labels[:, :, 3] / self.stride
        # # h: [B, K]
        # truth_h_all = labels[:, :, 4] / self.stride
        # xc/yc转换成INT16格式i/j，映射到指定网格中
        # truth_i_all = truth_x_all.to(torch.int16).numpy()
        # truth_j_all = truth_y_all.to(torch.int16).numpy()
        truth_i_all = truth_x_all.to(torch.int16)
        truth_j_all = truth_y_all.to(torch.int16)
        # print(truth_x_all)
        # print(truth_y_all)

        # 先创建掩码，包括target、tgt_mask、tgt_scale、obj_mask
        # 然后获取当前特征数据的锚点框列表，创建锚点框参考ref_anchors
        # 最后预处理真值框，包括

        # labels = labels.cpu().data
        # [N, K, 5] -> [N, K] -> [N]
        # 计算有效的真值标签框数目
        # 首先判断是否bbox的xywh有大于0，然后求和计算每幅图像拥有的目标个数
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        # 逐图像处理
        for b in range(batchsize):
            # 获取该幅图像定义的真值标签框个数
            n = int(nlabel[b])
            if n == 0:
                # 如果为0，说明该图像没有符合条件的真值标签框，那么跳过损失计算
                continue
            # 去除空的边界框，获取真正的标注框坐标
            # truth_box = dtype(np.zeros((n, 4)))
            truth_box = torch.zeros((n, 4), dtype=dtype).to(self.device)
            # 重新赋值，在数据类定义中，前n个就是真正的真值边界框
            # 赋值宽和高
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            # 真值标签框的x_center，也就是第i个网格
            truth_i = truth_i_all[b, :n]
            # 真值标签框的y_center，也就是第j个网格
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            # 首先计算真值边界框和锚点框之间的IoU
            # 注意：此时truth_box和ref_anchors的x_center/y_center坐标都是0/0，所以
            # x_center/y_center/w/h可以看成x_top_left/y_top_left/x_bottom_right/y_bottom_right
            # 设置xyxy=True，进行IoU计算
            # ([n, 4], [9, 4]) -> [n, 9]
            # 计算所有锚点框与真值标注框之间IoU的目的是为了找到真值标注框与哪个锚点框最匹配，
            # 如果真值标注框与锚点框的最大IoU超过阈值，并且该锚点框作用于该层特征数据，那么该真值标注框对应网格中使用？？？
            # print(truth_box.dtype, self.ref_anchors.dtype)
            anchor_ious_all = bboxes_iou(truth_box, self.ref_anchors, xyxy=True)
            assert isinstance(anchor_ious_all, torch.Tensor)
            # 找出和真值边界框之间的IoU最大的锚点框的下标
            # [n, 9] -> [n]
            # best_n_all = np.argmax(anchor_ious_all, axis=1)
            best_n_all = torch.argmax(anchor_ious_all, dim=1)
            # 求余操作，3的余数，作用？？？
            # [n] -> [n]
            best_n = best_n_all % 3
            assert isinstance(best_n, torch.Tensor)
            # (best_n_all == self.anch_mask[0]): [n] == 第一个锚点框下标
            # (beat_n_all == self.anch_mask[1]): [n] == 第二个锚点框下标
            # (beat_n_all == self.anch_mask[1]): [n] == 第三个锚点框下标
            # [n] | [n] | [n] = [n]
            # 计算每个真值标注框最匹配的锚点框作用在当前层特征数据的掩码
            best_n_mask = ((best_n_all == self.anch_mask[0]) | (
                    best_n_all == self.anch_mask[1]) | (best_n_all == self.anch_mask[2]))
            assert isinstance(best_n_mask, torch.Tensor)

            # 赋值x_center和y_center
            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            # 计算预测框和真值边界框的IoU
            # ([n_anchors*F_H*F_W, 4], [n, 4]) -> [B*n_anchors*F_H*F_W, n]
            # 预测框坐标：xc/yc是相对于指定网格的比率计算，w/h是相对于特征图空间尺寸的对数运算
            # 真值标注框：xc/yc是相对于输入模型图像的比率计算，w/h是相对于输入模型图像的比率计算，也就是说，参照物是特征图空间尺寸
            pred_ious = bboxes_iou(pred[b].reshape(-1, 4), truth_box, xyxy=False)
            assert isinstance(pred_ious, torch.Tensor)
            # pred[b].view(-1, 4), truth_box, xyxy=False)
            # 计算每个预测框与重叠最大的真值标签框的IoU
            # pred_best_iou: [n_anchors*F_H*F_W]
            # 所有的预测框
            # pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = torch.max(pred_ious, dim=1)[0]
            # 计算掩码，IoU比率要大于忽略阈值。也就是说，如果IoU大于忽略阈值（也就是说预测框坐标与真值标注框坐标非常接近），那么该预测框不参与损失计算
            # pred_best_iou: [n_anchors*F_H*F_W]，取值为true/false
            pred_best_iou = (pred_best_iou > self.ignore_thresh)
            # 改变形状，[n_anchors*F_H*F_W] -> [n_anchors, F_H, F_W]
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            # RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.
            # obj_mask[b] = 1 - pred_best_iou
            # 目标掩码，预测框与真值标注框之间的IoU大于忽略阈值的不参与计算
            # [n_anchors, F_H, F_W]
            obj_mask[b] = ~pred_best_iou
            # obj_mask: 取值为True/False
            # True表示该预测框参与损失计算

            if sum(best_n_mask) == 0:
                # 如果真值边界框和当前层使用的锚点框之间不存在最佳匹配，那么不计算损失
                # 目标：不同层的特征数据负责不同大小的边界框预测
                continue

            # 遍历真值标签框
            for ti in range(best_n.shape[0]):
                # 该真值标签框是否和本层特征使用的锚点框最佳匹配
                # 真值框和锚点框最佳匹配，所以对应网格中制定的预测框可以参与损失计算
                if best_n_mask[ti] == 1:
                    # 如果是，那么计算预测框损失
                    # 获取第ti个真值标签框对应的网格位置
                    i, j = truth_i[ti], truth_j[ti]
                    # 该真值标签框最佳匹配的锚点框
                    # ??? 为什么要这样呢，明明有些锚点框不作用于当前特征层数据
                    # 通过best_n_mask已经确保了该锚点作用于当前特征层数据
                    a = best_n[ti]
                    # print(b, a, j, i, n, ti)
                    # print(truth_i)
                    # print(truth_j)
                    # b: 第b张图像
                    # a: 第a个锚点框，对应第a个预测框
                    # j: 第j列网格
                    # i: 第i行网格
                    # 目标掩码：第[b, a, j, i]个预测框的掩码设置为1，表示参与损失计算
                    # obj_mask: [B, n_anchors, F_H, F_W]
                    #
                    # 掩码设置
                    # 该锚点框对应的网格坐标以及对应列表下标都需要设置为1，表示参与计算
                    #
                    # obj_mask经过了两次设置，
                    # 1. 第一次设置是计算预测框与真值标签框
                    obj_mask[b, a, j, i] = 1
                    # 坐标以及分类掩码：因为采用多标签训练方式，实际损失计算采用二元逻辑回归损失
                    # tgt_mask: [B, n_anchors, F_H, F_W, 4+n_classes]
                    tgt_mask[b, a, j, i, :] = 1
                    # tgt_scale: [B, n_anchors, F_H, F_W, 2]
                    # ???
                    # 这个是加权掩码
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

                    # target: [B, n_anchors, F_H, F_W, n_ch]
                    # 每个真值标注框对应一个预测框
                    #
                    # truth_x_all: [B, K]
                    # 计算第b张图像第ti个真值标签框的xc相对于其所属网格的大小
                    # 设置对应网格中真值标签框xc的大小
                    # 实际大小，应该是真值框坐标减去对应网格下标
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(dtype)
                    # truth_y_all: [B, K]
                    # 计算第b张图像第ti个真值标签框的yc相对于其所属网格的大小
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(dtype)
                    # truth_w_all: [B, K]
                    # truth_w_all[b, ti]: 第b张图像第ti个真值标签框的w。
                    # 注意：w为真值标签框宽与实际输入图像宽的比率　乘以　当前特征数据宽，也就是说，经过了倍数缩放
                    #
                    # best_n: [n]
                    # best_n[ti]: 第ti个真值标签框对应的锚点框下标
                    # self.masked_anchors: [3, 2] 该层特征使用的锚点框列表。注意：其w/h经过了倍数缩放
                    # torch.Tensor(self.masked_anchors)[best_n[ti], 0]: 第ti个真值框匹配的锚点框的w
                    #
                    # log(w_truth / w_anchor):
                    # 计算第b张图像第ti个真值标签框的宽与对应锚点框的宽的比率的对数
                    # 对于长/宽而言，
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 1] + 1e-16)
                    # 该预测框的目标置信度设置为1，说明该预测框有效
                    target[b, a, j, i, 4] = 1
                    # 该b张第ti个真值标签框的类下标参与计算
                    target[b, a, j, i, 5 + labels[b, ti, 0].to(torch.int16)] = 1

        return target, obj_mask, tgt_mask, tgt_scale

    def make_pred(self, outputs):
        B, C, H, W = outputs.shape[:4]
        n_ch = 5 + self.n_classes
        assert C == (self.num_anchors * n_ch)

        dtype = outputs.dtype
        device = outputs.device

        # grid coordinate
        # [W] -> [1, 1, W, 1] -> [B, H, W, num_anchors]
        x_shift = torch.broadcast_to(torch.arange(W).reshape(1, 1, 1, W),
                                     (B, self.num_anchors, H, W)).to(dtype=dtype, device=device)
        # [H] -> [1, H, 1, 1] -> [B, H, W, num_anchors]
        y_shift = torch.broadcast_to(torch.arange(H).reshape(1, 1, H, 1),
                                     (B, self.num_anchors, H, W)).to(dtype=dtype, device=device)

        masked_anchors = torch.Tensor(self.masked_anchors)

        # broadcast anchors to all grids
        # [num_anchors] -> [1, 1, 1, num_anchors] -> [B, H, W, num_anchors]
        w_anchors = torch.broadcast_to(masked_anchors[:, 0].reshape(1, self.num_anchors, 1, 1),
                                       (B, self.num_anchors, H, W)).to(dtype=dtype, device=device)
        h_anchors = torch.broadcast_to(masked_anchors[:, 1].reshape(1, self.num_anchors, 1, 1),
                                       (B, self.num_anchors, H, W)).to(dtype=dtype, device=device)

        # Reshape
        # [B, num_anchors * (4+1+num_classes), H, W] ->
        # [B, H, W, num_anchors * (4+1+num_classes)] ->
        # [B, H, W, num_anchors, 4+1+num_classes]
        # outputs = outputs.permute(0, 2, 3, 1).reshape(B, H, W, self.num_anchors, n_ch)
        outputs = outputs.reshape(B, self.n_anchors, n_ch, H, W).permute(0, 1, 3, 4, 2)

        # logistic activation
        # xy + obj_pred + cls_pred
        # outputs[..., np.r_[:2, 4:5]] = torch.sigmoid(outputs[..., np.r_[:2, 4:5]])
        outputs[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(outputs[..., np.r_[:2, 4:n_ch]])

        pred = outputs.clone()
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
        pred[..., 2] = torch.exp(outputs[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(outputs[..., 3]) * h_anchors

        return pred[..., :4], outputs

    def forward(self, outputs, targets):
        assert isinstance(outputs, list)
        # assert isinstance(targets, dict)
        assert isinstance(targets, Tensor)
        # [B, K(每幅图片拥有的真值标签框数目), 5(cls_id + bbox)]
        # print(labels)
        # print("img_info:", target['img_info'])

        loss_list = []
        # for output_dict in outputs:
        for idx, output in enumerate(outputs):
            # 逐层计算损失
            # assert isinstance(output_dict, dict)
            assert isinstance(output, Tensor)
            # 获取当前特征层下标
            # layer_no = output_dict['layer_no']
            layer_no = idx
            # 获取特征层数据 [B, n_anchors * (xywh+conf+n_classes), F_H, F_W]
            # output = output_dict['output'].to(self.device)
            output = output.to(self.device)

            # 获取当前YOLO层特征数据使用的锚点框
            # [3, 3] -> [3]
            self.anch_mask = self.anchor_masks[layer_no]
            # 当前YOLO层特征数据使用的锚点框个数，默认为3
            self.n_anchors = len(self.anch_mask)

            # 数值类型以及对应设备
            dtype = output.dtype
            # 第N个YOLO层使用的步长，也就是输入图像大小和使用的特征数据之间的缩放比率
            self.stride = self.strides[layer_no]
            # 按比例缩放锚点框长／宽
            # [9, 2]
            self.all_anchors_grid = [(w / self.stride, h / self.stride) for w, h in self.anchors]
            # 采集指定YOLO使用的锚点
            # [3, 2]
            self.masked_anchors = [self.all_anchors_grid[i] for i in self.anch_mask]
            self.num_anchors = len(self.masked_anchors)
            # [9, 4]
            # self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
            self.ref_anchors = torch.zeros((len(self.all_anchors_grid), 4), dtype=dtype).to(self.device)
            # print("ref_anchors:", self.ref_anchors.dtype)
            # 赋值，锚点框宽／高
            self.ref_anchors[:, 2:] = torch.from_numpy(np.array(self.all_anchors_grid)).to(dtype=dtype,
                                                                                           device=self.device)
            # print("ref_anchors 2:", self.ref_anchors.dtype)

            # 获取预测边界框　[B, n_anchors, F_H, F_W, 4(xywh)]
            # pred = output_dict['pred'].cpu().data
            # pred = output_dict['pred'].clone().to(self.device)
            pred, output = self.make_pred(output.clone())

            # labels = targets['padded_labels'].clone().to(self.device)
            labels = targets.clone().to(self.device)
            target, obj_mask, tgt_mask, tgt_scale = self.build_target(output, pred, layer_no, labels)
            # target = target.to(self.device)
            # obj_mask = obj_mask.to(self.device)
            # tgt_mask = tgt_mask.to(self.device)
            # tgt_scale = tgt_scale.to(self.device)

            # loss calculation
            # 掩码的目的是为了屏蔽不符合要求的预测框
            # 首先过滤掉不符合条件的置信度
            output[..., 4] *= obj_mask
            # 然后过滤掉不符合条件的坐标以及类别
            # 4 + 1 + n_classes
            n_ch = output.shape[-1]
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            # 针对w/h，某些预测框还需要乘以？？？
            output[..., 2:4] *= tgt_scale

            # 掩码分两部分：
            # 一部分是预测框数据，另一部分是对应标签
            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            # 加权二值交叉熵损失
            bceloss = nn.BCELoss(weight=tgt_scale * tgt_scale, reduction="sum").to(self.device)  # weighted BCEloss
            # 计算预测框xc/yc的损失
            loss_xy = bceloss(output[..., :2], target[..., :2])
            # 计算预测框w/h的损失
            loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4]) / 2
            # 计算目标置信度损失
            loss_obj = self.bce_loss(output[..., 4], target[..., 4])
            # 计算各个类别的分类概率损失
            loss_cls = self.bce_loss(output[..., 5:], target[..., 5:])
            # 计算统一损失
            loss_l2 = self.l2_loss(output, target)

            # 最终损失 = xc/yc损失 + w/h损失 + obj损失 + 分类损失
            loss = loss_xy + loss_wh + loss_obj + loss_cls

            # loss_xy + loss_wh + loss_obj + loss_cls + loss_xy + loss_wh + loss_obj + loss_cls + loss_l2 =
            # 2*loss_xy + 2*loss_wh + 2*loss_obj + 2*loss_cls + loss_l2
            # 因为loss_wh = self.l2_loss(...) / 2, 所以上式等同于
            # 2*bceloss + self.l2_loss + 2*self.bce_loss + 2*self.bce_loss + self.l2_loss

            # loss_list.append([loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2])
            # loss_list.append(loss + loss_xy + loss_wh + loss_obj + loss_cls + loss_l2)
            loss_list.append(loss)

        return sum(loss_list)