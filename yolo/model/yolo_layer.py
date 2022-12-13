import torch
import torch.nn as nn
import numpy as np
from yolo.utils.utils import bboxes_iou


class YOLOLayer(nn.Module):
    """
    YOLO网络的核心，对于输入的特征数据，如何利用锚点框执行预测操作
    detection layer corresponding to yolo_layer.c of darknet
    """

    def __init__(self, config_model, layer_no, in_ch, ignore_thre=0.7):
        """
        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """

        super(YOLOLayer, self).__init__()
        # 特征图相对于
        strides = [32, 16, 8]  # fixed
        # 预设的锚点框列表，保存了所有的锚点框长宽
        self.anchors = config_model['ANCHORS']
        # 指定不同YOLO层使用的锚点框
        self.anch_mask = config_model['ANCH_MASK'][layer_no]
        # 某一个YOLO层使用的锚点框个数，默认为3
        self.n_anchors = len(self.anch_mask)
        # 数据集类别数
        self.n_classes = config_model['N_CLASSES']
        # 阈值
        self.ignore_thre = ignore_thre
        # 损失函数，work for ???
        self.l2_loss = nn.MSELoss(size_average=False)
        self.bce_loss = nn.BCELoss(size_average=False)
        # 第N个YOLO层使用的步长，也就是输入图像大小和使用的特征数据之间的缩放比率
        self.stride = strides[layer_no]
        # 按比例缩放锚点框长／宽
        self.all_anchors_grid = [(w / self.stride, h / self.stride)
                                 for w, h in self.anchors]
        # 采集指定YOLO使用的锚点
        self.masked_anchors = [self.all_anchors_grid[i]
                               for i in self.anch_mask]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)
        # 1x1卷积操作，计算特征图中每个网格的预测框（锚点框数量*(类别数+4(xywh)+1(置信度))）
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=self.n_anchors * (self.n_classes + 5),
                              kernel_size=1, stride=1, padding=0)

    def forward(self, xin, labels=None):
        """
        In this
        Args:
            xin (torch.Tensor): input feature map whose size is :math:`(N, C, H, W)`, \
                where N, C, H, W denote batchsize, channel width, height, width respectively.
            labels (torch.Tensor): label data whose size is :math:`(N, K, 5)`. \
                N and K denote batchsize and number of labels.
                Each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
        Returns:
            loss (torch.Tensor): total loss - the target of backprop.
            loss_xy (torch.Tensor): x, y loss - calculated by binary cross entropy (BCE) \
                with boxsize-dependent weights.
            loss_wh (torch.Tensor): w, h loss - calculated by l2 without size averaging and \
                with boxsize-dependent weights.
            loss_obj (torch.Tensor): objectness loss - calculated by BCE.
            loss_cls (torch.Tensor): classification loss - calculated by BCE for each class.
            loss_l2 (torch.Tensor): total l2 loss - only for logging.
        """
        output = self.conv(xin)

        # 批量大小
        batchsize = output.shape[0]
        # 特征图空间尺寸
        fsize = output.shape[2]
        # 输出通道数
        # n_ch = 4(xywh) + 1(conf) + n_classes
        n_ch = 5 + self.n_classes
        dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor

        # [B, C_out, F_H, F_W] -> [B, n_anchors, n_ch, F_H, F_W]
        # C_out = n_anchors * (5 + n_classes)
        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        # [B, n_anchors, n_ch, F_H, F_W] -> [B, n_anchors, F_H, F_W, n_ch]
        output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

        # logistic activation for xy, obj, cls
        # 针对预测坐标(xy)和预测分类结果执行sigmoid运算，将数值归一化到(0, 1)之间
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

        # calculate pred - xywh obj cls
        # 网格坐标
        # [0, 1, 2, ..., F_W - 1] -> [B, n_anchors, F_H, F_W]
        x_shift = dtype(np.broadcast_to(np.arange(fsize, dtype=np.float32), output.shape[:4]))
        # [0, 1, 2, ..., F_H - 1] -> [F_H, 1] -> [B, n_anchors, F_H, F_W]
        y_shift = dtype(np.broadcast_to(np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4]))

        # [n_anchors, 2]
        masked_anchors = np.array(self.masked_anchors)

        # [n_anchors] -> [1, n_anchors, 1, 1] -> [B, n_anchors, F_H, F_W]
        w_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 0], (1, self.n_anchors, 1, 1)), output.shape[:4]))
        # [n_anchors] -> [1, n_anchors, 1, 1] -> [B, n_anchors, F_H, F_W]
        h_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 1], (1, self.n_anchors, 1, 1)), output.shape[:4]))

        pred = output.clone()
        # 预测框坐标x0加上每个网格的左上角坐标x
        # b_x = sigmoid(t_x) + c_x
        pred[..., 0] += x_shift
        # 预测框坐标y0加上每个网格的左上角坐标y
        # b_y = sigmoid(t_y) + c_y
        pred[..., 1] += y_shift
        # 计算预测框长/宽的实际长度
        # b_w = exp(t_w) * p_w
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        # b_h = exp(t_h) * p_h
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors

        if labels is None:  # not training
            # 推理阶段，不计算损失
            # 将预测框坐标按比例返回到原图大小
            pred[..., :4] *= self.stride
            # [B, n_anchors, F_H, F_W, n_ch] -> [B, n_anchors * F_H * F_W, n_ch]
            # return pred.view(batchsize, -1, n_ch).data
            return pred.reshape(batchsize, -1, n_ch).data

        # 训练阶段，计算损失
        #
        # 获取预测框的xywh
        pred = pred[..., :4].data

        # target assignment

        # [B, n_anchors, F_H, F_W, 4+n_classes]
        tgt_mask = torch.zeros(batchsize, self.n_anchors,
                               fsize, fsize, 4 + self.n_classes).type(dtype)
        # [B, n_anchors, F_H, F_W]
        obj_mask = torch.ones(batchsize, self.n_anchors,
                              fsize, fsize).type(dtype)
        # [B, n_anchors, F_H, F_W, 2]
        tgt_scale = torch.zeros(batchsize, self.n_anchors,
                                fsize, fsize, 2).type(dtype)

        # [B, n_anchors, F_H, F_W, n_ch]
        # n_ch = 4(xywh) + 1(conf) + n_classes
        target = torch.zeros(batchsize, self.n_anchors,
                             fsize, fsize, n_ch).type(dtype)

        labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = labels[:, :, 1] * fsize
        truth_y_all = labels[:, :, 2] * fsize
        truth_w_all = labels[:, :, 3] * fsize
        truth_h_all = labels[:, :, 4] * fsize
        truth_i_all = truth_x_all.to(torch.int16).numpy()
        truth_j_all = truth_y_all.to(torch.int16).numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = dtype(np.zeros((n, 4)))
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors)
            best_n_all = np.argmax(anchor_ious_all, axis=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_mask[0]) | (
                    best_n_all == self.anch_mask[1]) | (best_n_all == self.anch_mask[2]))

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(
                pred[b].reshape(-1, 4), truth_box, xyxy=False)
            # pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            # RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.
            # obj_mask[b] = 1 - pred_best_iou
            obj_mask[b] = ~pred_best_iou

            if sum(best_n_mask) == 0:
                continue

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - \
                                            truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - \
                                            truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti,
                                                  0].to(torch.int16).numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(
                        2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

        # loss calculation

        output[..., 4] *= obj_mask
        output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        output[..., 2:4] *= tgt_scale

        target[..., 4] *= obj_mask
        target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        target[..., 2:4] *= tgt_scale

        bceloss = nn.BCELoss(weight=tgt_scale * tgt_scale, size_average=False)  # weighted BCEloss
        loss_xy = bceloss(output[..., :2], target[..., :2])
        loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4]) / 2
        loss_obj = self.bce_loss(output[..., 4], target[..., 4])
        loss_cls = self.bce_loss(output[..., 5:], target[..., 5:])
        loss_l2 = self.l2_loss(output, target)

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2
