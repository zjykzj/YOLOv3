# -*- coding: utf-8 -*-

"""
@date: 2023/1/3 下午5:59
@file: cocodataset.py
@author: zj
@description: 
"""

import cv2
import os.path

import numpy as np
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset


# def label2yolobox(labels, info_img):
#     """
#     Transform coco labels to yolo box labels
#     Args:
#         labels (numpy.ndarray): label data whose shape is :math:`(N, 5)`.
#             Each label consists of [class, x, y, w, h] where \
#                 class (float): class index.
#                 x, y, w, h (float) : coordinates of \
#                     left-top points, width, and height of a bounding box.
#                     Values range from 0 to width or height of the image.
#         info_img : tuple of h, w, nh, nw, dx, dy.
#             h, w (int): original shape of the image
#             nh, nw (int): shape of the resized image without padding
#             dx, dy (int): pad size
#         maxsize (int): target image size after pre-processing
#         lrflip (bool): horizontal flip flag
#
#     Returns:
#         labels:label data whose size is :math:`(N, 5)`.
#             Each label consists of [class, xc, yc, w, h] where
#                 class (float): class index.
#                 xc, yc (float) : center of bbox whose values range from 0 to 1.
#                 w, h (float) : size of bbox whose values range from 0 to 1.
#     """
#     src_h, src_w, dst_h, dst_w = info_img
#     # xywh -> xyxy
#     x1 = labels[:, 1] / src_w
#     y1 = labels[:, 2] / src_h
#     x2 = (labels[:, 1] + labels[:, 3]) / src_w
#     y2 = (labels[:, 2] + labels[:, 4]) / src_h
#
#     # 计算目标图像对应边界框坐标以及宽高，计算相对比率
#     # x_center, y_center, b_w, b_h
#     #
#     # dst_x / src_x = dst_w / src_w
#     # dst_x = src_x * dst_w / src_w
#     # labels[:, 1] = ((x1 + x2) / 2) * dst_w / src_w
#     # labels[:, 2] = ((y1 + y2) / 2) * dst_h / src_h
#     # labels[:, 3] *= dst_w / src_w
#     # labels[:, 4] *= dst_h / dst_h
#     labels[:, 1] = ((x1 + x2) / 2)
#     labels[:, 2] = ((y1 + y2) / 2)
#     labels[:, 3] /= src_w
#     labels[:, 4] /= dst_h
#     return labels


def resize_and_pad(src_img, bboxes, dst_size, jitter_ratio=0.0):
    """
    src_img: [H, W, 3]
    bboxes: [K, 4] x1/y1/b_w/b_h
    """
    src_h, src_w = src_img.shape[:2]

    dh = jitter_ratio * src_h
    dw = jitter_ratio * src_w
    new_ratio = (src_w + np.random.uniform(low=-dw, high=dw)) / (src_h + np.random.uniform(low=-dh, high=dh))
    if new_ratio < 1:
        # 高大于宽
        # 设置目标大小为高，等比例缩放宽，剩余部分进行填充
        dst_h = dst_size
        dst_w = new_ratio * dst_size
    else:
        # 宽大于等于高
        # 设置目标大小为宽，等比例缩放高，剩余部分进行填充
        dst_w = dst_size
        dst_h = dst_size / new_ratio
    dst_w = int(dst_w)
    dst_h = int(dst_h)

    # 等比例进行上下或者左右填充
    dx = (dst_size - dst_w) // 2
    dy = (dst_size - dst_h) // 2

    # 先进行图像缩放，然后创建目标图像，填充ROI区域
    resized_img = cv2.resize(src_img, (dst_w, dst_h))
    padded_img = np.zeros((dst_size, dst_size, 3), dtype=np.uint8) * 127
    padded_img[dy:dy + dst_h, dx:dx + dst_w, :] = resized_img

    # 进行缩放以及填充后需要相应的修改坐标位置
    # x_left_top
    bboxes[:, 0] = bboxes[:, 0] / src_w * dst_w + dx
    # y_left_top
    bboxes[:, 1] = bboxes[:, 1] / src_h * dst_h + dy
    # 对于宽/高而言，仅需缩放对应比例即可，不需要增加填充坐标
    # box_w
    bboxes[:, 2] = bboxes[:, 2] / src_w * dst_w
    # box_h
    bboxes[:, 3] = bboxes[:, 3] / src_h * dst_h

    img_info = [src_h, src_w, dst_h, dst_w, dst_size, dx, dy]
    return padded_img, bboxes, img_info


def left_right_flip(img, bboxes):
    dst_img = np.flip(img, axis=2).copy()

    h, w = img.shape[:2]
    # 左右翻转，所以宽/高不变，变换左上角坐标(x1, y1)和右上角坐标(x2, y1)进行替换
    x2 = bboxes[:, 0] + bboxes[:, 2]
    # y1/2/h不变，仅变换x1 = w - x2
    bboxes[:, 0] = w - x2

    return dst_img, bboxes


def rand_scale(s):
    """
    calculate random scaling factor
    Args:
        s (float): range of the random scale.
    Returns:
        random scaling factor (float) whose range is
        from 1 / s to s .
    """
    scale = np.random.uniform(low=1, high=s)
    if np.random.rand() > 0.5:
        return scale
    return 1 / scale


def color_dithering(src_img, hue, saturation, exposure):
    """
    src_img: 图像 [H, W, 3]
    hue: 色调
    saturation: 饱和度
    exposure: 曝光度
    """
    dhue = np.random.uniform(low=-hue, high=hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)

    img = cv2.cvtColor(src_img, cv2.COLOR_RGB2HSV)
    img = np.asarray(img, dtype=np.float32) / 255.
    img[:, :, 1] *= dsat
    img[:, :, 2] *= dexp
    H = img[:, :, 0] + dhue

    if dhue > 0:
        H[H > 1.0] -= 1.0
    else:
        H[H < 0.0] += 1.0

    img[:, :, 0] = H
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = np.asarray(img, dtype=np.float32)

    return img


def label2yolobox(labels):
    """
    Transform coco labels to yolo box labels
    Args:
        labels (numpy.ndarray): label data whose shape is :math:`(N, 5)`.
            Each label consists of [class, x, y, w, h] where \
                class (float): class index.
                x, y, w, h (float) : coordinates of \
                    left-top points, width, and height of a bounding box.
                    Values range from 0 to width or height of the image.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
        maxsize (int): target image size after pre-processing
        lrflip (bool): horizontal flip flag

    Returns:
        labels:label data whose size is :math:`(N, 5)`.
            Each label consists of [class, xc, yc, w, h] where
                class (float): class index.
                xc, yc (float) : center of bbox whose values range from 0 to 1.
                w, h (float) : size of bbox whose values range from 0 to 1.
    """
    # x1/y1/w/h -> x1/y1/x2/y2
    x1 = labels[:, 1]
    y1 = labels[:, 2]
    x2 = (labels[:, 1] + labels[:, 3])
    y2 = (labels[:, 2] + labels[:, 4])

    # x1/y1/x2/y2 -> xc/yc/w/h
    labels[:, 1] = ((x1 + x2) / 2)
    labels[:, 2] = ((y1 + y2) / 2)
    return labels


class COCODataset(Dataset):

    def __init__(self, root, name='train2017', img_size=416, min_size=1, model_type='YOLO', is_train=True):
        self.root = root
        self.name = name
        self.img_size = img_size
        self.min_size = min_size
        self.model_type = model_type
        self.is_train = is_train

        if 'train' in self.name:
            json_file = 'instances_train2017.json'
        elif 'val' in self.name:
            json_file = 'instances_val2017.json'
        else:
            raise ValueError(f"{name} does not match any files")
        annotation_file = os.path.join(self.root, 'annotations', json_file)
        self.coco = COCO(annotation_file)

        # 获取图片ID列表
        self.ids = self.coco.getImgIds()
        # 获取类别ID
        self.class_ids = sorted(self.coco.getCatIds())

        # 单张图片预设的最大真值边界框数目
        self.max_num_labels = 50

        # 图像预处理
        self.jitter_ratio = 0.3
        self.is_flip = True
        self.hue = 0.1
        self.saturation = 1.5
        self.exposure = 1.5

    def __len__(self):
        return len(self.ids)

    def preprocess(self, img, bboxes):
        """
        训练阶段，执行：

        1. 图像翻转
        2. 空间抖动
        3. 图像缩放
        4. 颜色抖动

        推理阶段，执行：

        1. 图像缩放
        """
        # BGR -> RGB
        img = img[:, :, ::-1]
        if self.is_train:
            # 首先进行缩放+填充+空间抖动
            img, bboxes, img_info = resize_and_pad(img, bboxes, self.img_size, self.jitter_ratio)
            # 然后进行左右翻转
            if self.is_flip and np.random.randn() > 0.5:
                img, bboxes = left_right_flip(img, bboxes)
            # 最后进行颜色抖动
            if self.hue > 0 or self.saturation > 0 or self.exposure > 0:
                img = color_dithering(img, self.hue, self.saturation, self.exposure)
        else:
            # 进行缩放+填充，不执行空间抖动
            img, bboxes, img_info = resize_and_pad(img, bboxes, self.img_size, 0.)

        return img, bboxes, img_info

    def __getitem__(self, index):
        # 获取ID
        img_id = self.ids[index]
        # 获取图像路径
        img_file = os.path.join(self.root, 'images', self.name, '{:012}'.format(img_id) + '.jpg')
        # 获取标注框信息
        anno_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)
        labels = []
        for anno in annotations:
            if anno['bbox'][2] > self.min_size and anno['bbox'][3] > self.min_size:
                tmp_label = [self.class_ids.index(anno['category_id'])]
                # bbox: [x, y, w, h]
                tmp_label.extend(anno['bbox'])
                labels.insert(0, tmp_label)
        labels = np.stack(labels)

        # 读取图像
        img = cv2.imread(img_file)
        # 图像预处理
        img, bboxes, img_info = self.preprocess(img, labels[:, 1:])
        labels[:, 1:] = bboxes
        assert isinstance(img_info, list)
        img_info.append(img_id)
        img_info.append(index)
        assert np.all(bboxes < self.img_size), print(img_info, '\n', bboxes)
        # 数据预处理
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous() / 255

        # 每幅图像设置固定个数的真值边界框，不足的填充为0
        padded_labels = np.zeros((self.max_num_labels, 5))
        if len(labels) > 0:
            labels = np.stack(labels)
            if 'YOLO' in self.model_type:
                labels = label2yolobox(labels)
                assert np.all(labels < self.img_size), print(img_info, '\n', labels)
            padded_labels[range(len(labels))[:self.max_num_labels]] = labels[:self.max_num_labels]
        padded_labels = torch.from_numpy(padded_labels)

        # return img, padded_labels, labels, info_img
        # img: [3, H, W]
        # padded_labels: [K, 5]
        target = dict({
            'padded_labels': padded_labels,
            "img_info": img_info
        })
        print(padded_labels)
        return img, target


if __name__ == '__main__':
    dataset = COCODataset("COCO", name='train2017', img_size=608, is_train=True)
    # dataset = COCODataset("COCO", name='val2017', img_size=416, is_train=False)

    # img, target = dataset.__getitem__(333)
    # img, target = dataset.__getitem__(57756)
    img, target = dataset.__getitem__(87564)
    print(img.shape)
    padded_labels = target['padded_labels']
    img_info = target['img_info']
    print(padded_labels.shape)
    print(img_info)
    print(padded_labels)
