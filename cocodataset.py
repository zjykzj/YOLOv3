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


def label2yolobox(labels, info_img):
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
    src_h, src_w, dst_h, dst_w = info_img
    # xywh -> xyxy
    x1 = labels[:, 1]
    y1 = labels[:, 2]
    x2 = (labels[:, 1] + labels[:, 3])
    y2 = (labels[:, 2] + labels[:, 4])

    # 计算目标图像对应边界框坐标以及宽高，计算相对比率
    # x_center, y_center, b_w, b_h
    #
    # dst_x / src_x = dst_w / src_w
    # dst_x = src_x * dst_w / src_w
    labels[:, 1] = ((x1 + x2) / 2) * dst_w / src_w
    labels[:, 2] = ((y1 + y2) / 2) * dst_h / src_h
    labels[:, 3] *= dst_w / src_w
    labels[:, 4] *= dst_h / dst_h
    return labels


class COCODataset(Dataset):

    def __init__(self, root, name='train2017', img_size=416, min_size=1, model_type='YOLO'):
        self.root = root
        self.name = name
        self.img_size = img_size
        self.min_size = min_size
        self.model_type = model_type

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
        self.max_labels = 50

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_file = os.path.join(self.root, 'images', self.name, '{:012}'.format(img_id) + '.jpg')
        img = cv2.imread(img_file)
        src_h, src_w = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))
        info_img = [src_h, src_w, self.img_size, self.img_size]

        # load labels
        anno_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        labels = []
        for anno in annotations:
            if anno['bbox'][2] > self.min_size and anno['bbox'][3] > self.min_size:
                tmp_label = [anno['category_id']]
                tmp_label.extend(anno['bbox'])
                labels.insert(0, tmp_label)

        padded_labels = np.zeros((self.max_labels, 5))
        if len(labels) > 0:
            labels = np.stack(labels)
            if 'YOLO' in self.model_type:
                labels = label2yolobox(labels, info_img)
            padded_labels[range(len(labels))[:self.max_labels]] = labels[:self.max_labels]
        padded_labels = torch.from_numpy(padded_labels)

        return img, padded_labels, labels, info_img


if __name__ == '__main__':
    # dataset = COCODataset("COCO", name='train2017', img_size=416)
    dataset = COCODataset("COCO", name='val2017', img_size=416)

    img, padded_labels, labels, info_img = dataset.__getitem__(333)
    print(img.shape)
    print(len(labels))
    print(labels)
    print(padded_labels)
