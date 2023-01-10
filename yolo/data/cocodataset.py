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

from yolo.util.utils import label2yolobox


class COCODataset(Dataset):

    def __init__(self, root, name: str = 'train2017', img_size: int = 416, min_size: int = 1,
                 model_type: str = 'YOLO', is_train: bool = True, transform=None, max_num_labels=50):
        self.root = root
        self.name = name
        self.img_size = img_size
        self.min_size = min_size
        self.model_type = model_type
        self.is_train = is_train
        self.transform = transform
        # 单张图片预设的最大真值边界框数目
        self.max_num_labels = max_num_labels

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

    def __len__(self):
        return len(self.ids)

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
        labels = np.array(labels)

        # 读取图像
        img = cv2.imread(img_file)
        # 图像预处理
        if self.transform is not None:
            if len(labels) > 0:
                img, bboxes, img_info = self.transform(img, labels[:, 1:], self.img_size)
                labels[:, 1:] = bboxes
            else:
                img, bboxes, img_info = self.transform(img, labels, self.img_size)
        assert isinstance(img_info, list)
        img_info.append(img_id)
        img_info.append(index)
        assert np.all(bboxes <= self.img_size), print(img_info, '\n', bboxes)
        # 数据预处理
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous() / 255

        # 每幅图像设置固定个数的真值边界框，不足的填充为0
        padded_labels = np.zeros((self.max_num_labels, 5))
        if len(labels) > 0:
            labels = np.stack(labels)
            if 'YOLO' in self.model_type:
                labels = label2yolobox(labels)
                assert np.all(labels <= self.img_size), print(img_info, '\n', labels)
            padded_labels[range(len(labels))[:self.max_num_labels]] = labels[:self.max_num_labels]
        padded_labels = torch.from_numpy(padded_labels)

        # return img, padded_labels, labels, info_img
        # img: [3, H, W]
        # padded_labels: [K, 5]
        target = dict({
            'padded_labels': padded_labels,
            "img_info": img_info
        })
        # print(padded_labels)
        return img, target

    def set_img_size(self, img_size):
        self.img_size = img_size

    def get_img_size(self):
        return self.img_size


if __name__ == '__main__':
    dataset = COCODataset("COCO", name='train2017', img_size=608, is_train=True)
    # dataset = COCODataset("COCO", name='val2017', img_size=416, is_train=False)

    # img, target = dataset.__getitem__(333)
    # img, target = dataset.__getitem__(57756)
    # img, target = dataset.__getitem__(87564)
    img, target = dataset.__getitem__(51264)
    print(img.shape)
    padded_labels = target['padded_labels']
    img_info = target['img_info']
    print(padded_labels.shape)
    print(img_info)
    print(padded_labels)
