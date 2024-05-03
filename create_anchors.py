# -*- coding: utf-8 -*-

"""
@date: 2024/3/2 下午2:49
@file: create_anchors.py
@author: zj
@description: Create a list of anchor points with different numbers
"""

from utils.dataloaders import LoadImagesAndLabels
from utils.autoanchor import kmean_anchors


def create_dataset(path):
    imgsz = 640
    batch_size = 16
    stride = 32

    dataset = LoadImagesAndLabels(
        path,
        imgsz,
        batch_size,
        augment=False,  # augmentation
        hyp=None,  # hyperparameters
        rect=False,  # rectangular batches
        cache_images=False,
        single_cls=False,
        stride=int(stride),
        pad=0.0,
        image_weights=False,
        prefix='')

    return dataset, imgsz


if __name__ == '__main__':
    # na: 5 - anchors: 13,15, 25,38, 62,61, 104,142, 282,273
    # na: 9 - anchors: 11,12, 15,34, 36,25, 34,63, 82,57, 69,140, 164,123, 150,269, 386,317
    path = '../datasets/coco/train2017.txt'
    # na: 5 - anchors: 46,61, 95,137, 255,174, 190,326, 453,371
    # na: 9 - anchors: 35,54, 85,71, 69,149, 161,144, 133,263, 224,318, 404,211, 345,430, 538,386
    # path = [
    #     '../datasets/VOC/images/train2012',
    #     '../datasets/VOC/images/train2007',
    #     '../datasets/VOC/images/val2012',
    #     '../datasets/VOC/images/val2007',
    # ]
    dataset, imgsz = create_dataset(path)

    thr = 4.0
    # num anchors
    for na in range(1, 10):
        anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        print(f"na: {na} - anchors: \n{anchors}")
