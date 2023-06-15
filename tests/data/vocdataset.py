# -*- coding: utf-8 -*-

"""
@date: 2023/6/15 下午3:50
@file: vocdataset.py
@author: zj
@description: 
"""

import random
import time

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from yolo.data.dataset.vocdataset import VOCDataset
from yolo.data.transform import Transform
from yolo.data.target import Target

root = '../datasets/voc'

NUM_CLASSES = 20

W = 13
H = 13


def load_cfg(cfg_file):
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    return cfg


def assert_data(images, targets):
    B = len(images)

    # [B, num_max_det, 5] -> [B, num_max_det] -> [B]
    gt_num_objs = (targets.sum(dim=2) > 0).sum(dim=1)

    for bi in range(B):
        num_obj = gt_num_objs[bi]
        if num_obj == 0:
            continue
        # [4]: [x_c, y_c, w, h]
        gt_boxes = targets[bi][:num_obj][..., 1:]
        # [num_obj]
        gt_cls_ids = targets[bi][:num_obj][..., 0]

        gt_boxes[..., 0::2] *= W
        gt_boxes[..., 1::2] *= H

        for ni in range(num_obj):
            # [4]: [xc, yc, w, h]
            gt_box = gt_boxes[ni]
            # 对应网格下标
            cell_idx_x, cell_idx_y = torch.floor(gt_box[:2])
            assert cell_idx_x < W and cell_idx_y < H, f"{cell_idx_x} {cell_idx_y} {W} {H}"

            gt_class = gt_cls_ids[ni]
            assert int(gt_class) < NUM_CLASSES, f"{int(gt_class)} {NUM_CLASSES}"


def test_train(cfg_file, name):
    cfg = load_cfg(cfg_file)
    print(f"load cfg: {cfg_file}")

    train_dataset = VOCDataset(root, name, train=True, transform=Transform(cfg, is_train=True))
    print("Total len:", len(train_dataset))

    end = time.time()
    for i in range(len(train_dataset)):
        image, target = train_dataset.__getitem__(i)
        images = image.unsqueeze(0)
        targets = target.unsqueeze(0)

        assert_data(images, targets)
        print(f"[{i}/{len(train_dataset)}] {images.shape} {targets.shape}")
    print(f"Avg one time: {(time.time() - end) / len(train_dataset)}")

    # for num_workers in [16, 8, 4]:
    #     print(f"Train Dataloader, num_workers: {num_workers}")
    #     train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=num_workers,
    #                                   shuffle=False, sampler=None, pin_memory=True)
    #     end = time.time()
    #     for i, (images, targets) in enumerate(tqdm(train_dataloader)):
    #         assert_data(images, targets)
    #         # print(f"[{i}/{len(train_dataloader)}] {images.shape} {targets.shape}")
    #     print(f"Avg one time: {(time.time() - end) / len(train_dataset)}")


def test_val(cfg_file, name):
    cfg = load_cfg(cfg_file)
    print(f"load cfg: {cfg_file}")

    # test_dataset = VOCDataset(root, name, S=7, B=2, train=False, transform=Transform(is_train=False))
    # image, target = test_dataset.__getitem__(300)
    # print(image.shape, target.shape)

    val_dataset = VOCDataset(root, name, train=False, transform=Transform(cfg, is_train=False))
    print("Total len:", len(val_dataset))

    # i = 170
    # image, target = val_dataset.__getitem__(i)
    # print(i, image.shape, target['target'].shape, len(target['img_info']), target['image_name'])

    # for i in [31, 62, 100, 166, 169, 170, 633]:
    #     image, target = val_dataset.__getitem__(i)
    #     print(i, image.shape, target['target'].shape, len(target['img_info']))

    end = time.time()
    for i in range(len(val_dataset)):
        image, target = val_dataset.__getitem__(i)
        assert isinstance(target, Target)
        print(i, image.shape, target.target.shape, len(target.img_info), target.img_id)

        images = image.unsqueeze(0)
        targets = target.target.unsqueeze(0)
        assert_data(images, targets)
    print(f"Avg one time: {(time.time() - end) / len(val_dataset)}")


if __name__ == '__main__':
    random.seed(10)

    cfg_file = 'tests/data/voc.cfg'

    print("=> Pascal VOC Train")
    name = 'voc2yolov5-train'
    test_train(cfg_file, name)
    print("=> Pascal VOC Val")
    name = 'voc2yolov5-val'
    test_val(cfg_file, name)
