# -*- coding: utf-8 -*-

import os

from torch.utils.data import Dataset
from pycocotools.coco import COCO

from yolo.utils.utils import *

"""
图像预处理：

1. 左右翻转
2. 空间抖动
3. 颜色抖动
4. 图像缩放
5. 颜色通道转换

对于真值标签框，忽略小于指定大小的边界框；并且指定了每幅图像使用的标签个数


"""


class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(self, model_type, data_dir='COCO', json_file='instances_train2017.json',
                 name='train2017', img_size=416, min_size=1):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            model_type (str): model name specified in config file
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
        """
        # 数据集根路径
        self.data_dir = data_dir
        # 标注文件
        self.json_file = json_file
        # 模型类型，是否是YOLO
        self.model_type = model_type
        # 初始化COCO数据类
        self.coco = COCO(self.data_dir + 'annotations/' + self.json_file)
        # 获取图片ID
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.name = name
        self.max_labels = 50
        # 输入数据大小
        self.img_size = img_size
        # 忽略小于min_size的边界框
        self.min_size = min_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        id_ = self.ids[index]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        # load image and preprocess
        # img_file = os.path.join(self.data_dir, self.name,
        img_file = os.path.join(self.data_dir, "images", self.name, '{:012}'.format(id_) + '.jpg')
        img = cv2.imread(img_file)

        if self.json_file == 'instances_val5k.json' and img is None:
            # img_file = os.path.join(self.data_dir, 'train2017', '{:012}'.format(id_) + '.jpg')
            img_file = os.path.join(self.data_dir, "images", 'train2017', '{:012}'.format(id_) + '.jpg')
            img = cv2.imread(img_file)
        assert img is not None

        # 对于目标检测任务的图像预处理，需要考虑到预处理前后真值边界框的同步变化
        # 图像缩放 + 填充
        img, info_img = preprocess(img, self.img_size)

        # 归一化
        img = np.transpose(img / 255., (2, 0, 1))

        # load labels
        labels = []
        for anno in annotations:
            if anno['bbox'][2] > self.min_size and anno['bbox'][3] > self.min_size:
                labels.append([])
                # 类别ID
                labels[-1].append(self.class_ids.index(anno['category_id']))
                # 类别框
                labels[-1].extend(anno['bbox'])

        padded_labels = np.zeros((self.max_labels, 5))
        if len(labels) > 0:
            labels = np.stack(labels)
            if 'YOLO' in self.model_type:
                labels = label2yolobox(labels, info_img, self.img_size)
            padded_labels[range(len(labels))[:self.max_labels]] = labels[:self.max_labels]
        padded_labels = torch.from_numpy(padded_labels)

        return img, padded_labels, info_img, id_
