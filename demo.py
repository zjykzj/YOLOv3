# -*- coding: utf-8 -*-
import os.path

from typing import List, Tuple, Dict

import cv2
import yaml

import argparse
from argparse import Namespace

import numpy as np
from numpy import ndarray

import torch.cuda
from torch import Tensor
from torch.nn import Module

from yolo.data.cocodataset import get_coco_label_names
from yolo.data.transform import Transform
from yolo.model.yolov3 import YOLOv3
from yolo.util.utils import yolobox2label, postprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/yolov3_default.cfg')
    parser.add_argument('--ckpt', type=str,
                        help='path to the check point file')
    parser.add_argument('--image', type=str)
    parser.add_argument('--background', action='store_true',
                        default=False, help='background(no-display mode. save "./mountain_output.png")')
    parser.add_argument('--detect_thresh', type=float,
                        default=None, help='confidence threshold')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    return args, cfg


def image_preprocess(args: Namespace, cfg: Dict):
    """
    图像预处理

    读取图像，执行
    1. 图像格式转换
    2. 数据维度转换
    3. 图像大小缩放
    4. 数据归一化
    """
    transform = Transform(cfg, is_train=False)

    # BGR
    img = cv2.imread(args.image)
    # img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
    img_raw = img.copy()

    imgsize = cfg['TEST']['IMGSIZE']
    img, _, img_info = transform(img, np.array([]), imgsize)
    # 数据预处理
    img = torch.from_numpy(img).permute(2, 0, 1).contiguous() / 255
    img = img.unsqueeze(0)
    print("img:", img.shape)

    # 返回输入图像数据、原始图像数据、图像缩放前后信息
    return img, img_raw, img_info


def model_init(args: Namespace, cfg: Dict):
    """
    创建模型，赋值预训练权重
    """
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = YOLOv3(cfg['MODEL'], device=device).to(device)
    assert args.ckpt, '--ckpt must be specified'

    if args.ckpt:
        print("=> loading checkpoint '{}'".format(args.ckpt))
        checkpoint = torch.load(args.ckpt, map_location=device)

        state_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model, device


def parse_info(outputs: List, info_img: List or Tuple):
    coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()

    bboxes = list()
    classes = list()
    colors = list()

    # x1/y1: 左上角坐标
    # x2/y2: 右下角坐标
    # conf: 置信度
    # cls_conf: 分类置信度
    # cls_pred: 分类下标
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:
        cls_id = coco_class_ids[int(cls_pred)]
        print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
        print('\t+ Label: %s, Conf: %.5f' %
              (coco_class_names[cls_id], cls_conf.item()))
        box = yolobox2label([y1, x1, y2, x2], info_img)
        bboxes.append(box)
        classes.append(cls_id)
        colors.append(coco_class_colors[int(cls_pred)])

    return bboxes, classes, colors, coco_class_names


def process(args: Namespace, cfg: Dict, img: Tensor, model: Module, device: torch.device):
    """
    模型推理 + 后处理（置信度阈值过滤 + IoU阈值过滤）
    """
    confthre = cfg['TEST']['CONFTHRE']
    nmsthre = cfg['TEST']['NMSTHRE']
    if args.detect_thresh:
        confthre = args.detect_thresh

    with torch.no_grad():
        # img: [1, 3, 416, 416]
        # 执行模型推理，批量计算每幅图像的预测框坐标以及对应的目标置信度+分类概率
        outputs = model(img.to(device)).cpu()
        # outputs: [B, N_bbox, 4(xywh)+1(conf)+num_classes]
        # 图像后处理，执行预测边界框的坐标转换以及置信度阈值过滤+NMS IoU阈值过滤
        outputs = postprocess(outputs, 80, confthre, nmsthre)

    return outputs


# def show_bbox(args: Namespace, img_raw: ndarray, bboxes: List, classes: List, coco_class_names: List, colors: List):
#     if args.background:
#         import matplotlib
#         matplotlib.use('Agg')
#
#     from yolo.util.vis_bbox import vis_bbox
#     import matplotlib.pyplot as plt
#
#     vis_bbox(
#         img_raw, bboxes, label=classes, label_names=coco_class_names,
#         instance_colors=colors, linewidth=2)
#     plt.show()
#
#     if args.background:
#         plt.savefig('mountain_output.png')


def show_bbox(save_dir: str,  # 保存路径
              img_raw_list: List[ndarray],  # 原始图像数据列表, BGR ndarray
              img_name_list: List[str],  # 图像名列表
              bboxes_list: List,  # 预测边界框
              names_list: List,  # 预测边界框对象名
              colors_list: List):  # 预测边界框绘制颜色
    """
    对于绘图，输入如下数据：
    1. 原始图像
    2. 预测框坐标
    3. 数据集名 + 分类概率
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    line_width = 2
    txt_color = (255, 255, 255)

    for img_raw, img_name, bboxes, names, colors in zip(
            img_raw_list, img_name_list, bboxes_list, names_list, colors_list):
        im = img_raw
        lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

        for box, pred_name, color in zip(bboxes, names, colors):
            # box: [y1, x1, y2, x2]
            # print(box, name, color)
            assert len(box) == 4, box
            color = tuple([int(x) for x in color])

            # [y1, x1, y2, x2] -> [x1, y1] [x2, y2]
            p1, p2 = (int(box[1]), int(box[0])), (int(box[3]), int(box[2]))
            cv2.rectangle(im, p1, p2, color, 2)

            w, h = cv2.getTextSize(f'{pred_name}', 0, fontScale=0.5, thickness=1)[0]
            p1, p2 = (int(box[1]), int(box[0] - h)), (int(box[1] + w), int(box[0]))
            cv2.rectangle(im, p1, p2, color, thickness=-1)
            org = (int(box[1]), int(box[0]))
            cv2.putText(im, f'{pred_name}', org, cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.5, color=(0, 0, 0), thickness=1)

            # # p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            # p1, p2 = (int(box[1]), int(box[0])), (int(box[3]), int(box[2]))
            # cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
            #
            # tf = 1  # font thickness
            # w, h = cv2.getTextSize(name, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            # outside = p1[1] - h >= 3
            # p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            # cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            # cv2.putText(im,
            #             name, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            #             0,
            #             0.5,
            #             txt_color,
            #             thickness=tf,
            #             lineType=cv2.LINE_AA)

        im_path = os.path.join(save_dir, img_name)
        print(f"\t+ img path: {im_path}")
        cv2.imwrite(im_path, im)


def main():
    """
    Visualize the detection result for the given image and the pre-trained model.
    操作流程：

    1. 解析命令行参数 + 配置文件（使用easydict）
    2. 读取图像，预处理（图像通道转换 + 图像缩放 + 数据归一化 + 维度转换 + 数据格式转换）
    3. 创建模型，加载预训练权重
    4. 模型推理 + 数据后处理（置信度阈值过滤 + NMS阈值过滤）
    5. 预测框坐标转换
    6. 预测框绘制
    """
    args, cfg = parse_args()

    img, img_raw, img_info = image_preprocess(args, cfg)
    model, device = model_init(args, cfg)

    outputs = process(args, cfg, img, model, device)
    if outputs[0] is None:
        print("No Objects Deteted!!")
        return

    bboxes, classes, colors, coco_class_names = parse_info(outputs, img_info[:6])

    img_raw_list = [img_raw]
    image_name = os.path.basename(args.image)
    img_name_list = [image_name]
    bboxes_list = [bboxes]
    colors_list = [colors]

    pred_name_list = list()
    for cls_id in classes:
        pred_name_list.append(coco_class_names[cls_id + 1])
    pred_name_list = [pred_name_list]

    save_dir = './results'
    show_bbox(save_dir, img_raw_list, img_name_list, bboxes_list, pred_name_list, colors_list)
    # show_bbox(args, img_raw, bboxes, classes, coco_class_names, colors)


if __name__ == '__main__':
    main()
