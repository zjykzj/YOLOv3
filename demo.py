from typing import List, Tuple, Optional

import argparse
from argparse import Namespace

import torch
import yaml

from numpy import ndarray

from torch import Tensor
from torch.nn import Module
from torch.autograd import Variable

from easydict import EasyDict

from yolo.model.yolov3 import *
from yolo.utils.utils import *
from yolo.utils.parse_yolo_weights import parse_yolo_weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cfg', type=str, default='config/yolov3_default.cfg')
    parser.add_argument('--ckpt', type=str,
                        help='path to the check point file')
    parser.add_argument('--weights_path', type=str,
                        default=None, help='path to weights file')
    parser.add_argument('--image', type=str)
    parser.add_argument('--background', action='store_true',
                        default=False, help='background(no-display mode. save "./output.png")')
    parser.add_argument('--detect_thresh', type=float,
                        default=None, help='confidence threshold')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(d=cfg)
    return args, cfg


def image_preprocess(args: Namespace, cfg: EasyDict):
    """
    图像预处理

    读取图像，执行
    1. 图像格式转换
    2. 数据维度转换
    3. 图像大小缩放
    4. 数据归一化
    5. 赋值到指定设备（cpu/gpu）
    """
    # 输入图像大小
    imgsize = cfg.TEST.IMGSIZE
    # BGR
    img = cv2.imread(args.image)
    # 原始图像处理
    # [H, W, C] -> [C, H, W]　同时　BGR -> RGB
    img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
    # 输入图像处理，增加一个图像缩放
    img, info_img = preprocess(img, imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
    # 图像归一化 + 通道转换（[H, W, C] -> [C, H, W]）
    img = np.transpose(img / 255., (2, 0, 1))
    # [C, H, W] -> [1, C, H, W]
    img = torch.from_numpy(img).float().unsqueeze(0)

    if args.gpu >= 0:
        img = Variable(img.type(torch.cuda.FloatTensor))
    else:
        img = Variable(img.type(torch.FloatTensor))

    # 返回输入图像数据、原始图像数据、图像缩放前后信息
    return img, img_raw, info_img


def model_preprocess(args: Namespace, cfg: EasyDict):
    """
    创建模型，赋值预训练权重
    """
    model = YOLOv3(cfg.MODEL)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    assert args.weights_path or args.ckpt, 'One of --weights_path and --ckpt must be specified'

    if args.weights_path:
        print("loading yolo weights %s" % (args.weights_path))
        # 加载YOLO权重
        parse_yolo_weights(model, args.weights_path)
    elif args.ckpt:
        print("loading checkpoint %s" % (args.ckpt))
        state = torch.load(args.ckpt)
        if 'model_state_dict' in state.keys():
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)

    model.eval()
    return model


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


def process(args: Namespace, cfg: EasyDict, model: Module, img: Tensor):
    """
    模型推理 + 后处理（置信度阈值过滤 + IoU阈值过滤）
    """
    confthre = cfg.TEST.CONFTHRE
    nmsthre = cfg.TEST.NMSTHRE
    if args.detect_thresh:
        confthre = args.detect_thresh

    with torch.no_grad():
        # img: [1, 3, 416, 416]
        # 执行模型推理，批量计算每幅图像的预测框坐标以及对应的目标置信度+分类概率
        outputs = model(img).cpu()
        # outputs: [B, N_bbox, 4(xywh)+1(conf)+num_classes]
        # 图像后处理，执行预测边界框的坐标转换以及置信度阈值过滤+NMS IoU阈值过滤
        outputs = postprocess(outputs, 80, confthre, nmsthre)

    return outputs


def show_bbox(args: Namespace, img_raw: ndarray, bboxes: List, classes: List, coco_class_names: List, colors: List):
    if args.background:
        import matplotlib
        matplotlib.use('Agg')

    from yolo.utils.vis_bbox import vis_bbox
    import matplotlib.pyplot as plt

    vis_bbox(
        img_raw, bboxes, label=classes, label_names=coco_class_names,
        instance_colors=colors, linewidth=2)
    plt.show()

    if args.background:
        plt.savefig('output.png')


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

    img, img_raw, info_img = image_preprocess(args, cfg)
    model = model_preprocess(args, cfg)

    outputs = process(args, cfg, model, img)
    if outputs[0] is None:
        print("No Objects Deteted!!")
        return

    bboxes, classes, colors, coco_class_names = parse_info(outputs, info_img)
    show_bbox(args, img_raw, bboxes, classes, coco_class_names, colors)


if __name__ == '__main__':
    main()
