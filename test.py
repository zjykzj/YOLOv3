from __future__ import division

import argparse
import yaml

import torch

from yolo.utils.cocoapi_evaluator import COCOAPIEvaluator
from yolo.utils.parse_yolo_weights import parse_yolo_weights
from yolo.model.yolov3 import *

"""
操作流程：

1. 解析命令行参数 + 配置文件
2. 创建模型，初始化权重
3. 创建COCO评估器
4. 开始评估
"""


def parse_args():
    parser = argparse.ArgumentParser()
    # 模型配置以及训练配置
    parser.add_argument('--cfg', type=str, default='config/yolov3_default.cfg',
                        help='config file. see readme')
    # 权重路径
    parser.add_argument('--weights_path', type=str,
                        default=None, help='darknet weights file')
    # 数据加载线程数
    parser.add_argument('--n_cpu', type=int, default=0,
                        help='number of workers')
    # ???
    parser.add_argument('--checkpoint', type=str,
                        help='pytorch checkpoint file path')
    # 是否使用GPU（注意：本仓库仅实现了单GPU训练）
    parser.add_argument('--use_cuda', type=bool, default=True)
    return parser.parse_args()


def main():
    """
    YOLOv3 trainer. See README for details.
    """
    args = parse_args()
    print("Setting Arguments.. : ", args)

    cuda = torch.cuda.is_available() and args.use_cuda

    # Parse config settings
    with open(args.cfg, 'r') as f:
        # cfg = yaml.load(f)
        cfg = yaml.safe_load(f)

    print("successfully loaded config file: ", cfg)

    # 阈值
    ignore_thre = cfg['TRAIN']['IGNORETHRE']
    # Initiate model
    # YOLOv3还是通过模型定义方式获取YOLO模型！！！
    model = YOLOv3(cfg['MODEL'], ignore_thre=ignore_thre)

    # 预训练权重加载，共两种方式
    if args.weights_path:
        # 方式一：Darknet格式权重文件
        print("loading darknet weights....", args.weights_path)
        parse_yolo_weights(model, args.weights_path)
    elif args.checkpoint:
        # 方式二：Pytorch格式权重文件
        print("loading pytorch ckpt...", args.checkpoint)
        state = torch.load(args.checkpoint)
        if 'model_state_dict' in state.keys():
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)

    if cuda:
        # GPU训练
        print("using cuda")
        model = model.cuda()

    # COCO评估器，指定
    # 1. 模型类型，对于YOLO，需要转换边界框坐标格式
    # 2. 数据集路径，默认'COCO/'
    # 3. 测试图像大小：YOLOv3采用416
    # 4. 置信度阈值：YOLOv3采用0.8
    # 5. NMS阈值：YOLOv3采用0.45
    evaluator = COCOAPIEvaluator(model_type=cfg['MODEL']['TYPE'],
                                 data_dir='COCO/',
                                 img_size=cfg['TEST']['IMGSIZE'],
                                 confthre=cfg['TEST']['CONFTHRE'],
                                 nmsthre=cfg['TEST']['NMSTHRE'])

    # 每隔eval_interval进行评估
    print("Begin evaluating ...")
    ap50_95, ap50 = evaluator.evaluate(model)


if __name__ == '__main__':
    main()
