import json
import tempfile

import torch
from tqdm import tqdm

from pycocotools.cocoeval import COCOeval
from torch.autograd import Variable

from yolo.data.cocodataset import *
from yolo.utils.utils import *


class COCOAPIEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """

    def __init__(self, model_type, data_dir, img_size, confthre, nmsthre):
        """
        Args:
            model_type (str): model name specified in config file
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """

        augmentation = {'LRFLIP': False, 'JITTER': 0, 'RANDOM_PLACING': False,
                        'HUE': 0, 'SATURATION': 0, 'EXPOSURE': 0, 'RANDOM_DISTORT': False}

        self.dataset = COCODataset(model_type=model_type,
                                   data_dir=data_dir,
                                   img_size=img_size,
                                   json_file='instances_val2017.json',
                                   name='val2017')
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)
        self.img_size = img_size
        # self.confthre = 0.005  # from darknet
        self.confthre = 0.50  # from darknet
        self.nmsthre = nmsthre  # 0.45 (darknet)

    @torch.no_grad()
    def evaluate(self, model):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        # 推理模式
        model.eval()
        # 判断是否存在GPU环境
        cuda = torch.cuda.is_available()
        # 数据类型
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        ids = []
        data_dict = []
        # 批量加载数据（批量大小为１）
        # img: 处理后图像，numpy格式 [C, H, W]
        # _:
        # info_img: (原始高，原始宽，缩放后高，缩放后宽，ROI区域左上角x0，ROI区域左上角y0)
        # id_: 原始图像列表下标
        for img, _, info_img, id_ in tqdm(self.dataloader):
            info_img = [float(info) for info in info_img]
            # 从这里也判断出是单个推理
            id_ = int(id_)
            # 将原始图像下标挨个保存
            ids.append(id_)
            with torch.no_grad():
                # Numpy Ndarray -> Torch Tensor
                img = Variable(img.type(Tensor))
                # 模型推理，返回预测结果
                # img: [B, 3, 416, 416]
                outputs = model(img)
            # 后处理，进行置信度阈值过滤 + NMS阈值过滤
            # 输入outputs: [B, 预测框数目, 85(xywh + obj_confg + num_classes)]
            # 输出outputs: [B, 过滤后的预测框数目, 7(xyxy + obj_conf + cls_conf + cls_id)]
            outputs = postprocess(outputs, 80, self.confthre, self.nmsthre)
            # 从这里也可以看出是单张推理
            # 如果结果为空，那么不执行后续运算
            if outputs[0] is None:
                continue
            # 提取单张图片的运行结果
            # outputs: [N_ind, 7]
            outputs = outputs[0].cpu().data

            for output in outputs:
                x1 = float(output[0])
                y1 = float(output[1])
                x2 = float(output[2])
                y2 = float(output[3])
                # 分类标签
                label = self.dataset.class_ids[int(output[6])]
                # 转换到原始图像边界框坐标
                box = yolobox2label((y1, x1, y2, x2), info_img)
                # [y1, x1, y2, x2] -> [x1, y1, w, h]
                bbox = [box[1], box[0], box[3] - box[1], box[2] - box[0]]
                # 置信度 = 目标置信度 * 分类置信度
                score = float(output[4].data.item() * output[5].data.item())  # object score * class score
                # 保存计算结果
                A = {"image_id": id_, "category_id": label, "bbox": bbox,
                     "score": score, "segmentation": []}  # COCO json format
                data_dict.append(A)

        annType = ['segm', 'bbox', 'keypoints']

        # 计算完成所有测试图像的预测结果后
        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataset.coco
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, 'w'))
            cocoDt = cocoGt.loadRes(tmp)
            cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # AP50_95, AP50
            return cocoEval.stats[0], cocoEval.stats[1]
        else:
            return 0, 0
