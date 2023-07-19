<div align="right">
  语言:
    🇨🇳
  <a title="英语" href="./README.md">🇺🇸</a>
</div>

<div align="center"><a title="" href="https://github.com/zjykzj/YOLOv3"><img align="center" src="./imgs/YOLOv3.png" alt=""></a></div>

<p align="center">
  «YOLOv3» 复现了论文 "YOLOv3: An Incremental Improvement"
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square" alt=""></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg" alt=""></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg" alt=""></a>
</p>

* 使用`VOC07+12 trainval`数据集进行训练，使用`VOC2007 Test`进行测试，输入大小为`416x416`。测试结果如下：

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-zkss{background-color:#FFF;border-color:inherit;color:#333;text-align:center;vertical-align:top}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-fr9f{background-color:#FFF;border-color:inherit;color:#333;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-y5w1{background-color:#FFF;border-color:inherit;color:#00E;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-9y4h{background-color:#FFF;border-color:inherit;color:#1F2328;text-align:center;vertical-align:middle}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-fr9f"></th>
    <th class="tg-fr9f"><span style="font-style:normal">Original (darknet)</span></th>
    <th class="tg-y5w1">DeNA/PyTorch_YOLOv3</th>
    <th class="tg-y5w1"><span style="font-weight:700;font-style:normal">zjykzj/YOLOv3(This)</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-fr9f">ARCH</td>
    <td class="tg-zkss">YOLOv3</td>
    <td class="tg-zkss">YOLOv3</td>
    <td class="tg-zkss">YOLOv3</td>
  </tr>
  <tr>
    <td class="tg-fr9f">COCO AP[IoU=0.50:0.95]</td>
    <td class="tg-zkss">0.310</td>
    <td class="tg-9y4h">0.311</td>
    <td class="tg-9y4h">0.314(<a href="https://github.com/zjykzj/YOLOv3/releases/tag/v4.0">v4.0</a>)/0.315(<a href="https://github.com/zjykzj/YOLOv3/releases/tag/v2.0">v2.0</a>)</td>
  </tr>
  <tr>
    <td class="tg-baqh">COCO AP[IoU=0.50]</td>
    <td class="tg-baqh">0.553</td>
    <td class="tg-baqh">0.558</td>
    <td class="tg-baqh">0.535(<a href="https://github.com/zjykzj/YOLOv3/releases/tag/v4.0">v4.0</a>)/0.543(<a href="https://github.com/zjykzj/YOLOv3/releases/tag/v2.0">v2.0</a>)</td>
  </tr>
</tbody>
</table>

## 内容列表

- [内容列表](#内容列表)
- [最近新闻](#最近新闻)
- [背景](#背景)
- [数据准备](#数据准备)
  - [Pascal VOC](#pascal-voc)
  - [COCO](#coco)
- [安装](#安装)
  - [需求](#需求)
  - [容器](#容器)
- [用法](#用法)
  - [训练](#训练)
  - [评估](#评估)
  - [示例](#示例)
- [主要维护人员](#主要维护人员)
- [致谢](#致谢)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 最近新闻

* ***[2023/07/19][v4.0](https://github.com/zjykzj/YOLOv3/releases/tag/v4.0). 添加[ultralytics/yolov5](https://github.com/ultralytics/yolov5)([485da42](https://github.com/ultralytics/yolov5/commit/485da42273839d20ea6bdaf142fd02c1027aba61)) 预处理实现以及支持AMP训练。***
* ***[2023/06/22][v3.2](https://github.com/zjykzj/YOLOv3/releases/tag/v3.2). 移除多余的代码和实现。***
* ***[2023/06/22][v3.1](https://github.com/zjykzj/YOLOv3/releases/tag/v3.1). 重构数据模块和预处理模块。***
* ***[2023/05/24][v3.0](https://github.com/zjykzj/YOLOv3/releases/tag/v3.0). 参考[zjykzj/YOLOv2](https://github.com/zjykzj/YOLOv2)重构整个工程，同时使用`YOLOv2Loss`训练`Pascal VOC`和`COCO`数据集。***
* ***[2023/04/16][v2.0](https://github.com/zjykzj/YOLOv3/releases/tag/v2.0). 修复预处理实现，现在YOLOv3的训练性能已经接近于原始论文实现。***
* ***[2023/02/16][v1.0](https://github.com/zjykzj/YOLOv3/releases/tag/v1.0). 初步实现YOLOv3网络训练和推理实现。***

## 背景

创建此仓库的目的是为了更好地理解YOLO系列目标检测网络。注意：该项目的实现参考了[DeNA/PyTorch_YOLOv3](https://github.com/DeNA/PyTorch_YOLOv3)和[NVIDIA/apex](https://github.com/NVIDIA/apex)

## 数据准备

### Pascal VOC

使用脚本[voc2yolov5.py](https://github.com/zjykzj/vocdev/blob/master/py/voc2yolov5.py)

```shell
python voc2yolov5.py -s /home/zj/data/voc -d /home/zj/data/voc/voc2yolov5-train -l trainval-2007 trainval-2012
python voc2yolov5.py -s /home/zj/data/voc -d /home/zj/data/voc/voc2yolov5-val -l test-2007
```

然后将数据集所在的文件夹软链接到指定位置：

```shell
ln -s /path/to/voc /path/to/YOLOv1/../datasets/voc
```

### COCO

使用脚本[get_coco.sh](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh)

## 安装

### 需求

查看[NVIDIA/apex](https://github.com/NVIDIA/apex)

### 容器

开发环境（使用nvidia docker容器）

```shell
docker run --gpus all -it --rm -v </path/to/YOLOv1>:/app/YOLOv1 -v </path/to/voc>:/app/datasets/voc nvcr.io/nvidia/pytorch:22.08-py3
```

## 用法

### 训练

* 单个GPU

```shell
CUDA_VISIBLE_DEVICES=0 python main_amp.py -c configs/yolov3_coco.cfg --opt-level=O1 ../datasets/coco
```

* 多个GPUs

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "36321" main_amp.py -c configs/yolov3_coco.cfg --opt-level=O1 ../datasets/coco
```

### 评估

```shell
python eval.py -c configs/yolov3_coco.cfg -ckpt outputs/yolov3_coco/model_best.pth.tar ../datasets/coco
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.314
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.535
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.323
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.133
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.342
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.467
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.272
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.436
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.252
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.473
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.594
python eval.py -c configs/yolov3_voc.cfg -ckpt outputs/yolov3_voc/model_best.pth.tar ../datasets/voc
VOC07 metric? Yes
AP for aeroplane = 0.8442
AP for bicycle = 0.8575
AP for bird = 0.7730
AP for boat = 0.6824
AP for bottle = 0.6737
AP for bus = 0.8505
AP for car = 0.8663
AP for cat = 0.8667
AP for chair = 0.6073
AP for cow = 0.8196
AP for diningtable = 0.7213
AP for dog = 0.8433
AP for horse = 0.8761
AP for motorbike = 0.8568
AP for person = 0.8245
AP for pottedplant = 0.5211
AP for sheep = 0.8140
AP for sofa = 0.7385
AP for train = 0.8304
AP for tvmonitor = 0.7727
Mean AP = 0.7820
```

### 示例

```shell
python demo.py -c 0.6 configs/yolov3_coco.cfg outputs/yolov3_coco/model_best.pth.tar --exp coco assets/coco/
```

<p align="left"><img src="results/coco/bus.jpg" height="240"\>  <img src="results/coco/zidane.jpg" height="240"\></p>

```shell
python demo.py -c 0.6 configs/yolov3_voc.cfg outputs/yolov3_voc/model_best.pth.tar --exp voc assets/voc2007-test/
```

<p align="left"><img src="results/voc/000237.jpg" height="240"\>  <img src="results/voc/000386.jpg" height="240"\></p>

## 主要维护人员

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## 致谢

* [DeNA/PyTorch_YOLOv3](https://github.com/DeNA/PyTorch_YOLOv3)
* [NVIDIA/apex](https://github.com/NVIDIA/apex)
* [ZJCV/ZCls2](https://github.com/ZJCV/ZCls2)
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [zjykzj/YOLOv2](https://github.com/zjykzj/YOLOv2)

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/zjykzj/YOLOv3/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2023 zjykzj