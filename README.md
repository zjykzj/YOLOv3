<div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div>

<div align="center"><a title="" href="https://github.com/zjykzj/YOLOv3"><img align="center" src="./imgs/YOLOv3.png" alt=""></a></div>

<p align="center">
  «YOLOv3» reproduced the paper "YOLOv3: An Incremental Improvement"
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square" alt=""></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg" alt=""></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg" alt=""></a>
</p>

* Train using the `COCO train2017` dataset and test using the `COCO val2017` dataset with an input size of `416x416`. give the result as follows (*No version of the COCO dataset used in the paper was found*)

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
    <td class="tg-9y4h">0.304</td>
  </tr>
  <tr>
    <td class="tg-baqh">COCO AP[IoU=0.50]</td>
    <td class="tg-baqh">0.553</td>
    <td class="tg-baqh">0.558</td>
    <td class="tg-baqh">0.529</td>
  </tr>
</tbody>
</table>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Latest News](#latest-news)
- [Background](#background)
- [Prepare Data](#prepare-data)
  - [Pascal VOC](#pascal-voc)
  - [COCO](#coco)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Container](#container)
- [Usage](#usage)
  - [Train](#train)
  - [Eval](#eval)
  - [Demo](#demo)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Latest News

* ***[2023/06/22][v3.2](https://github.com/zjykzj/YOLOv3/releases/tag/v3.2). Remove Excess Code and Implementation.***
* ***[2023/06/22][v3.1](https://github.com/zjykzj/YOLOv3/releases/tag/v3.1). Reconstruct DATA Module and Preprocessing Module.***
* ***[2023/05/24][v3.0](https://github.com/zjykzj/YOLOv3/releases/tag/v3.0). Refer to [zjykzj/YOLOv2](https://github.com/zjykzj/YOLOv2) to reconstruct the entire project and train `Pascal VOC` and `COCO` datasets with `YOLOv2Loss`.***
* ***[2023/04/16][v2.0](https://github.com/zjykzj/YOLOv3/releases/tag/v2.0). Fixed preprocessing implementation, YOLOv3 network performance close to the original paper implementation.***
* ***[2023/02/16][v1.0](https://github.com/zjykzj/YOLOv3/releases/tag/v1.0). implementing preliminary YOLOv3 network training and inference implementation.***

## Background

The purpose of creating this warehouse is to better understand the YOLO series object detection network. Note: The realization of the project depends heavily on the implementation
of [DeNA/PyTorch_YOLOv3](https://github.com/DeNA/PyTorch_YOLOv3) and [NVIDIA/apex](https://github.com/NVIDIA/apex)

## Prepare Data

### Pascal VOC

Use this script [voc2yolov5.py](https://github.com/zjykzj/vocdev/blob/master/py/voc2yolov5.py)

```shell
python voc2yolov5.py -s /home/zj/data/voc -d /home/zj/data/voc/voc2yolov5-train -l trainval-2007 trainval-2012
python voc2yolov5.py -s /home/zj/data/voc -d /home/zj/data/voc/voc2yolov5-val -l test-2007
```

Then softlink the folder where the dataset is located to the specified location:

```shell
ln -s /path/to/voc /path/to/YOLOv3/../datasets/voc
```

### COCO

Use this script [get_coco.sh](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh)


## Installation

### Requirements

Refer to [requirements.txt](./requirements.txt) for installing the training environment

```shell
pip install -r requirements.txt
```

### Container

Development environment (Use nvidia docker container)

```shell
docker run --gpus all -it --rm -v </path/to/YOLOv3>:/app/YOLOv3 -v </path/to/COCO>:/app/YOLOv3/COCO nvcr.io/nvidia/pytorch:22.08-py3
```

## Usage

### Train

* One GPU

```shell
CUDA_VISIBLE_DEVICES=0 python main_amp.py -c configs/yolov3_voc.cfg --opt-level=O0 ../datasets/voc
```

* Multi-GPUs

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "36321" main_amp.py -c configs/yolov3_coco.cfg --opt-level=O1 ../datasets/coco
```

### Eval

```shell
python eval.py -c configs/yolov3_coco.cfg -ckpt outputs/yolov3_coco/model_best.pth.tar --traversal ../datasets/coco
Input Size：[320x320] ap50_95: = 0.2710 ap50: = 0.4824
Input Size：[352x352] ap50_95: = 0.2835 ap50: = 0.4992
Input Size：[384x384] ap50_95: = 0.2934 ap50: = 0.5131
Input Size：[416x416] ap50_95: = 0.3039 ap50: = 0.5278
Input Size：[448x448] ap50_95: = 0.3083 ap50: = 0.5337
Input Size：[480x480] ap50_95: = 0.3131 ap50: = 0.5386
Input Size：[512x512] ap50_95: = 0.3134 ap50: = 0.5403
Input Size：[544x544] ap50_95: = 0.3168 ap50: = 0.5459
Input Size：[576x576] ap50_95: = 0.3163 ap50: = 0.5439
Input Size：[608x608] ap50_95: = 0.3149 ap50: = 0.5435
python eval.py -c configs/yolov3_voc.cfg -ckpt outputs/yolov3_voc/model_best.pth.tar --traversal ../datasets/voc
Input Size：[320x320] ap50_95: = -1.0000 ap50: = 0.7226
Input Size：[352x352] ap50_95: = -1.0000 ap50: = 0.7312
Input Size：[384x384] ap50_95: = -1.0000 ap50: = 0.7432
Input Size：[416x416] ap50_95: = -1.0000 ap50: = 0.7535
Input Size：[448x448] ap50_95: = -1.0000 ap50: = 0.7573
Input Size：[480x480] ap50_95: = -1.0000 ap50: = 0.7555
Input Size：[512x512] ap50_95: = -1.0000 ap50: = 0.7494
Input Size：[544x544] ap50_95: = -1.0000 ap50: = 0.7578
Input Size：[576x576] ap50_95: = -1.0000 ap50: = 0.7460
Input Size：[608x608] ap50_95: = -1.0000 ap50: = 0.7428
```

### Demo

```shell
python demo.py -c 0.6 configs/yolov3_voc.cfg outputs/yolov3_voc/model_best.pth.tar --exp voc assets/voc2007-test/
```

<p align="left"><img src="results/voc/000237.jpg" height="240"\>  <img src="results/voc/000386.jpg" height="240"\></p>

```shell
python demo.py -c 0.6 configs/yolov3_coco.cfg outputs/yolov3_coco/model_best.pth.tar --exp coco assets/coco/
```

<p align="left"><img src="results/coco/bus.jpg" height="240"\>  <img src="results/coco/zidane.jpg" height="240"\></p>

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [DeNA/PyTorch_YOLOv3](https://github.com/DeNA/PyTorch_YOLOv3)
* [NVIDIA/apex](https://github.com/NVIDIA/apex)
* [ZJCV/ZCls2](https://github.com/ZJCV/ZCls2)
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [zjykzj/YOLOv2](https://github.com/zjykzj/YOLOv2)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/YOLOv3/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2022 zjykzj