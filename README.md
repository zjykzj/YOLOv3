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

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-7btt"><span style="font-style:normal">Original (darknet)</span></th>
    <th class="tg-7btt">DeNA/PyTorch_YOLOv3</th>
    <th class="tg-7btt"><span style="font-style:normal">ultralytics/yolov3</span></th>
    <th class="tg-7btt"><span style="font-style:normal">ultralytics/yolov3-tiny</span></th>
    <th class="tg-7btt"><span style="font-style:normal">ultralytics/yolov3-spp</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7btt">dataset</td>
    <td class="tg-c3ow">coco 2017 train/val</td>
    <td class="tg-c3ow">coco 2017 train/val</td>
    <td class="tg-c3ow">coco 2017 train/val</td>
    <td class="tg-c3ow">coco 2017 train/val</td>
    <td class="tg-c3ow">coco 2017 train/val</td>
  </tr>
  <tr>
    <td class="tg-7btt">train epoch</td>
    <td class="tg-c3ow">300</td>
    <td class="tg-c3ow">300</td>
    <td class="tg-c3ow">100</td>
    <td class="tg-c3ow">100</td>
    <td class="tg-c3ow">100</td>
  </tr>
  <tr>
    <td class="tg-7btt">COCO AP[IoU=0.50:0.95], inference</td>
    <td class="tg-c3ow">0.310</td>
    <td class="tg-c3ow">0.311</td>
    <td class="tg-c3ow">0.442</td>
    <td class="tg-c3ow">0.180</td>
    <td class="tg-7btt">0.447</td>
  </tr>
  <tr>
    <td class="tg-c3ow"><span style="font-weight:600;font-style:normal">COCO AP[IoU=0.50], inference</span></td>
    <td class="tg-c3ow">0.553</td>
    <td class="tg-c3ow">0.558</td>
    <td class="tg-c3ow">0.642</td>
    <td class="tg-c3ow">0.347</td>
    <td class="tg-7btt">0.645</td>
  </tr>
  <tr>
    <td class="tg-7btt">conf_thre</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">0.005</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">0.001</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">0.001</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">0.001</span></td>
  </tr>
  <tr>
    <td class="tg-7btt">nms_thre</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">0.45</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">0.6</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">0.6</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">0.6</span></td>
  </tr>
  <tr>
    <td class="tg-7btt">input_size</td>
    <td class="tg-c3ow">416</td>
    <td class="tg-c3ow">416</td>
    <td class="tg-c3ow">640</td>
    <td class="tg-c3ow">640</td>
    <td class="tg-c3ow">640</td>
  </tr>
</tbody>
</table>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

The purpose of creating this warehouse is to better understand the YOLO series object detection network. Note: The
realization of the project depends heavily on the implementation
of [DeNA/PyTorch_YOLOv3](https://github.com/DeNA/PyTorch_YOLOv3) and [NVIDIA/apex](https://github.com/NVIDIA/apex)

## Installation

Development environment (Use nvidia docker container)

```shell
docker run --gpus all -it --rm -v </path/to/YOLOv3>:/app/YOLOv3 -v </path/to/COCO>:/app/YOLOv3/COCO nvcr.io/nvidia/pytorch:22.08-py3
```

## Usage

### Train

* One GPU

```shell
CUDA_VISIBLE_DEVICES=0 python main_amp.py -c config/yolov3_default.cfg --opt-level=O0 COCO
```

* Multi GPU

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "32111" main_amp.py -c config/yolov3_default.cfg --opt-level=O0 COCO
```

### Test

```shell
python test.py --cfg config/yolov3_default.cfg --checkpoint outputs/yolov3_default/checkpoint_91.pth.tar COCO
```

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.292
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.512
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.300
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.117
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.311
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.444
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.256
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.390
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.411
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.215
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.443
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.573
```

### Demo

```shell
python demo.py --cfg config/yolov3_default.cfg --ckpt outputs/yolov3_default/checkpoint_91.pth.tar --image data/innsbruck.png --detect_thresh 0.5 --background
```

<p align="left"><img src="data/innsbruck_output.png" height="160"\>  <img src="data/mountain_output.png" height="160"\></p>

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [DeNA/PyTorch_YOLOv3](https://github.com/DeNA/PyTorch_YOLOv3)
* [NVIDIA/apex](https://github.com/NVIDIA/apex)
* [ZJCV/ZCls2](https://github.com/ZJCV/ZCls2)

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