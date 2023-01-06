
# YOLOv3

Base library from [DeNA/PyTorch_YOLOv3](https://github.com/DeNA/PyTorch_YOLOv3) and [apex/examples/imagenet](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

## Train

1. 默认将配置文件保存在

* 单GPU训练

```shell
CUDA_VISIBLE_DEVICES=3 python main_amp.py -b 16 --workers 4 --opt-level O0 ./COCO/
```

* 多GPU训练

```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 main_amp.py -b 16 --workers 4 --opt-level O0 ./COCO/
```

## ERROR

```text
RuntimeError: unable to write to file </torch_3377_1311053497_0>: No space left on device (28)
Fix:
    https://discuss.pytorch.org/t/unable-to-write-to-file-torch-18692-1954506624/9990/4
    docker run --gpus all --privileged -it --ipc=host -v /home/zj:/home/zj -v /data:/data --rm nvcr.io/nvidia/pytorch:22.08-py3
```

```text
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.
Fix:
    RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input #32564
    https://github.com/pytorch/pytorch/issues/32564#issuecomment-633739690
```

YOLOv3和YOLOv7应该是一样的操作步骤，唯一的差别在于模型的提取

## 训练配置

* 损失函数：YOLOLoss
* 优化器：SGD
* 学习率调度器： Warmup + StepLR

批量大小：4
子批次大小：16

也就是单次批量大小为64

学习率设置：

初始学习率 / 批量大小 / 子批次大小 = 0.001 / 4 / 16 = 1.5625e-05

初始学习率设置：0.001
动量大小：0.9
权重衰减大小：5e-4

## 数据

### 数据预处理

* 在训练阶段，执行 
  * 左右翻转 
  * 空间抖动 
  * 图像缩放（等比填充） 
  * 颜色抖动

前面三个属于空间变换，会改变图像原始坐标系，所以对应的边界框坐标也会发生变化；最后一个属于颜色变换，不会对边界框造成影响

* 在推理阶段，执行
  * 图像缩放（等比填充）

完成上述图像处理后均会执行数据归一化（除以255）操作。注意：针对不同格式的边界框还需要进行《格式转换》，将x1/y1/x2/y2 -> xc/yc/w/h

### 数据加载

从COCO数据集中采集图像以及对应的类别标签和真值边界框

### 操作流程

1. 输入采样图像的下标
2. 根据下标获取对应的图像路径
3. 根据下标获取对应的标注信息，包括类别下标以及真值标注框
4. 预处理操作
   1. 读取图像，进行格式转换以及维度转换
   2. 针对图像执行图像预处理，相应的转换标注框坐标
   3. 针对图像执行数据归一化操作
5. 在训练阶段，对于损失函数，需要知道预处理后的真值标注框的坐标
6. 在推理阶段，对于COCO评估器，需要将预测边界框转换回原始图像边界框，所以需要知道
   1. 原始图像大小
   2. 缩放后图像大小
   3. 填充后ROI区域左上角坐标
7. 综合来说，返回2个item:
   1. item1: torch.Tensor格式，预处理后的图像数据
   2. item2: 字典格式，包含了类别下标、预处理后的标注框、图像缩放/填充/抖动前后的大小以及ROI左上角坐标、提取图像列表下标


```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.077
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.124
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.084
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.010
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.061
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.147
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.072
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.085
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.085
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.010
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.065
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.165
```