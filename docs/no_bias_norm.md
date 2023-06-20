
# NO_BIAS/NO_NORM

* commit id: `c331ab14a9a0b27eef1a2e850a0feb335aa93dad`

I conducted experiments on whether to perform weight decay for the bias of convolutional layers and the weights and bias of normalized layers

The specific configuration file settings are as follows:

```text
OPTIMIZER :
    TYPE: SGD
    LR: 0.0001
    MOMENTUM: 0.9
# weight decay
    DECAY: 0.0005
    NO_BIAS: True <-----
    NO_NORM: True <-----
```

The specific code implementation is in the file [build.py](../yolo/optim/optimizers/build.py)

## Experiment 1

Compare [yolov3_voc.cfg](../configs/yolov3_voc.cfg) (`Use Bias/Norm`) with [yolov3_voc_v3.cfg](../configs/yolov3_voc_v3.cfg) (`No Bias/Norm`), use [YOLOv3Loss](../yolo/model/yololoss.py):

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "36321" main_amp.py -c configs/yolov3_voc.cfg --opt-level=O1 ../datasets/voc
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "36321" main_amp.py -c configs/yolov3_voc_v3.cfg --opt-level=O1 ../datasets/voc
```

The training results are as follows:

```shell
python eval.py -c configs/yolov3_voc.cfg -ckpt outputs/yolov3_voc/model_best.pth.tar --traversal ../datasets/voc
[06/20 02:19:09][INFO] eval.py:  89: Input Size：[320x320] ap50_95: = -1.0000 ap50: = 0.7145
[06/20 02:19:09][INFO] eval.py:  89: Input Size：[352x352] ap50_95: = -1.0000 ap50: = 0.7204
[06/20 02:19:09][INFO] eval.py:  89: Input Size：[384x384] ap50_95: = -1.0000 ap50: = 0.7392
[06/20 02:19:09][INFO] eval.py:  89: Input Size：[416x416] ap50_95: = -1.0000 ap50: = 0.7492
[06/20 02:19:09][INFO] eval.py:  89: Input Size：[448x448] ap50_95: = -1.0000 ap50: = 0.7526
[06/20 02:19:09][INFO] eval.py:  89: Input Size：[480x480] ap50_95: = -1.0000 ap50: = 0.7599
[06/20 02:19:09][INFO] eval.py:  89: Input Size：[512x512] ap50_95: = -1.0000 ap50: = 0.7540
[06/20 02:19:09][INFO] eval.py:  89: Input Size：[544x544] ap50_95: = -1.0000 ap50: = 0.7527
[06/20 02:19:09][INFO] eval.py:  89: Input Size：[576x576] ap50_95: = -1.0000 ap50: = 0.7425
[06/20 02:19:09][INFO] eval.py:  89: Input Size：[608x608] ap50_95: = -1.0000 ap50: = 0.7367
python eval.py -c configs/yolov3_voc_v3.cfg -ckpt outputs/yolov3_voc_v3/model_best.pth.tar --traversal ../datasets/voc
[06/20 10:24:34][INFO] eval.py:  89: Input Size：[320x320] ap50_95: = -1.0000 ap50: = 0.7110
[06/20 10:24:34][INFO] eval.py:  89: Input Size：[352x352] ap50_95: = -1.0000 ap50: = 0.7239
[06/20 10:24:34][INFO] eval.py:  89: Input Size：[384x384] ap50_95: = -1.0000 ap50: = 0.7415
[06/20 10:24:34][INFO] eval.py:  89: Input Size：[416x416] ap50_95: = -1.0000 ap50: = 0.7555
[06/20 10:24:34][INFO] eval.py:  89: Input Size：[448x448] ap50_95: = -1.0000 ap50: = 0.7529
[06/20 10:24:34][INFO] eval.py:  89: Input Size：[480x480] ap50_95: = -1.0000 ap50: = 0.7565
[06/20 10:24:34][INFO] eval.py:  89: Input Size：[512x512] ap50_95: = -1.0000 ap50: = 0.7585
[06/20 10:24:34][INFO] eval.py:  89: Input Size：[544x544] ap50_95: = -1.0000 ap50: = 0.7575
[06/20 10:24:34][INFO] eval.py:  89: Input Size：[576x576] ap50_95: = -1.0000 ap50: = 0.7573
[06/20 10:24:34][INFO] eval.py:  89: Input Size：[608x608] ap50_95: = -1.0000 ap50: = 0.7372
```

## Experiment 2

Compare [yolov3_voc_v2.cfg](../configs/yolov3_voc_v2.cfg) (`Use Bias/Norm`) with [yolov3_voc_v4.cfg](../configs/yolov3_voc_v4.cfg) (`No Bias/Norm`), use [YOLOv3LossV2](../yolo/model/yololossv2.py):

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "36611" main_amp.py -c configs/yolov3_voc_v2.cfg --opt-level=O0 ../datasets/voc
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "36611" main_amp.py -c configs/yolov3_voc_v4.cfg --opt-level=O1 ../datasets/voc
```

The training results are as follows:

```shell
python eval.py -c configs/yolov3_voc_v2.cfg -ckpt outputs/yolov3_voc_v2/model_best.pth.tar --traversal ../datasets/voc
[06/20 02:28:58][INFO] eval.py:  89: Input Size：[320x320] ap50_95: = -1.0000 ap50: = 0.7109
[06/20 02:28:58][INFO] eval.py:  89: Input Size：[352x352] ap50_95: = -1.0000 ap50: = 0.7240
[06/20 02:28:58][INFO] eval.py:  89: Input Size：[384x384] ap50_95: = -1.0000 ap50: = 0.7405
[06/20 02:28:58][INFO] eval.py:  89: Input Size：[416x416] ap50_95: = -1.0000 ap50: = 0.7507
[06/20 02:28:58][INFO] eval.py:  89: Input Size：[448x448] ap50_95: = -1.0000 ap50: = 0.7514
[06/20 02:28:58][INFO] eval.py:  89: Input Size：[480x480] ap50_95: = -1.0000 ap50: = 0.7557
[06/20 02:28:58][INFO] eval.py:  89: Input Size：[512x512] ap50_95: = -1.0000 ap50: = 0.7527
[06/20 02:28:58][INFO] eval.py:  89: Input Size：[544x544] ap50_95: = -1.0000 ap50: = 0.7492
[06/20 02:28:58][INFO] eval.py:  89: Input Size：[576x576] ap50_95: = -1.0000 ap50: = 0.7431
[06/20 02:28:58][INFO] eval.py:  89: Input Size：[608x608] ap50_95: = -1.0000 ap50: = 0.7394
python eval.py -c configs/yolov3_voc_v4.cfg -ckpt outputs/yolov3_voc_v4/model_best.pth.tar --traversal ../datasets/voc
[06/20 10:30:23][INFO] eval.py:  89: Input Size：[320x320] ap50_95: = -1.0000 ap50: = 0.7226
[06/20 10:30:23][INFO] eval.py:  89: Input Size：[352x352] ap50_95: = -1.0000 ap50: = 0.7312
[06/20 10:30:23][INFO] eval.py:  89: Input Size：[384x384] ap50_95: = -1.0000 ap50: = 0.7432
[06/20 10:30:23][INFO] eval.py:  89: Input Size：[416x416] ap50_95: = -1.0000 ap50: = 0.7535
[06/20 10:30:23][INFO] eval.py:  89: Input Size：[448x448] ap50_95: = -1.0000 ap50: = 0.7573
[06/20 10:30:23][INFO] eval.py:  89: Input Size：[480x480] ap50_95: = -1.0000 ap50: = 0.7555
[06/20 10:30:23][INFO] eval.py:  89: Input Size：[512x512] ap50_95: = -1.0000 ap50: = 0.7494
[06/20 10:30:23][INFO] eval.py:  89: Input Size：[544x544] ap50_95: = -1.0000 ap50: = 0.7578
[06/20 10:30:23][INFO] eval.py:  89: Input Size：[576x576] ap50_95: = -1.0000 ap50: = 0.7460
[06/20 10:30:23][INFO] eval.py:  89: Input Size：[608x608] ap50_95: = -1.0000 ap50: = 0.7428
```