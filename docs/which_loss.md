
# Which Loss

I have attempted to implement two YOLOv3Losses, one referring to [zjykzj/YOLOv2](https://github.com/zjykzj/YOLOv2) and the other referring to [DeNA/PyTorch_YOLOv3](https://github.com/DeNA/PyTorch_YOLOv3).

I trained the VOC and Coco datasets separately.

* commit id: `c331ab14a9a0b27eef1a2e850a0feb335aa93dad`

## VOC

```shell
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

>From the training results, it can be observed that there is not much difference between YOLOv3Loss and YOLOv3LossV2.

## COCO

```shell
python eval.py -c configs/yolov3_coco.cfg -ckpt outputs/yolov3_coco/model_best.pth.tar --traversal ../datasets/coco
[06/23 08:01:48][INFO] eval.py:  89: Input Size：[320x320] ap50_95: = 0.0602 ap50: = 0.1682
[06/23 08:01:48][INFO] eval.py:  89: Input Size：[352x352] ap50_95: = 0.0647 ap50: = 0.1802
[06/23 08:01:48][INFO] eval.py:  89: Input Size：[384x384] ap50_95: = 0.0679 ap50: = 0.1861
[06/23 08:01:48][INFO] eval.py:  89: Input Size：[416x416] ap50_95: = 0.0716 ap50: = 0.1957
[06/23 08:01:48][INFO] eval.py:  89: Input Size：[448x448] ap50_95: = 0.0734 ap50: = 0.2014
[06/23 08:01:48][INFO] eval.py:  89: Input Size：[480x480] ap50_95: = 0.0749 ap50: = 0.2038
[06/23 08:01:48][INFO] eval.py:  89: Input Size：[512x512] ap50_95: = 0.0762 ap50: = 0.2088
[06/23 08:01:48][INFO] eval.py:  89: Input Size：[544x544] ap50_95: = 0.0768 ap50: = 0.2100
[06/23 08:01:48][INFO] eval.py:  89: Input Size：[576x576] ap50_95: = 0.0773 ap50: = 0.2120
[06/23 08:01:48][INFO] eval.py:  89: Input Size：[608x608] ap50_95: = 0.0772 ap50: = 0.2123
python eval.py -c configs/yolov3_coco_v2.cfg -ckpt outputs/yolov3_coco_v2/model_best.pth.tar --traversal ../datasets/coco
[06/23 08:11:51][INFO] eval.py:  89: Input Size：[320x320] ap50_95: = 0.2710 ap50: = 0.4824
[06/23 08:11:51][INFO] eval.py:  89: Input Size：[352x352] ap50_95: = 0.2835 ap50: = 0.4992
[06/23 08:11:51][INFO] eval.py:  89: Input Size：[384x384] ap50_95: = 0.2934 ap50: = 0.5131
[06/23 08:11:51][INFO] eval.py:  89: Input Size：[416x416] ap50_95: = 0.3039 ap50: = 0.5278
[06/23 08:11:51][INFO] eval.py:  89: Input Size：[448x448] ap50_95: = 0.3083 ap50: = 0.5337
[06/23 08:11:51][INFO] eval.py:  89: Input Size：[480x480] ap50_95: = 0.3131 ap50: = 0.5386
[06/23 08:11:51][INFO] eval.py:  89: Input Size：[512x512] ap50_95: = 0.3134 ap50: = 0.5403
[06/23 08:11:51][INFO] eval.py:  89: Input Size：[544x544] ap50_95: = 0.3168 ap50: = 0.5459
[06/23 08:11:51][INFO] eval.py:  89: Input Size：[576x576] ap50_95: = 0.3163 ap50: = 0.5439
[06/23 08:11:51][INFO] eval.py:  89: Input Size：[608x608] ap50_95: = 0.3149 ap50: = 0.5435
```

>It is obvious that YOLOv3Lossv2 has better training effects than YOLOv3Loss.

## Summary

The biggest difference between YOLOv3Loss and YOLOv3LossV2 lies in the use of BCELoss. YOLOv3Loss supports mixed precision training, which uses `binary_cross_entropy_with_logits` calculation of binary Cross-Entropy loss. On the contrary, YOLOv3LossV2 uses BCELoss and does not support mixed precision training, it will encounter the following issues:

```text
NotImplementedError:                                                                                                                                                                                                                                                              
amp does not work out-of-the-box with `F.binary_cross_entropy` or `torch.nn.BCELoss.` It requires that the output of the previous function be already a FloatTensor.                                                                                                              
                                                                                                                                                                                                                                                                                  
Most models have a Sigmoid right before BCELoss. In that case, you can use                                                                                                                                                                                                        
    torch.nn.BCEWithLogitsLoss                                                                                                                                                                                                                                                    
to combine Sigmoid+BCELoss into a single layer that is compatible with amp.                                                                                                                                                                                                       
Another option is to add                                                                                                                                                                                                                                                          
    amp.register_float_function(torch, 'sigmoid')                                                                                                                                                                                                                                 
before calling `amp.init()`.                                                                                                                                                                                                                                                      
If you _really_ know what you are doing, you can disable this warning by passing allow_banned=True to `amp.init()`.
```

I tried switching to `binary_cross_entropy_with_logits` implementation, but the calculated loss will be very outrageous, which is very different from the loss calculation of BCELoss. It is possible that YOLOv3Loss is also related to this. If I switch to the implementation of BCELoss and do not perform mixed precision training, its effect will be better.


    
