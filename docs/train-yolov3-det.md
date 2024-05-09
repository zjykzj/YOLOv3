# YOLOv3

## VOC

* YOLOv3

```shell
# python3 val.py --weights runs/yolov3_voc.pt --data VOC.yaml --device 4
val: data=/workdir/custom/YOLOv3_bbb/data/VOC.yaml, weights=['runs/yolov3_voc.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=4, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=False, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
yolov3_voc summary: 198 layers, 67238145 parameters, 0 gradients, 151.5 GFLOPs
val: Scanning /workdir/datasets/VOC/labels/test2007.cache... 4952 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4952/4952 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 155/155 01:06
                   all       4952      12032      0.798      0.761      0.821      0.552
             aeroplane       4952        285      0.939      0.839      0.908      0.594
               bicycle       4952        337      0.919      0.831      0.907      0.623
                  bird       4952        459      0.801      0.712      0.796      0.489
                  boat       4952        263      0.671       0.73       0.76      0.428
                bottle       4952        469      0.726      0.684      0.714      0.444
                   bus       4952        213      0.883      0.816      0.911       0.73
                   car       4952       1201      0.821      0.892      0.906      0.645
                   cat       4952        358      0.887      0.788      0.896      0.643
                 chair       4952        756      0.634       0.61      0.652      0.397
                   cow       4952        244      0.744      0.702      0.765      0.494
           diningtable       4952        206      0.677      0.794      0.772      0.546
                   dog       4952        489      0.872      0.713      0.848      0.581
                 horse       4952        348      0.905      0.805      0.905      0.653
             motorbike       4952        325      0.884      0.799      0.901        0.6
                person       4952       4528       0.83      0.833      0.873      0.543
           pottedplant       4952        480      0.654      0.515      0.544      0.274
                 sheep       4952        242      0.768      0.764      0.815      0.541
                  sofa       4952        239      0.689      0.769      0.815      0.604
                 train       4952        282      0.825      0.862      0.902      0.633
             tvmonitor       4952        308      0.822      0.773      0.826      0.567
Speed: 0.1ms pre-process, 6.9ms inference, 1.6ms NMS per image at shape (32, 3, 640, 640)
```

* YOLOv3-Fast

```shell
# python3 val.py --weights runs/yolov3-fast_voc.pt --data VOC.yaml --device 4
yolov3-fast_voc summary: 108 layers, 39945921 parameters, 0 gradients, 76.0 GFLOPs
val: Scanning /workdir/datasets/VOC/labels/test2007.cache... 4952 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4952/4952 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 155/155 00:55
                   all       4952      12032      0.727      0.717      0.756      0.444
             aeroplane       4952        285      0.679      0.804      0.785      0.398
               bicycle       4952        337      0.865        0.8      0.871      0.543
                  bird       4952        459      0.793      0.644      0.733      0.422
                  boat       4952        263      0.622      0.677      0.673      0.343
                bottle       4952        469      0.731      0.674      0.694      0.409
                   bus       4952        213      0.746      0.793      0.823      0.549
                   car       4952       1201      0.765      0.884      0.876      0.562
                   cat       4952        358      0.667      0.751      0.751      0.413
                 chair       4952        756      0.639      0.558      0.621      0.359
                   cow       4952        244      0.746      0.586      0.744      0.433
           diningtable       4952        206      0.542      0.712      0.659      0.362
                   dog       4952        489      0.782      0.666      0.769      0.433
                 horse       4952        348      0.857      0.782      0.864      0.535
             motorbike       4952        325      0.818       0.76      0.835      0.494
                person       4952       4528       0.81      0.799      0.857       0.51
           pottedplant       4952        480      0.617      0.479      0.485      0.229
                 sheep       4952        242      0.772      0.728      0.797      0.507
                  sofa       4952        239      0.629      0.703      0.723      0.424
                 train       4952        282      0.701      0.805      0.809      0.467
             tvmonitor       4952        308      0.761       0.74      0.758       0.49
Speed: 0.1ms pre-process, 4.2ms inference, 2.1ms NMS per image at shape (32, 3, 640, 640)
```

## COCO

* YOLOv3

```shell
# python3 val.py --weights runs/yolov3_coco.pt --data coco.yaml --device 4
yolov3_coco summary: 198 layers, 67561245 parameters, 0 gradients, 152.5 GFLOPs
val: Scanning /workdir/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   1%|          | 1/157 00:01WARNING ⚠️ NMS time limit 2.100s exceeded
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   1%|▏         | 2/157 00:09WARNING ⚠️ NMS time limit 2.100s exceeded
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   2%|▏         | 3/157 00:15WARNING ⚠️ NMS time limit 2.100s exceeded
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 01:43
                   all       5000      36335       0.66      0.572      0.594      0.364
Speed: 0.1ms pre-process, 6.9ms inference, 3.8ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp5/yolov3_coco_predictions.json...
loading annotations into memory...
Done (t=0.45s)
creating index...
index created!
Loading and preparing results...
DONE (t=8.56s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=89.04s).
Accumulating evaluation results...
DONE (t=23.05s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.370
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.604
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.220
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.408
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.483
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.308
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.398
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.603
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.702
```

* YOLOv3-Fast

```shell
# python3 val.py --weights runs/yolov3-fast_coco.pt --data coco.yaml --device 4
yolov3-fast_coco summary: 108 layers, 40269021 parameters, 0 gradients, 77.0 GFLOPs
val: Scanning /workdir/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   1%|          | 1/157 00:01WARNING ⚠️ NMS time limit 2.100s exceeded
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   1%|▏         | 2/157 00:09WARNING ⚠️ NMS time limit 2.100s exceeded
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   2%|▏         | 3/157 00:15WARNING ⚠️ NMS time limit 2.100s exceeded
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 01:35
                   all       5000      36335      0.619      0.521      0.536      0.303
Speed: 0.1ms pre-process, 4.3ms inference, 4.5ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp4/yolov3-fast_coco_predictions.json...
loading annotations into memory...
Done (t=0.45s)
creating index...
index created!
Loading and preparing results...
DONE (t=8.93s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=91.16s).
Accumulating evaluation results...
DONE (t=23.00s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.309
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.547
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.310
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.189
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.358
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.390
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.270
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.365
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.563
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.632
```