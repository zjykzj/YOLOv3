# YOLOv3

## VOC

* YOLOv3

```shell
# python3 train.py --data VOC.yaml --weights "" --cfg yolov3_voc.yaml --img 640 --device 0 --yolov3loss
# python3 val.py --weights runs/train/exp/weights/best.pt --data VOC.yaml --device 0
yolov3_voc summary: 198 layers, 67238145 parameters, 0 gradients, 151.5 GFLOPs
val: Scanning /workdir/datasets/VOC/labels/test2007.cache... 4952 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4952/4952 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 155/155 01:04
                   all       4952      12032      0.811      0.742      0.816      0.568
             aeroplane       4952        285      0.938      0.791      0.897      0.607
               bicycle       4952        337      0.929      0.815      0.903      0.629
                  bird       4952        459      0.817      0.688      0.802       0.53
                  boat       4952        263      0.717      0.704      0.733      0.446
                bottle       4952        469      0.786       0.65      0.717      0.454
                   bus       4952        213      0.873      0.812      0.895      0.728
                   car       4952       1201      0.874      0.872      0.926      0.697
                   cat       4952        358      0.856      0.793      0.879      0.662
                 chair       4952        756      0.654      0.569       0.65      0.414
                   cow       4952        244      0.731      0.664      0.754      0.506
           diningtable       4952        206       0.72      0.762      0.782      0.566
                   dog       4952        489      0.864      0.712      0.837      0.606
                 horse       4952        348      0.892      0.805      0.894      0.645
             motorbike       4952        325      0.878      0.774      0.875      0.587
                person       4952       4528      0.872      0.808      0.891      0.586
           pottedplant       4952        480       0.68      0.506      0.571      0.305
                 sheep       4952        242      0.729      0.744      0.813      0.574
                  sofa       4952        239      0.688      0.791        0.8      0.606
                 train       4952        282      0.857       0.83      0.878      0.633
             tvmonitor       4952        308      0.861      0.756       0.82      0.579
Speed: 0.1ms pre-process, 6.8ms inference, 1.6ms NMS per image at shape (32, 3, 640, 640)
```

* YOLOv3-Fast

```shell
# python3 train.py --data VOC.yaml --weights "" --cfg yolov3-fast_voc.yaml --img 640 --device 0 --yolov3loss
# python3 val.py --weights runs/train/exp2/weights/best.pt --data VOC.yaml --device 0
yolov3-fast_voc summary: 108 layers, 39945921 parameters, 0 gradients, 76.0 GFLOPs
val: Scanning /workdir/datasets/VOC/labels/test2007.cache... 4952 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4952/4952 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 155/155 00:52
                   all       4952      12032      0.734      0.704      0.745       0.45
             aeroplane       4952        285      0.759      0.747      0.796      0.427
               bicycle       4952        337      0.858      0.773      0.856       0.53
                  bird       4952        459      0.792      0.613      0.724      0.421
                  boat       4952        263      0.667      0.631      0.653      0.353
                bottle       4952        469      0.784      0.648      0.702      0.426
                   bus       4952        213      0.767      0.798      0.827       0.57
                   car       4952       1201      0.791      0.859      0.875      0.574
                   cat       4952        358      0.643      0.761      0.673      0.373
                 chair       4952        756      0.659      0.553      0.613      0.373
                   cow       4952        244       0.73      0.615      0.725      0.458
           diningtable       4952        206       0.56      0.684       0.66      0.366
                   dog       4952        489      0.715      0.671      0.732      0.417
                 horse       4952        348      0.832      0.767      0.852      0.544
             motorbike       4952        325        0.8      0.772      0.819      0.488
                person       4952       4528      0.855      0.783      0.863      0.526
           pottedplant       4952        480       0.69       0.44      0.518      0.256
                 sheep       4952        242      0.717      0.702      0.751      0.502
                  sofa       4952        239       0.58      0.711      0.695      0.424
                 train       4952        282      0.695      0.826      0.809      0.468
             tvmonitor       4952        308      0.785      0.734      0.756      0.499
Speed: 0.1ms pre-process, 4.3ms inference, 1.6ms NMS per image at shape (32, 3, 640, 640)
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