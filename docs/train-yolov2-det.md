# YOLOv2

## VOC

* YOLOv2

```shell
# Train
python train.py --data VOC.yaml --weights "" --cfg yolov2_voc.yaml --img 640 --device 0 --yolov2loss
# Eval
# python3 val.py --weights runs/train/voc/exp/weights/best.pt --data VOC.yaml --img 640 --device 0
Fusing layers...
yolov2_voc summary: 53 layers, 50645053 parameters, 0 gradients, 69.5 GFLOPs
val: Scanning /workdir/datasets/VOC/labels/test2007.cache... 4952 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4952/4952 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 155/155 00:41
                   all       4952      12032      0.735      0.711      0.751      0.443
             aeroplane       4952        285      0.664      0.702      0.705      0.372
               bicycle       4952        337      0.862      0.814      0.864      0.526
                  bird       4952        459      0.779      0.636      0.729       0.39
                  boat       4952        263      0.618      0.593      0.622      0.305
                bottle       4952        469      0.783      0.516      0.613      0.319
                   bus       4952        213      0.798      0.808      0.857      0.597
                   car       4952       1201      0.814      0.836      0.859      0.554
                   cat       4952        358      0.707      0.811      0.776      0.477
                 chair       4952        756      0.642      0.557      0.628      0.341
                   cow       4952        244      0.748      0.758      0.789      0.489
           diningtable       4952        206      0.583        0.7       0.65      0.388
                   dog       4952        489      0.739      0.753        0.8      0.486
                 horse       4952        348      0.822      0.835      0.874      0.539
             motorbike       4952        325      0.808      0.751      0.831      0.498
                person       4952       4528      0.843      0.772      0.848      0.484
           pottedplant       4952        480      0.695      0.418      0.503      0.226
                 sheep       4952        242       0.73      0.682      0.758      0.467
                  sofa       4952        239      0.584      0.741      0.735      0.458
                 train       4952        282      0.727      0.839      0.825      0.477
             tvmonitor       4952        308      0.748      0.701      0.763      0.466
Speed: 0.1ms pre-process, 3.1ms inference, 1.3ms NMS per image at shape (32, 3, 640, 640)
```

* YOLOv2-Fast

```shell
# Train
python train.py --data VOC.yaml --weights "" --cfg yolov2-fast_voc.yaml --img 640 --device 0 --yolov2loss
# Eval
# python3 val.py --weights runs/train/voc/exp3/weights/best.pt --data VOC.yaml --img 640 --device 0
Fusing layers...
yolov2-fast_voc summary: 33 layers, 42367485 parameters, 0 gradients, 48.5 GFLOPs
val: Scanning /workdir/datasets/VOC/labels/test2007.cache... 4952 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4952/4952 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 155/155 00:37
                   all       4952      12032      0.626      0.612      0.626      0.298
             aeroplane       4952        285      0.551      0.621       0.61      0.226
               bicycle       4952        337      0.721      0.713      0.765      0.401
                  bird       4952        459      0.692      0.494      0.577      0.255
                  boat       4952        263      0.533      0.491      0.492      0.199
                bottle       4952        469      0.712      0.373       0.45      0.208
                   bus       4952        213        0.6      0.732      0.663      0.342
                   car       4952       1201      0.649      0.742      0.712      0.365
                   cat       4952        358      0.454      0.693      0.605      0.253
                 chair       4952        756      0.635      0.439       0.52      0.268
                   cow       4952        244      0.641      0.611      0.672      0.342
           diningtable       4952        206      0.521      0.573      0.539      0.206
                   dog       4952        489       0.54      0.642       0.63      0.282
                 horse       4952        348      0.661      0.773      0.789      0.401
             motorbike       4952        325      0.635      0.714      0.731      0.364
                person       4952       4528       0.75      0.707       0.76      0.371
           pottedplant       4952        480      0.663      0.352      0.434      0.173
                 sheep       4952        242      0.752      0.607      0.692      0.386
                  sofa       4952        239      0.493      0.598      0.554      0.263
                 train       4952        282      0.545      0.766      0.654      0.271
             tvmonitor       4952        308      0.769      0.597      0.668      0.381
Speed: 0.1ms pre-process, 2.3ms inference, 1.5ms NMS per image at shape (32, 3, 640, 640)
```

## COCO

* YOLOv2

```shell
# Train
python train.py --data coco.yaml --weights "" --cfg yolov2.yaml --img 640 --device 0 --yolov2loss
# Eval
# python3 val.py --weights runs/train/coco/exp/weights/best.pt --data coco.yaml --img 640 --device 0
Fusing layers...
yolov2 summary: 53 layers, 50952553 parameters, 0 gradients, 69.7 GFLOPs
val: Scanning /workdir/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   1%|          | 1/157 00:01WARNING ⚠️ NMS time limit 2.100s exceeded
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   2%|▏         | 3/157 00:08WARNING ⚠️ NMS time limit 2.100s exceeded
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 00:57
                   all       5000      36335      0.627       0.48      0.507      0.286
Speed: 0.1ms pre-process, 3.1ms inference, 2.0ms NMS per image at shape (32, 3, 640, 640)
```

* YOLOv2-Fast

```shell
# Train
python3 train.py --data coco.yaml --weights "" --cfg yolov2-fast.yaml --img 640 --device 0 --yolov2loss
# Eval
# python3 val.py --weights runs/train/coco/exp2/weights/best.pt --data coco.yaml --img 640 --device 0
Fusing layers...
yolov2-fast summary: 33 layers, 42674985 parameters, 0 gradients, 48.8 GFLOPs
val: Scanning /workdir/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   1%|          | 1/157 00:01WARNING ⚠️ NMS time limit 2.100s exceeded
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   1%|▏         | 2/157 00:04WARNING ⚠️ NMS time limit 2.100s exceeded
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   2%|▏         | 3/157 00:07WARNING ⚠️ NMS time limit 2.100s exceeded
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 00:53
                   all       5000      36335      0.549      0.402      0.412      0.201
Speed: 0.1ms pre-process, 2.4ms inference, 2.1ms NMS per image at shape (32, 3, 640, 640)
```