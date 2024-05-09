
# Classify

## YOLOv2-fast

```shell
python classify/train.py --model runs/yolov2-fast_coco_wo_pretrained/weights/best.pt --cutoff 8 --data imagenet --img 224 --batch-size 256 --device 0
...
Model summary: 32 layers, 8754440 parameters, 8754440 gradients, 21.8 GFLOPs
...
Starting runs/yolov2-fast_coco_wo_pretrained/weights/best.pt training on imagenet dataset with 1000 classes for 10 epochs...

     Epoch   GPU_mem  train_loss    val_loss    top1_acc    top5_acc
      1/10     10.7G        3.76        3.33       0.453       0.714: 100%|██████████| 5005/5005 2:43:27
      2/10     13.2G        3.09        2.98       0.534       0.779: 100%|██████████| 5005/5005 2:11:29
      3/10     13.2G        2.88        2.85       0.568       0.806: 100%|██████████| 5005/5005 2:13:28
      4/10     13.2G        2.73        2.82       0.579       0.812: 100%|██████████| 5005/5005 2:13:01
      5/10     13.2G        2.59         2.8       0.585       0.817: 100%|██████████| 5005/5005 2:12:28
      6/10     13.2G        2.45        2.78       0.589       0.819: 100%|██████████| 5005/5005 2:15:09
      7/10     13.2G        2.29        2.78        0.59       0.821: 100%|██████████| 5005/5005 2:16:45
      8/10     13.2G        2.09        2.78       0.589        0.82: 100%|██████████| 5005/5005 2:14:54
      9/10     13.2G        1.88         2.8       0.585       0.817: 100%|██████████| 5005/5005 2:16:02
     10/10     13.2G        1.67        2.83       0.577       0.812: 100%|██████████| 5005/5005 2:13:07

Training complete (22.833 hours)
Results saved to runs/train-cls/exp
Predict:         python classify/predict.py --weights runs/train-cls/exp/weights/best.pt --source im.jpg
Validate:        python classify/val.py --weights runs/train-cls/exp/weights/best.pt --data /workdir/custom/datasets/imagenet
Export:          python export.py --weights runs/train-cls/exp/weights/best.pt --include onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train-cls/exp/weights/best.pt')
Visualize:       https://netron.app
```

## YOLOv2

```shell
# python classify/train.py --model runs/yolov2_coco_wo_pretrained/weights/best.pt --cutoff 18 --data imagenet --img 224 --batch-size 256 --device 0
...
Model summary: 62 layers, 17035656 parameters, 17035656 gradients, 42.8 GFLOPs
...
Starting runs/yolov2_coco_wo_pretrained/weights/best.pt training on imagenet dataset with 1000 classes for 10 epochs...

     Epoch   GPU_mem  train_loss    val_loss    top1_acc    top5_acc
      1/10     13.1G        6.29                                    :   2%|▏         | 80/5005 00:54
      1/10     13.1G        3.78        3.16       0.456       0.731: 100%|██████████| 5005/5005 2:13:59
      2/10     15.6G        2.94         2.7        0.57       0.816: 100%|██████████| 5005/5005 2:57:28
      3/10     15.6G         2.7        2.53       0.615       0.846: 100%|██████████| 5005/5005 2:35:56
      4/10     15.6G        2.54        2.48       0.629       0.856: 100%|██████████| 5005/5005 2:17:19
      5/10     15.6G        2.41        2.45       0.637       0.861: 100%|██████████| 5005/5005 2:13:11
      6/10     15.6G        2.28        2.42       0.644       0.865: 100%|██████████| 5005/5005 2:20:10
      7/10     15.6G        2.15        2.39        0.65        0.87: 100%|██████████| 5005/5005 2:17:45
      8/10     15.6G           2        2.37       0.653       0.873: 100%|██████████| 5005/5005 2:16:04
      9/10     15.6G        1.83        2.35       0.656       0.875: 100%|██████████| 5005/5005 2:14:13
     10/10     15.6G        1.64        2.36       0.657       0.875: 100%|██████████| 5005/5005 2:15:57

Training complete (23.704 hours)
Results saved to runs/train-cls/exp2
Predict:         python classify/predict.py --weights runs/train-cls/exp2/weights/best.pt --source im.jpg
Validate:        python classify/val.py --weights runs/train-cls/exp2/weights/best.pt --data /workdir/custom/datasets/imagenet
Export:          python export.py --weights runs/train-cls/exp2/weights/best.pt --include onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train-cls/exp2/weights/best.pt')
Visualize:       https://netron.app
```

## ShuffleNetV2

```shell
# python classify/train.py --model shufflenet_v2_x1_0 --data imagenet --img 224 --batch-size 256 --device 0
...
Model summary: 205 layers, 2278604 parameters, 2278604 gradients, 3.3 GFLOPs
...
Starting shufflenet_v2_x1_0 training on imagenet dataset with 1000 classes for 10 epochs...

     Epoch   GPU_mem  train_loss    val_loss    top1_acc    top5_acc
      1/10     3.48G        2.34        2.61       0.608       0.836: 100%|██████████| 5005/5005 3:17:47
      2/10     4.41G        2.45        2.54       0.617       0.844: 100%|██████████| 5005/5005 2:36:47
      3/10     4.41G        2.46        2.47       0.636       0.856: 100%|██████████| 5005/5005 2:22:04
      4/10     4.41G         2.4        2.44       0.641       0.861: 100%|██████████| 5005/5005 2:17:44
      5/10     4.41G        2.34        2.43       0.645       0.863: 100%|██████████| 5005/5005 2:17:54
      6/10     4.41G        2.27        2.42       0.646       0.865: 100%|██████████| 5005/5005 2:19:22
      7/10     4.41G        2.21        2.41       0.649       0.866: 100%|██████████| 5005/5005 2:21:13
      8/10     4.41G        2.14         2.4       0.652       0.867: 100%|██████████| 5005/5005 2:19:35
      9/10     4.41G        2.07        2.38       0.654       0.869: 100%|██████████| 5005/5005 2:17:49
     10/10     4.41G        1.99        2.37       0.656       0.869: 100%|██████████| 5005/5005 2:17:34

Training complete (24.465 hours)
Results saved to runs/train-cls/exp6
Predict:         python classify/predict.py --weights runs/train-cls/exp6/weights/best.pt --source im.jpg
Validate:        python classify/val.py --weights runs/train-cls/exp6/weights/best.pt --data /workdir/custom/datasets/imagenet
Export:          python export.py --weights runs/train-cls/exp6/weights/best.pt --include onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train-cls/exp6/weights/best.pt')
Visualize:       https://netron.app
```

## How to Eval

```shell
python3 classify/val.py --weights runs/train-cls/exp/weights/best.pt --data ../datasets/imagenet --img 224
```