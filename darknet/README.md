
# Darknet53

## Benchmark training

### Train

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "31226" main_amp.py -b 256 --workers 4 --lr 0.1 --weight-decay 1e-4 --epochs 90 --opt-level O1 ./imagenet/
```

```text
Prec@1 76.814 Prec@5 93.286
```

### Recipe

* `Model`: 
  * Type: Darknet53
  * Activation: LeakyReLU (0.1)
* `Train`:
  * Epochs: 90
  * Hybrid train: True
  * Distributed train: True
  * GPUs: 4
  * `Data`:
    * `Dataset`: 
      * Type: ImageNet Train
    * `Transform`:
      * RandomResizedCrop: 224
      * RandomHorizontalFlip
    * `Sampler`:
      * Type: DistributedSampler
    * `Dataloader`:
      * Batch size: 256
      * Num workers: 4
  * `Criterion`: CrossEntropyLoss
  * `Optimizer`: 
    * Type: SGD
    * LR: 1e-1
    * Weight decay: 1e-4
    * Momentum: 0.9
  * `Lr_Scheduler`:
    * Warmup: 5
    * MultiStep: [30, 60, 80]
* `Test`:
  * `Data`:
    * `Dataset`:
      * Type: ImageNet Val
      * Image size: 224
    * `Transform`:
      * Resize: 256
      * CenterCrop: 224

## Enhanced training

### Train

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "31226" main_amp.py -b 128 --workers 4 --lr 0.1 --weight-decay 1e-4 --epochs 120 --opt-level O1 ./imagenet/
```

Set `nn.LeakyReLU(0.01, inplace=True)`

```text
Prec@1 78.104 Prec@5 94.042
```

Set `nn.LeakyReLU(0.1, inplace=False)`

```text
Prec@1 77.920 Prec@5 93.810
```

### Recipe

* `Model`: 
  * Type: Darknet53
  * Activation: LeakyReLU (0.01)
* `Train`:
  * Epochs: 120
  * Hybrid train: True
  * Distributed train: True
  * GPUs: 4
  * `Data`:
    * `Dataset`: 
      * Type: ImageNet Train
    * `Transform`:
      * RandomResizedCrop: 256
      * RandomHorizontalFlip
      * RandAugment
    * `Sampler`:
      * Type: DistributedSampler
    * `Dataloader`:
      * Batch size: 256
      * Num workers: 4
  * `Criterion`: 
    * Type: LabelSmoothingLoss
    * Factor: 0.1
  * `Optimizer`: 
    * Type: SGD
    * LR: 1e-1
    * Weight decay: 1e-4
    * Momentum: 0.9
  * `Lr_Scheduler`:
    * Warmup: 5
    * MultiStep: [60, 90, 110]
* `Test`:
  * `Data`:
    * `Dataset`:
      * Type: ImageNet Val
    * `Transform`:
      * Resize: 288
      * CenterCrop: 256