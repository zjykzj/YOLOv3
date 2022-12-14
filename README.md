
# YOLOv3

Base library from [DeNA/PyTorch_YOLOv3](https://github.com/DeNA/PyTorch_YOLOv3) and [apex/examples/imagenet](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

## Train

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