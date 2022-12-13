
# YOLOv3

Base library from [DeNA/PyTorch_YOLOv3](https://github.com/DeNA/PyTorch_YOLOv3) and [apex/examples/imagenet](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

## Train

```shell
python -m torch.distributed.launch --nproc_per_node=2 main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O1 ./data/imagenet/
```

## ERROR

```text
# RuntimeError: unable to write to file </torch_3377_1311053497_0>: No space left on device (28)
# Fix: https://discuss.pytorch.org/t/unable-to-write-to-file-torch-18692-1954506624/9990/4
# docker run --gpus all --privileged -it --ipc=host -v /home/zj:/home/zj -v /data:/data --rm nvcr.io/nvidia/pytorch:22.08-py3
```
