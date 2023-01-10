
# Trouble Shooting

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