
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