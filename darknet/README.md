
# Darknet53

```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port "31226" main_amp.py -b 256 --workers 4 --lr 0.1 --weight-decay 1e-4 --epochs 90 --opt-level O1 ./imagenet/
```

```text
Prec@1 76.814 Prec@5 93.286
```