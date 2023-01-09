# -*- coding: utf-8 -*-

import os
import shutil
import time
import json
import random
import tempfile
import argparse

from tqdm import tqdm
from pycocotools.cocoeval import COCOeval

from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from yolo.utils.utils import *


def parse():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size per process (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR',
                        help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/64.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')

    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    global best_ap50, best_ap50_95, args

    args = parse()
    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    best_ap50 = 0
    best_ap50_95 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    # load cfg
    cfg_file = 'config/yolov3_default.cfg'
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    print("args:", args)
    print("cfg:", cfg)
    # create model
    from yolov3 import YOLOv3
    model = YOLOv3(cfg['MODEL'])
    model = model.cuda().to(memory_format=memory_format)

    # Scale learning rate based on global batch size
    args.lr = args.lr * float(args.batch_size * args.world_size) / 64.
    # args.lr = args.lr / args.batch_size / args.world_size
    # optimizer setup
    # set weight decay only on conv.weight
    # 仅针对卷积层权重执行权重衰减
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if 'conv.weight' in key:
            # params += [{'params': value, 'weight_decay': args.weight_decay * args.batch_size * args.world_size}]
            params += [{'params': value, 'weight_decay': args.weight_decay}]
        else:
            params += [{'params': value, 'weight_decay': 0.0}]
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                # weight_decay=args.weight_decay * args.batch_size * args.world_size)
                                weight_decay=args.weight_decay)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    from yololoss import YOLOLoss
    criterion = YOLOLoss(cfg['MODEL'], ignore_thre=0.7)

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                global best_ap50, best_ap50_95
                best_ap50 = checkpoint['ap50']
                best_ap50_95 = checkpoint['ap50_95']
                # global best_prec1
                # best_prec1 = checkpoint['best_prec1']
                state_dict = checkpoint['state_dict']
                state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
                model.load_state_dict(state_dict)
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        resume()

    # Data loading code
    from cocodataset import COCODataset
    train_dataset = COCODataset('COCO', name='train2017', img_size=608, is_train=True)
    val_dataset = COCODataset('COCO', name='val2017', img_size=416, is_train=False)

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # collate_fn = lambda b: fast_collate(b, memory_format)
    collate_fn = torch.utils.data.default_collate

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn)

    # conf_thresh = cfg['TEST']['CONFTHRE']
    conf_thresh = 0.005
    # conf_thresh = 0.5
    nms_thresh = cfg['TEST']['NMSTHRE']
    if args.evaluate and args.local_rank == 0:
        print("Begin evaluating ...")
        # ap50_95, ap50 = evaluator.evaluate(model)
        validate(val_loader, model, conf_thresh, nms_thresh)
        return

    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # pytorch-accurate time

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            # evaluate on validation set
            print("Begin evaluating ...")
            ap50_95, ap50 = validate(val_loader, model, conf_thresh, nms_thresh)

            # remember best prec@1 and save checkpoint
            is_best = ap50 > best_ap50
            if is_best:
                best_ap50 = ap50
                best_ap50_95 = ap50_95
            save_checkpoint({
                'epoch': epoch + 1,
                'ap50': ap50,
                'ap50_95': ap50_95,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                # 'best_prec1': best_prec1,
                'best_ap50': best_ap50,
                'best_ap50_95': best_ap50_95,
                'optimizer': optimizer.state_dict(),
            }, is_best)

        if torch.cuda.is_available():
            torch.cuda.synchronize()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    assert hasattr(train_loader.dataset, 'set_img_size')
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        if args.prof >= 0 and i == args.prof:
            print("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        current_lr = adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        # compute output
        if args.prof >= 0: torch.cuda.nvtx.range_push("forward")
        output = model(input)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        # for param in model.parameters():
        #     print(param.data.double().sum().item(), param.grad.data.double().sum().item())

        if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
        optimizer.step()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if i % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                # prec1 = reduce_tensor(prec1)
                # prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            # top1.update(to_python_float(prec1), input.size(0))
            # top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            img_size = train_loader.dataset.get_img_size()
            if args.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Lr {5:.8f}\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'ImgSize: {6}x{6}\t'.format(
                    epoch, i, len(train_loader),
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    current_lr,
                    img_size,
                    batch_time=batch_time,
                    loss=losses))

            # 每隔10轮都重新指定输入图像大小
            img_size = (random.randint(0, 9) % 10 + 10) * 32
            train_loader.dataset.set_img_size(img_size)
        # if args.prof >= 0: torch.cuda.nvtx.range_push("prefetcher.next()")
        # input, target = prefetcher.next()
        # if args.prof >= 0: torch.cuda.nvtx.range_pop()

        # Pop range "Body of iteration {}".format(i)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0 and i == args.prof + 10:
            print("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()


def validate(val_loader, model, conf_threshold, nms_threshold):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    ids = list()
    data_list = list()
    # 数据类型
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    end = time.time()
    for i, (img, target) in enumerate(tqdm(val_loader)):
        assert isinstance(target, dict)
        img_info = [x.cpu().item() for x in target['img_info']]
        id_ = img_info[-2]

        # 从这里也判断出是单个推理
        id_ = int(id_)
        # 将原始图像下标挨个保存
        ids.append(id_)
        with torch.no_grad():
            # Numpy Ndarray -> Torch Tensor
            img = Variable(img.type(Tensor))
            # 模型推理，返回预测结果
            # img: [B, 3, 416, 416]
            outputs = model(img)
        # 后处理，进行置信度阈值过滤 + NMS阈值过滤
        # 输入outputs: [B, 预测框数目, 85(xywh + obj_confg + num_classes)]
        # 输出outputs: [B, 过滤后的预测框数目, 7(xyxy + obj_conf + cls_conf + cls_id)]
        outputs = postprocess(outputs, 80, conf_threshold, nms_threshold)
        # 从这里也可以看出是单张推理
        # 如果结果为空，那么不执行后续运算
        if outputs[0] is None:
            continue
        # 提取单张图片的运行结果
        # outputs: [N_ind, 7]
        outputs = outputs[0].cpu().data

        for output in outputs:
            x1 = float(output[0])
            y1 = float(output[1])
            x2 = float(output[2])
            y2 = float(output[3])
            # 分类标签
            label = val_loader.dataset.class_ids[int(output[6])]
            # 转换到原始图像边界框坐标
            box = yolobox2label((y1, x1, y2, x2), img_info[:6])
            # [y1, x1, y2, x2] -> [x1, y1, w, h]
            bbox = [box[1], box[0], box[3] - box[1], box[2] - box[0]]
            # 置信度 = 目标置信度 * 分类置信度
            score = float(output[4].data.item() * output[5].data.item())  # object score * class score
            # 保存计算结果
            A = {"image_id": id_, "category_id": label, "bbox": bbox,
                 "score": score, "segmentation": []}  # COCO json format
            data_list.append(A)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time))

    annType = ['segm', 'bbox', 'keypoints']

    # 计算完成所有测试图像的预测结果后
    # Evaluate the Dt (detection) json comparing with the ground truth
    if len(data_list) > 0:
        cocoGt = val_loader.dataset.coco
        # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
        _, tmp = tempfile.mkstemp()
        json.dump(data_list, open(tmp, 'w'))
        cocoDt = cocoGt.loadRes(tmp)
        cocoEval = COCOeval(val_loader.dataset.coco, cocoDt, annType[1])
        cocoEval.params.imgIds = ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        # AP50_95, AP50
        return cocoEval.stats[0], cocoEval.stats[1]
    else:
        return 0, 0


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30
    # factor = epoch // 10

    if epoch >= 80:
        # if epoch >= 27:
        factor = factor + 1

    lr = args.lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    # if(args.local_rank == 0):
    #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    main()
