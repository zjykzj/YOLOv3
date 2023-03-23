# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Export a YOLOv5 PyTorch model to onnx formats.

Format                      | `export.py`                   | Model
---                         | ---                           | ---
ONNX                        | `onnx`                        | yolov5s.onnx

Requirements:
    $ pip install -r requirements.txt onnx onnx-simplifier onnxruntime  # CPU
    $ pip install -r requirements.txt onnx onnx-simplifier onnxruntime-gpu  # GPU

Usage:
    $ python export.py --weights yolov5s.pt　...

Inference:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
"""

import argparse
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.yolo import ClassificationModel, Detect, DetectionModel, SegmentationModel
from utils.general import (LOGGER, Profile, check_img_size, check_requirements,
                           colorstr, file_size, get_default_args, print_args, url2file)
from utils.torch_utils import select_device, smart_inference_mode


def try_export(inner_func):
    # YOLOv5 export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args['prefix']
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f'{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)')
            return f, model
        except Exception as e:
            LOGGER.info(f'{prefix} export failure ❌ {dt.t:.1f}s: {e}')
            return None, None

    return outer_func


@try_export
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    check_requirements('onnx>=1.12.0')
    import onnx

    LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
    f = file.with_suffix('.onnx')

    output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output0']
    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
        if isinstance(model, SegmentationModel):
            dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
            dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
        elif isinstance(model, DetectionModel):
            dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1'))
            import onnxsim

            LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f'{prefix} simplifier failure: {e}')
    return f, model_onnx


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # FP16 half-precision export
        inplace=False,  # set YOLOv5 Detect() inplace=True
        optimize=False,  # TorchScript: optimize for mobile
        dynamic=False,  # ONNX/TF/TensorRT: dynamic axes
        simplify=False,  # ONNX: simplify model
        opset=12,  # ONNX: opset version
):
    t = time.time()
    file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)
    if half:
        assert device.type != 'cpu', '--half only compatible with GPU export, i.e. use --device 0'
        assert not dynamic, '--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both'
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    if optimize:
        assert device.type == 'cpu', '--optimize not compatible with cuda devices, i.e. use --device cpu'

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True

    for _ in range(2):
        y = model(im)  # dry runs
    if half:
        im, model = im.half(), model.half()  # to FP16
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

    # Exports
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    f, _ = export_onnx(model, im, file, opset, dynamic, simplify)

    # Finish
    cls, det, seg = (isinstance(model, x) for x in (ClassificationModel, DetectionModel, SegmentationModel))  # type
    det &= not seg  # segmentation models inherit from SegmentationModel(DetectionModel)
    dir = Path('segment' if seg else 'classify' if cls else '')
    h = '--half' if half else ''  # --half FP16 inference arg
    s = '# WARNING ⚠️ ClassificationModel not yet supported for PyTorch Hub AutoShape inference' if cls else \
        '# WARNING ⚠️ SegmentationModel not yet supported for PyTorch Hub AutoShape inference' if seg else ''
    LOGGER.info(f'\nExport complete ({time.time() - t:.1f}s)'
                f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                f"\nDetect:          python {dir / ('detect.py' if det else 'predict.py')} --weights {f} {h}"
                f"\nValidate:        python {dir / 'val.py'} --weights {f} {h}"
                f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f}')  {s}"
                f'\nVisualize:       https://netron.app')


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF/TensorRT: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=17, help='ONNX: opset version')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
