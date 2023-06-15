# -*- coding: utf-8 -*-

"""
@date: 2023/6/15 下午2:17
@file: transform.py
@author: zj
@description: 
"""

import cv2

import numpy as np


def preprocess(img, imgsize, jitter, random_placing=False):
    """
    Image preprocess for yolo input
    Pad the shorter side of the image and resize to (imgsize, imgsize)
    Args:
        img (numpy.ndarray): input image whose shape is :math:`(H, W, C)`.
            Values range from 0 to 255.
        imgsize (int): target image size after pre-processing
        jitter (float): amplitude of jitter for resizing
        random_placing (bool): if True, place the image at random position
    Returns:
        img (numpy.ndarray): input image whose shape is :math:`(C, imgsize, imgsize)`.
            Values range from 0 to 1.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    """
    h, w, _ = img.shape
    # img = img[:, :, ::-1]
    # assert img is not None

    if jitter > 0:
        # add jitter
        dw = jitter * w
        dh = jitter * h
        new_ar = (w + np.random.uniform(low=-dw, high=dw)) \
                 / (h + np.random.uniform(low=-dh, high=dh))
    else:
        new_ar = w / h

    if new_ar < 1:
        nh = imgsize
        nw = nh * new_ar
    else:
        nw = imgsize
        nh = nw / new_ar
    nw, nh = int(nw), int(nh)

    if random_placing:
        dx = int(np.random.uniform(imgsize - nw))
        dy = int(np.random.uniform(imgsize - nh))
    else:
        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2

    img = cv2.resize(img, (nw, nh))
    sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
    sized[dy:dy + nh, dx:dx + nw, :] = img

    info_img = (h, w, nh, nw, dx, dy)
    return sized, info_img


def t_preprocess():
    # img = cv2.imread("assets/mountain.png")
    img = cv2.imread("assets/coco/bus.jpg")
    cv2.imshow("img", img)

    img_size = 416

    for i in range(10):
        sized, info_img = preprocess(img, img_size, 0.3, True)
        cv2.imshow("sized", sized)
        cv2.waitKey(0)


def t_flip():
    img = cv2.imread("assets/coco/bus.jpg")
    cv2.imshow("img", img)

    flip = cv2.flip(img, 1)
    cv2.imshow("flip 1", flip)
    # cv2.waitKey(0)

    flip = cv2.flip(img, 0)
    cv2.imshow("flip 0", flip)
    cv2.waitKey(0)


def t_color_jitter():
    from yolo.data.transform import color_dithering

    img = cv2.imread("assets/coco/bus.jpg")
    cv2.imshow("img", img)

    for i in range(10):
        dst_img = color_dithering(img, 0.1, 1.5, 1.5).astype(np.uint8)
        cv2.imshow("dst", dst_img)
        cv2.waitKey(0)


if __name__ == '__main__':
    # t_flip()
    t_color_jitter()
