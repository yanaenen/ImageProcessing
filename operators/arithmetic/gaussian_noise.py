#!/usr/bin/python3
# -*- coding:utf-8 -*-

"""
author：yannan1
since：2023-08-30
"""

import random

import torch
from imgaug import augmenters as iaa


def imgaug_example(image, mean=0, std=3):
    """
    args:
        image: np.ndarray
        mean: 平均值, 取值范围[0, 1]
        std: 标准差, 取值范围[0, 15]
    """
    augmenter = iaa.AdditiveGaussianNoise(loc=mean, scale=std)
    image = augmenter(image=image)
    return image


def torch_example(image, mean=0, std=3):
    """
    args:
        image: torch.Tensor
        mean: 平均值, 取值范围[0, 1]
        std: 标准差, 取值范围[0, 15]
    """
    shape = image.shape
    gauss = torch.normal(mean=mean, std=std, size=shape).to(image.device)
    noisy_img = image + gauss
    image = torch.clip(noisy_img, min=0, max=255)

    return image

