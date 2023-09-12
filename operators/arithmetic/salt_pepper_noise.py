#!/usr/bin/python3
# -*- coding:utf-8 -*-

"""
author：yannan1
since：2023-08-22
"""
import random

import torch
from imgaug import augmenters as iaa


def imgaug_example(image, p=0.03):
    """
    args:
        image: np.ndarray
        p: 将像素替换为椒盐噪声的概率。 取值范围(0.0, 0.03)
    """
    augmenter = iaa.SaltAndPepper(p=p)
    image = augmenter(image=image)
    return image


def torch_example(image, p=0.03):
    """
    args:
        image: torch.Tensor
        p: 将像素替换为椒盐噪声的概率, 取值范围(0.0, 0.03)
    """

    _, imh, imw = image.shape
    num_salt = int(p * imh * imw)

    num_salt_0 = random.randint(0, num_salt)
    num_salt_255 = num_salt - num_salt_0

    coords1 = [torch.randint(low=0, high=i - 1, size=(num_salt_0,)) for i in [imh, imw]]
    image[:, coords1[0], coords1[1]] = 0

    coords2 = [torch.randint(low=0, high=i - 1, size=(num_salt_255,)) for i in [imh, imw]]
    image[:, coords2[0], coords2[1]] = 255

    return image






