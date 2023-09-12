#!/usr/bin/python3
# -*- coding:utf-8 -*-

"""
author：yannan1
since：2023-08-28
"""
import torch
from torch import nn
from torchvision import transforms
from imgaug import augmenters as iaa


def imgaug_example(image, kernel_size=3):
    """
    args:
        image: np.ndarray
        kernel_size: 核数, 取值范围(1, 7)
    """
    augmenter = iaa.AverageBlur(k=kernel_size)
    image = augmenter(image=image)
    return image


def torch_example(image, kernel_size=3):
    """
    args:
        image: torch.Tensor
        kernel_size: 核数, 取值范围(1, 7)
    """

    padding = int((kernel_size - 1) / 2)
    image = transforms.Pad(padding=padding, padding_mode='symmetric')(image)
    image = nn.AvgPool2d(kernel_size, stride=1)(image)

    return image


if __name__ == '__main__':
    a = torch.tensor([
        [[1., 2., 3., 4., 5.],
         [6., 7., 8., 9., 10.],
         [11., 12., 13., 14., 15.],
         [16., 17., 18., 19., 20.],
         [21., 22., 23., 24., 25.]]])

    torch_example(a, kernel_size=5)


