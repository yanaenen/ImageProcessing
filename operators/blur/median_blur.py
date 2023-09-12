#!/usr/bin/python3
# -*- coding:utf-8 -*-

"""
author：yannan1
since：2023-08-25
"""
import torch
import kornia as K
import torch.nn.functional as F
from imgaug import augmenters as iaa


def imgaug_example(image, kernel_size=3):
    """
    args:
        image: np.ndarray
        kernel_size: 核数, 取值范围(1, 7)
    """
    augmenter = iaa.MedianBlur(k=kernel_size)
    image = augmenter(image=image)
    return image


def torch_example(image, kernel_size=3):
    """
    args:
        image: torch.Tensor
        kernel_size: 核数, 取值范围(1, 7)
    """

    image = image.unsqueeze(0)
    image = K.filters.median_blur(image, kernel_size).squeeze(0)

    return image


def hand_coding(image, kernel_size=3):
    """
    args:
        image: torch.Tensor
        kernel_size: 核数, 取值范围(1, 7)
    """

    # get_kernel
    device, dtype = image.device, image.dtype
    kernel_num = kernel_size * kernel_size
    kernel = torch.zeros((kernel_num, kernel_num), device=device, dtype=dtype)
    idx = torch.arange(kernel_num, device=device)
    kernel[idx, idx] += 1.0
    kernel = kernel.view(kernel_num, 1, kernel_size, kernel_size)  # [9, 1, 3, 3]

    padding = (kernel_size - 1) // 2
    c, h, w = image.shape

    # features shape = [c, 9, h, w]
    features = F.conv2d(image.reshape(c, 1, h, w), kernel, padding=padding, stride=1)
    print(features)
    # features = features.view(c, -1, h, w)  # features shape = [c, 9, h, w]

    # return shape = [c, h, w]
    return features.median(dim=1)[0]


if __name__ == '__main__':
    # a = torch.ones((3, 5, 5))
    a = torch.tensor([
        [[1., 2., 3., 4., 5.],
         [6., 7., 8., 9., 10.],
         [11., 12., 13., 14., 15.],
         [16., 17., 18., 19., 20.],
         [21., 22., 23., 24., 25.]]])
    res = hand_coding(a)
    print(res)
