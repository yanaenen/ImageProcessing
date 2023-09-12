#!/usr/bin/python3
# -*- coding:utf-8 -*-

"""
author：yannan1
since：2023-09-01
"""
import torch
from imgaug import augmenters as iaa
from torchvision.transforms import functional as TF


def imgaug_example(image):
    augmenter = iaa.Flipud(p=1.0)
    image = augmenter(image=image)
    return image


def torch_example(image):
    return TF.vflip(image)


def hand_coding(image):
    imh = image.shape[-2]
    index = torch.arange(imh - 1, -1, -1)  # 创建索引张量
    return image[:, index, :]


class VerticalFlip:

    def __init__(self) -> None:
        """VerticalFlip, 垂直翻转
        """
        super().__init__()

    @staticmethod
    def _augment_image(image):
        return TF.vflip(image)

    @staticmethod
    def _augment_bboxes(bboxes, imh):
        bboxes[:, [1, 3]] = imh - bboxes[:, [3, 1]]
        return bboxes

    @staticmethod
    def _augment_keypoints(keypoints, imh):
        keypoints[:, 1] = imh - keypoints[:, 1]
        return keypoints

    def __call__(self, doc):
        """
        doc = {
            'image': torch.Tensor,  # C*H*W
            'mask': torch.Tensor,  # C*H*W
            'bboxes': torch.Tensor,  # [[x1, y1, x2, y2, label], ...]
            'heatmap': torch.Tensor,  # C*H*W
            'keypoints': torch.Tensor,  # [[x1, y1], [x2, y2], ...]
            'shape': torch.Size,  # image's shape [C, H, W]
        }
        """
        image = doc.get('image')
        if image is None:
            return doc

        doc['image'] = self._augment_image(image)

        bboxes = doc.get('bboxes')
        if bboxes is not None:
            imh = doc['shape'][-2]
            doc['bboxes'] = self._augment_bboxes(bboxes, imh)

        mask = doc.get('mask')
        if mask is not None:
            doc['mask'] = self._augment_image(mask)

        heatmap = doc.get('heatmap')
        if heatmap is not None:
            doc['heatmap'] = self._augment_image(heatmap)

        keypoints = doc.get('keypoints')
        if keypoints is not None:
            imh = doc['shape'][-2]
            doc['keypoints'] = self._augment_keypoints(keypoints, imh)

        return doc


if __name__ == '__main__':
    image = torch.tensor([[
        [1, 2, 3, 4, 5, 11],
        [6, 7, 8, 9, 10, 12]
    ]])
    print(torch_example(image))
