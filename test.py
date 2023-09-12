#!/usr/bin/python3
# -*- coding:utf-8 -*-

"""
author：yannan1
since：2023-08-22
"""
import cv2
import torch
import matplotlib.pyplot as plt

import operators

operator_name = 'horizontal_flip'
operator = getattr(operators, operator_name)

if __name__ == '__main__':
    img_path = '../data/dog-cat.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_img = torch.from_numpy(img).to(device)
    torch_img = torch_img.permute((2, 0, 1)).float()

    imgaug_out = operator.imgaug_example(img)
    torch_out = operator.torch_example(torch_img)

    torch_out = torch_out.permute((1, 2, 0)).cpu().numpy().astype('uint8')

    cv2.imwrite('salt_pepper_noise-imgaug.jpg', cv2.cvtColor(imgaug_out, cv2.COLOR_RGB2BGR))
    cv2.imwrite('salt_pepper_noise-torch.jpg', cv2.cvtColor(torch_out, cv2.COLOR_RGB2BGR))

    # fig, axes = plt.subplots(1, 3)
    # axes[0].imshow(img)
    # axes[0].axis('off')
    # # axes[0].set_title('原图')
    #
    # axes[1].imshow(imgaug_out)
    # axes[1].axis('off')
    # axes[1].text(0.5, -0.15, 'imgaug', transform=axes[1].transAxes,
    #              fontsize=12, ha='center')
    #
    # axes[2].imshow(torch_out)
    # axes[2].axis('off')
    # axes[1].text(0.5, -0.15, 'torch', transform=axes[2].transAxes,
    #              fontsize=12, ha='center')
    #
    # plt.subplots_adjust(wspace=0.02)
    # plt.show()
