import math

import cv2
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import confusion_matrix
from torch.nn import functional as func


def change_size(img, desired_size, mode):
    old_size = img.size
    delta_w = desired_size - old_size[0]
    delta_h = desired_size - old_size[1]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    if mode == 'RGB':
        color = [0, 0, 0]
    else:    # 'L'
        color = [0]
    img = np.array(img)
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=color)
    new_img = Image.fromarray(new_img)
    return new_img

# 结合张量
def concat_tensor(tensor1, tensor2):
    diff_y = tensor2.size()[2] - tensor1.size()[2]
    diff_x = tensor2.size()[3] - tensor1.size()[3]
    # padding_left, padding_right, padding_top, padding_bottom
    tensor1 = func.pad(tensor1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
    # t = torch.cat((tensor1, crop_img(tensor2, tensor1)), dim=1)
    t = torch.cat((tensor1, tensor2), dim=1)
    return t

# 裁剪操作
def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2] # 64
    tensor_size = tensor.size()[2]  # 56
    delta = tensor_size - target_size  # 8
    delta = delta // 2  # 4
    # delta:tensor_size-delta表示切片 即原本宽为64，取4~60部分，即变成宽为56
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


# def crop_img_to_slice(img, ima_name):
#     w = img.size[0] // 3
#     h = img.size[1] // 3
#     for j in range(3):
#         for i in range(3):
#             box = (w * i, h * j, w * (i + 1), h * (j + 1))
#             # （j, i） 左上右下
#             region = img.crop(box)  # 进行裁剪
#             # region.save('{}{}.png'.format(j, i))
#             region.show()

