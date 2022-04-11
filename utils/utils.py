import copy
import math
import os
from collections import Counter
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.DataStrength import ImageNew


def ImageRotate(image):
    image = cv2.resize(image, (224, 224))
    height, width = image.shape[:2]  # 输入(H,W,C)，取 H，W 的zhi
    center = (width / 2, height / 2)  # 绕图片中心进行旋转
    angle = random.randint(-180, 180)  # 旋转方向取（-180，180）中的随机整数值，负为逆时针，正为顺势针
    scale = 1  # 将图像缩放为80%

    # 获得旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # 进行仿射变换，边界填充为255，即白色，默认为黑色
    image_rotation = cv2.warpAffine(src=image, M=M, dsize=(height, width), borderValue=(255, 255, 255))

    return image_rotation


def ImageNew(src):
    blur_img = cv2.GaussianBlur(src, (0, 0), 5)
    usm = cv2.addWeighted(src, 1.5, blur_img, -0.5, 0)
    result = usm
    return result


def Image_GaussianBlur(img):
    kernel_size = (5, 5)
    sigma = 1.5
    img = cv2.GaussianBlur(img, kernel_size, sigma)
    return img


def Image_flip_horizontal(img):
    # 水平翻转
    return cv2.flip(img, 1)


def Image_flip_vertical(img):
    # 垂直翻转
    return cv2.flip(img, 0)


def Image_flip_hv(img):
    # 水平加垂直翻转
    return cv2.flip(img, -1)


def get_csv(path, label):
    dir_list = os.listdir(path)
    csv = pd.DataFrame(columns=['image', 'path', 'individual_id'])
    for item in dir_list:
        csv.loc[len(csv)] = [item, os.path.join(path, item), label]
    return csv


def get_img_for_tensor(path, w, h, isNew=False):
    img = cv2.imread(path)
    if isNew:
        img = ImageNew(img)
    img = ImageRotate(img)
    # 改变格式成规定的框和高
    if img.shape[2] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_0 = cv2.resize(img, (w, h))
    # 改变img1的时候不改变img
    img1 = np.zeros([3, h, w])
    # cv2读取的是bgr,转换成rgb就要做一下变通
    img1[0, :, :] = img_0[:, :, 2]
    img1[1, :, :] = img_0[:, :, 1]
    img1[2, :, :] = img_0[:, :, 0]
    return img1


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ---------------------------------------------------#
#   KNN
# ---------------------------------------------------#
def cal_distance(Feature_train, Feature_test, device, K):
    Feature_train = torch.from_numpy(Feature_train).to(device)
    Feature_test = Feature_test.to(device)
    with torch.no_grad():
        with tqdm(total=Feature_test.shape[0]) as pbar:
            val = 0
            for item in range(Feature_test.shape[0]):
                output = F.cosine_similarity(
                    torch.mul(torch.ones(Feature_train.shape).to(device), Feature_test[item, :].T),
                    Feature_train, dim=1).to(device)
                sorted, indices = torch.sort(output, descending=True)
                sorted = sorted.reshape(1, -1)
                indices = indices.reshape(1, -1)
                # output = output.reshape(1, -1)
                if val == 0:
                    Output_score = copy.copy(sorted[:K])
                    Output_index = copy.copy(indices[:K])
                    val = 1
                else:
                    Output_score = torch.cat((Output_score, sorted[:K]), 0)
                    Output_index = torch.cat((Output_index, indices[:K]), 0)
                pbar.update(1)
    return Output_score, Output_index


def KNN_by_iter(Feature_train, target_train, Feature_test, target_test, k, device,
                submission, new_d_test, new_id, save_path, Score_path=None, Index_path=None):
    # 计算距离
    # res = []
    if Score_path is None:
        Score, Index = cal_distance(Feature_train, Feature_test, device, K=10000)
        Score = Score.cpu().detach().numpy()
        Index = Index.cpu().detach().numpy()
        np.save(os.path.join(save_path, "Score.npy"), Score)
        np.save(os.path.join(save_path, "Index.npy"), Index)
    else:
        Score = np.load(Score_path)
        Index = np.load(Index_path)
    with tqdm(total=Feature_test.shape[0]) as pbar:
        for item in range(Feature_test.shape[0]):
            K = copy.copy(k)
            while True:
                target_train_index = target_train[Index[item, :K], 0]
                if len(np.unique(target_train_index)) >= 5:
                    break
                K += 5
            score = Score[item, :K]
            index = np.unique(target_train_index)
            res = np.zeros(len(index))
            for item2 in range(len(index)):
                res[item2] = score[target_train_index == index[item2]].mean()
                # res[item2] = sum(target_train_index == index[item2])
            res_sort = res.argsort()[-5:]
            submission.loc[
                submission[
                    submission.image == new_d_test[target_test[item, 0]]].index.tolist(), "predictions"] = \
                new_id[index[res_sort[-1]]] + ' ' + new_id[index[res_sort[-2]]] + ' ' + new_id[
                    index[res_sort[-3]]] + ' ' \
                + new_id[index[res_sort[-4]]] + ' ' + new_id[index[res_sort[-5]]]
            pbar.update(1)
    submission.to_csv(os.path.join(save_path, "submission.csv"), index=False)


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-1,
                        end_factor=1e-4):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
