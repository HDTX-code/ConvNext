import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.Canny import get_img


def get_pre(feature_test, Feature_train, target_train, dict_id, dict_id_all, it, device):
    Feature_train = torch.from_numpy(Feature_train).to(device)
    target_train = torch.from_numpy(target_train).to(device)
    feature_test = feature_test.to(device)

    new_d = {v: k for k, v in dict_id.items()}
    num = len(dict_id)

    with torch.no_grad():
        output = F.cosine_similarity(
            torch.mul(torch.ones(Feature_train.shape).to(device), feature_test.T),
            Feature_train, dim=1).to(device)
        kind = torch.zeros([num]).to(device)
        for j in range(num):
            kind[j] = output[target_train[:, 0] == j].mean().to(device)
        sorted, indices = torch.sort(kind, descending=True)
        sorted = sorted.cpu().detach().numpy()
        indices = indices.cpu().detach().numpy()
        Top = sorted[:it]
        Top_index = indices[:it]
        for item in range(it):
            Top_index[item] = dict_id_all[new_d[Top_index[item]]]
        return Top, Top_index
