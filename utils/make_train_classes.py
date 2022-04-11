import os

import numpy as np
import torch
from tqdm import tqdm

from utils.get_feature import get_feature
from utils.utils import get_lr


def fit_one_epoch_classes(model, criterion, optimizer, item, max_epoch,
                          Freeze_Epoch, train_loader, device,
                          save_interval, save_path, backbone, num_classes):
    with tqdm(total=(len(train_loader)), desc=f'Epoch {item}/{max_epoch}', postfix=dict) as pbar:
        # 开始训练
        model = model.train()
        model.to(device)

        Loss = 0

        # 训练
        for iteration, (image_tensor, target_t) in enumerate(train_loader):
            image_tensor = image_tensor.type(torch.FloatTensor).to(device)
            target_t = target_t.long().to(device)
            feature = model(image_tensor).to(device)
            loss = criterion(feature, target_t.reshape(-1)).to(device)
            Loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(**{'loss': loss.item(), 'lr': get_lr(optimizer)})
            pbar.update(1)
    with torch.no_grad():
        model = model.eval()
        model.to(device)
        Loss = Loss.cpu().detach().numpy() / len(train_loader)

        print("第{}轮 : Loss = {}".format(item, Loss))

        if (item % save_interval == 0 or item == max_epoch) and item > Freeze_Epoch:
            # 开始验证，获取特征矩阵
            # Feature_train, target_train = get_feature(model, train_loader, device, num_classes)
            # path_featureMap = os.path.join(save_path, "FeatureMap")
            # if not os.path.exists(path_featureMap):
            #     os.mkdir(path_featureMap)
            # Feature_train = Feature_train.cpu().detach().numpy()
            # target_train = target_train.cpu().detach().numpy()
            # np.save(os.path.join(path_featureMap, "Feature_train_{}.npy".format(item)), Feature_train)
            # np.save(os.path.join(path_featureMap, "target_train_{}.npy".format(item)), target_train)
            path_model = os.path.join(save_path, "model")
            if not os.path.exists(path_model):
                os.mkdir(path_model)
            torch.save(model.state_dict(), backbone + '_epoch:{}'.format(item) + '_loss:{}'.format(Loss))
