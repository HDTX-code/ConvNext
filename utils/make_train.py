import os.path
import sys

import numpy as np
import torch
from tqdm import tqdm

from utils.get_feature import get_feature
from utils.make_val import make_val
from utils.save_model import save_model
from utils.utils import get_lr


def make_train(model, metric_fc, criterion, optimizer, scheduler,
               train_loader, device, Str, num_classes,
               max_epoch, save_interval, save_path, backbone, epoch_start, epoch_end, Freeze_Epoch, val_loader=None):
    for item in range(epoch_start, epoch_end + 1):
        with tqdm(total=(len(train_loader)), desc=f'Epoch {item}/{max_epoch}', postfix=dict) as pbar:
            # 开始训练
            model = model.train()
            model.to(device)

            Loss = 0

            # 训练
            for iteration, (image_tensor, target_t) in enumerate(train_loader):
                image_tensor = image_tensor.type(torch.FloatTensor).to(device)
                feature = model(image_tensor).to(device)
                output = metric_fc(feature, target_t).to(device)
                loss = criterion(output.reshape(-1, num_classes).to(device),
                                 target_t.reshape(-1).long().to(device)).to(device)
                Loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(**{'loss_{}'.format(Str): loss.item(), 'lr': get_lr(optimizer)})
                pbar.update(1)
                # if iteration >= 1:
                #     break
            scheduler.step()

        with torch.no_grad():
            model = model.eval()
            model.to(device)

            metric_fc = metric_fc.eval()
            metric_fc.to(device)

            Loss = Loss.cpu().detach().numpy() / len(train_loader)

            print("第{}轮 : Loss_{} = {}".format(item, Str, Loss))

            if (item % save_interval == 0 or item == max_epoch) and item > Freeze_Epoch:
                # 开始验证，获取特征矩阵
                if item == max_epoch:
                    Feature_train, target_train = get_feature(model, train_loader, device, 512)
                    path_featureMap = os.path.join(save_path, "FeatureMap")
                    if not os.path.exists(path_featureMap):
                        os.mkdir(path_featureMap)
                    Feature_train = Feature_train.cpu().detach().numpy()
                    target_train = target_train.cpu().detach().numpy()
                    np.save(os.path.join(path_featureMap, "Feature_train_{}.npy".format(item)), Feature_train)
                    np.save(os.path.join(path_featureMap, "target_train_{}.npy".format(item)), target_train)
                if val_loader is not None:
                    # 计算验证得分
                    Feature_val, target_val = get_feature(model, val_loader, device, 512)
                    Score = make_val(Feature_train, target_train, Feature_val, target_val, device, num_classes)
                else:
                    Score = 0
                path_model = os.path.join(save_path, "model")
                if not os.path.exists(path_model):
                    os.mkdir(path_model)
                save_model(model, path_model, str(backbone) + Str, item, Loss, Score)
                # Feature_val = Feature_val.cpu().detach().numpy()
                # target_val = target_val.cpu().detach().numpy()
                # np.save(os.path.join(path_featureMap, "Feature_val_{}.npy".format(epoch_now)), Feature_val)
                # np.save(os.path.join(path_featureMap, "target_val_{}.npy".format(epoch_now)), target_val)
                # print("第{}轮 : Score={}".format(i, Score))
        # if i >= 1:
        #     break
    return model
