import copy

from __init__ import *


def go_predict(data_test_path, data_csv_path, save_path, model_path, dict_id_path, train_csv_train_path,
               w, h, num_workers, batch_size, backbone, data_train_path,
               backbone_1=None, backbone_2=None,
               model_path_1=None, model_path_2=None,
               dict_id_path_1=None, dict_id_path_2=None,
               train_csv_train_path_1=None, train_csv_train_path_2=None,
               Feature_train_path=None, target_train_path=None,
               Feature_test_path=None, target_test_path=None):
    with torch.no_grad():
        print(torch.cuda.is_available())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 建立全局字典
        train_csv = pd.read_csv(data_csv_path)
        train_csv_id = train_csv['individual_id'].unique()
        dict_id_all = dict(zip(train_csv_id, range(len(train_csv_id))))
        new_d_all = {v: k for k, v in dict_id_all.items()}

        model, dict_id, Feature_train, target_train = get_pre_need(model_path, dict_id_path, train_csv_train_path,
                                                                   device, w, h,
                                                                   data_train_path, batch_size,
                                                                   num_workers, save_path, backbone, Feature_train_path, target_train_path)
        model.eval()
        if model_path_1 is not None:
            model_1, dict_id_1, Feature_train_1, target_train_1 = get_pre_need(model_path_1, dict_id_path_1,
                                                                               train_csv_train_path_1, device, w, h,
                                                                               data_train_path, batch_size,
                                                                               num_workers, backbone_1)
            model_1.eval()
        if model_path_2 is not None:
            model_2, dict_id_2, Feature_train_2, target_train_2 = get_pre_need(model_path_2, dict_id_path_2,
                                                                               train_csv_train_path_2, device, w, h,
                                                                               data_train_path, batch_size,
                                                                               num_workers, backbone_2)
            model_2.eval()
        # 获取各个总类中心点
        Feature_train_num = np.zeros([len(dict_id), 512])
        for item in range(len(dict_id)):
            Feature_train_num[item] = np.mean(Feature_train[target_train[:, 0] == item, :], axis=0)
        if model_path_1 is not None:
            Feature_train_num_1 = np.zeros([len(dict_id_1), 512])
            for item in range(len(dict_id_1)):
                Feature_train_num_1[item] = np.mean(Feature_train_1[target_train_1[:, 0] == item, :], axis=0)
        else:
            Feature_train_num_1 = None
        if model_path_2 is not None:
            Feature_train_num_2 = np.zeros([len(dict_id_2), 512])
            for item in range(len(dict_id_2)):
                Feature_train_num_2[item] = np.mean(Feature_train_2[target_train_2[:, 0] == item, :], axis=0)
        else:
            Feature_train_num_2 = None

        path_list = os.listdir(data_test_path)
        # 建立test_dataloader的csv文件
        submission = pd.DataFrame(columns=['image', 'predictions'])
        for item in range(len(path_list)):
            submission.loc[item, "image"] = path_list[item]
        # 建立测试集地址字典
        dict_id_test = dict(zip(path_list, range(len(path_list))))
        new_d_test = {v: k for k, v in dict_id_test.items()}

        test_dataset = TestDataset(submission, dict_id_test, data_test_path, w, h)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers)

        # Feature_test, target_test = get_feature(model, test_dataloader, device, 512)
        if Feature_test_path is None:
            Feature_test, target_test = get_feature(model, test_dataloader, device, 512)
            # target_test = target_test.cpu().detach().numpy()
            # Feature_test = Feature_test.cpu().detach().numpy()
            np.save(os.path.join(save_path, "Feature_test.npy"), Feature_test.cpu().detach().numpy())
            np.save(os.path.join(save_path, "target_test.npy"), target_test.cpu().detach().numpy())
        else:
            Feature_test = torch.from_numpy(np.load(Feature_test_path))
            target_test = torch.from_numpy(np.load(target_test_path))
        if model_path_1 is not None:
            Feature_test_1, target_test_1 = get_feature(model_1, test_dataloader, device, 512)
        else:
            Feature_test_1 = None
        if model_path_2 is not None:
            Feature_test_2, target_test_2 = get_feature(model_2, test_dataloader, device, 512)
        else:
            Feature_test_2 = None

        target_test = target_test.cpu().detach().numpy()
        Top_all = np.zeros([len(path_list), 5])
        Top_index_all = np.zeros([len(path_list), 5])
        with tqdm(total=target_test.shape[0], postfix=dict) as pbar:
            for item in range(target_test.shape[0]):
                # Top, Top_index = get_pre(Feature_test[item, :], Feature_train, target_train, dict_id,
                #                          dict_id_all,
                #                          4, device)
                if model_path_1 is not None and model_path_2 is not None:
                    Top, Top_index = get_pre_num(Feature_test[item, :], Feature_train_num, dict_id, dict_id_all, 5,
                                                 device,
                                                 Feature_test_1[item, :], Feature_train_num_1,
                                                 Feature_test_2[item, :], Feature_train_num_2)
                elif model_path_1 is not None:
                    Top, Top_index = get_pre_num(Feature_test[item, :], Feature_train_num, dict_id, dict_id_all, 5,
                                                 device, Feature_test_1[item, :], Feature_train_num_1)
                elif model_path_2 is not None:
                    Top, Top_index = get_pre_num(Feature_test[item, :], Feature_train_num, dict_id, dict_id_all, 5,
                                                 device,
                                                 feature_test_2=Feature_test_2[item, :],
                                                 Feature_train_num_2=Feature_train_num_2)
                else:
                    Top, Top_index = get_pre_num(Feature_test[item, :], Feature_train_num, dict_id, dict_id_all, 5,
                                                 device)
                Top_all[item, :] = copy.copy(Top)
                Top_index_all[item, :] = copy.copy(Top_index)
                pbar.update(1)
        np.save(os.path.join(save_path, "Top.npy"), Top_all)
        np.save(os.path.join(save_path, "Top_index.npy"), Top_index_all)
        with tqdm(total=target_test.shape[0], postfix=dict) as pbar2:
            New_data = Top_all[np.argsort(Top_all[:, 0])[math.floor(0.12 * len(path_list))], 0]
            # Is_new = 'new_whale' if Top[0] < 0.75 else new_d_all[Top_index[4]]
            for item in range(len(path_list)):
                # submission.loc[
                #     submission[
                #         submission.image == new_d_test[target_test[item, 0]]].index.tolist(), "predictions"] = \
                #     new_d_all[Top_index_all[item, 0]] + ' ' + new_d_all[Top_index_all[item, 1]] + ' ' + new_d_all[
                #         Top_index_all[item, 2]] + ' ' + new_d_all[Top_index_all[item, 3]] + ' ' + new_d_all[
                #         Top_index_all[item, 4]]
                if Top_all[item, 0] <= New_data:
                    submission.loc[
                        submission[
                            submission.image == new_d_test[target_test[item, 0]]].index.tolist(), "predictions"] = \
                        'new_individual' + ' ' + new_d_all[Top_index_all[item, 0]] + ' ' + new_d_all[
                            Top_index_all[item, 1]] + ' ' + new_d_all[Top_index_all[item, 2]] + ' ' + new_d_all[Top_index_all[item, 3]]
                else:
                    submission.loc[
                        submission[
                            submission.image == new_d_test[target_test[item, 0]]].index.tolist(), "predictions"] = \
                        new_d_all[Top_index_all[item, 0]] + ' ' + new_d_all[Top_index_all[item, 1]] + ' ' + new_d_all[
                            Top_index_all[item, 2]] + ' ' + new_d_all[Top_index_all[item, 3]] + ' ' + new_d_all[Top_index_all[item, 4]]
                pbar2.update(1)
                pbar2.set_postfix(
                    **{'Top': Top_all[item, 0], 'new': New_data})
        submission.to_csv(os.path.join(save_path, "submission.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练训练参数设置')
    parser.add_argument('--backbone', type=str, default='resnet50', help='特征网络选择，默认resnet50', required=True)
    parser.add_argument('--backbone_1', type=str, default=None, help='特征网络选择，默认resnet101')
    parser.add_argument('--backbone_2', type=str, default=None, help='特征网络选择，默认resnet152')
    parser.add_argument('--dict_id_path', type=str, help='字典路径', required=True)
    parser.add_argument('--dict_id_path_1', type=str, help='字典路径', default=None)
    parser.add_argument('--dict_id_path_2', type=str, help='字典路径', default=None)
    parser.add_argument('--model_path', type=str, help='模型路径', required=True)
    parser.add_argument('--model_path_1', type=str, help='模型路径', default=None)
    parser.add_argument('--model_path_2', type=str, help='模型路径', default=None)
    parser.add_argument('--train_csv_train_path', type=str, help='本次训练csv路径', required=True)
    parser.add_argument('--train_csv_train_path_1', type=str, help='本次训练csv路径', default=None)
    parser.add_argument('--train_csv_train_path_2', type=str, help='本次训练csv路径', default=None)
    parser.add_argument('--Feature_train_path', type=str, help='训练集特征矩阵路径', default=None)
    parser.add_argument('--target_train_path', type=str, help='训练集标签矩阵路径', default=None)
    parser.add_argument('--Feature_test_path', type=str, help='测试集特征矩阵路径', default=None)
    parser.add_argument('--target_test_path', type=str, help='测试集标签矩阵路径', default=None)
    parser.add_argument('--data_test_path', type=str, help='测试集路径', required=True)
    parser.add_argument('--data_train_path', type=str, help='训练集路径', default="../input/data-do-cut/All/All")
    parser.add_argument('--data_csv_path', type=str, help='训练集csv路径',
                        default=r'../input/happy-whale-and-dolphin/train.csv')
    parser.add_argument('--save_path', type=str, help='存储路径', default=r'./')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--w', type=int, help='训练图片宽度', default=224)
    parser.add_argument('--h', type=int, help='训练图片高度', default=224)
    args = parser.parse_args()

    go_predict(backbone=args.backbone,
               backbone_1=args.backbone_1,
               backbone_2=args.backbone_2,
               dict_id_path=args.dict_id_path,
               dict_id_path_1=args.dict_id_path_1,
               dict_id_path_2=args.dict_id_path_2,
               train_csv_train_path=args.train_csv_train_path,
               train_csv_train_path_1=args.train_csv_train_path_1,
               train_csv_train_path_2=args.train_csv_train_path_2,
               save_path=args.save_path,
               data_test_path=args.data_test_path,
               data_csv_path=args.data_csv_path,
               model_path=args.model_path,
               num_workers=args.num_workers,
               batch_size=args.batch_size,
               data_train_path=args.data_train_path,
               Feature_train_path=args.Feature_train_path,
               target_train_path=args.target_train_path,
               Feature_test_path=args.Feature_test_path,
               target_test_path=args.target_test_path,
               w=args.w,
               h=args.h)
    # # -------------------------------#
    # #   路径设置
    # # -------------------------------#
    # data_test_path = r'../input/unet-test/unet_test'
    # data_csv_path = r'../input/humpback-whale-identification/train.csv'
    # save_path = r'./'
    # path = r'../input/arc-epoth-3'
    # # -------------------------------#
    # #   dataloader设置
    # # -------------------------------#
    # w = 256
    # h = 256
    # num_workers = 2
    # batch_size = 128
    # # -------------------------------#
    # #   开始预测
    # # -------------------------------#
