import torch
import torch.nn.functional as F


def get_pre_num(feature_test, Feature_train_num, dict_id, dict_id_all, it, device,
                feature_test_1=None, Feature_train_num_1=None,
                feature_test_2=None, Feature_train_num_2=None):
    Feature_train_num = torch.from_numpy(Feature_train_num).to(device)
    feature_test = feature_test.to(device)
    if feature_test_1 is not None:
        Feature_train_num_1 = torch.from_numpy(Feature_train_num_1).to(device)
        feature_test_1 = feature_test_1.to(device)
    if feature_test_2 is not None:
        Feature_train_num_2 = torch.from_numpy(Feature_train_num_2).to(device)
        feature_test_2 = feature_test_2.to(device)
    new_d = {v: k for k, v in dict_id.items()}

    with torch.no_grad():
        output = F.cosine_similarity(
            torch.mul(torch.ones(Feature_train_num.shape).to(device), feature_test.T),
            Feature_train_num, dim=1).to(device)
        if feature_test_1 is not None:
            output_1 = F.cosine_similarity(
                    torch.mul(torch.ones(Feature_train_num_1.shape).to(device), feature_test_1.T),
                    Feature_train_num_1, dim=1).to(device)
            output = output_1 + output
        if feature_test_2 is not None:
            output_2 = F.cosine_similarity(
                torch.mul(torch.ones(Feature_train_num_2.shape).to(device), feature_test_2.T),
                Feature_train_num_2, dim=1).to(device)
            output = output_2 + output
        sorted, indices = torch.sort(output, descending=True)
        sorted = sorted.cpu().detach().numpy()
        indices = indices.cpu().detach().numpy()
        Top = sorted[:it]
        Top_index = indices[:it]
        for item in range(it):
            Top_index[item] = dict_id_all[new_d[Top_index[item]]]
        return Top, Top_index
