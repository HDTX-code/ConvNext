import os

import torch


def save_model(model, save_path, name, iter_cnt, loss, score):
    save_name = os.path.join(save_path,
                             name + '-' + str(iter_cnt) + 'loss: ' + str(loss) + 'score: ' + str(score) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name
