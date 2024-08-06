import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from HG_util import HyperGraphEncoder



class get_model(nn.Module):
    def __init__(self, K_patch=138 * 8, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:  # 每个点是6维还是3维
            channel = 6
        else:
            channel = 3
        self.feat = HyperGraphEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, k)
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(512)
        self.K_patch = K_patch



    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc4(x)
        x = x.reshape(-1, 8, 138)
        return x, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target):
        target.transpose(2, 1)
        # print(f'Shape:{pred.shape},{target.shape}')
        loss = F.mse_loss(pred, target)
        return loss
