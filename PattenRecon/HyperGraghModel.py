import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class get_model(nn.Module):
    def __init__(self, k=138 * 8, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:  # 每个点是6维还是3维
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, k)
        self.fcequal = nn.Linear(256, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn2(self.dropout(self.fcequal(x))))
        x = F.relu(self.bn2(self.dropout(self.fcequal(x))))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        x = self.fc4(x)
        # x = F.log_softmax(x, dim=1)  #可能需要去掉
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
