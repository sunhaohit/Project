import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
import torch

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        # print((xyz - centroid) ** 2)
        dist = np.sum(np.array((xyz - centroid) ** 2), axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class MyDataset(Dataset):
    def __init__(self, root, split='train'):
        self.split = split
        pointcloud_path = os.path.join(root, 'pointcloud')
        VE_path = os.path.join(root, 'VE')

        ls_pc_fn = os.listdir(pointcloud_path)
        ls_ve_fn = os.listdir(VE_path)

        assert (split == 'train' or split == 'test')
        split_num = int(len(ls_pc_fn) * 0.8)

        ls_pc = []
        for i in ls_pc_fn:
            with open(os.path.join(pointcloud_path, i), 'r') as f:
                data = f.readlines()[10:]
                data = [np.array(i.split(' '), dtype=np.float32) for i in data]
                data = torch.tensor(data)
                # data = farthest_point_sample(data, 1024) #这里没必要采样1024个点，在PointNet的Encoder中会进行采样
                ls_pc.append(data)

        ls_ve = [pickle.load(open(os.path.join(VE_path, i), 'rb')) for i in ls_ve_fn]

        pc, ve = torch.stack(ls_pc, dim=0), torch.stack(ls_ve, dim=0)

        # print(pc)

        self.pc_train, self.pc_test = pc[:split_num], pc[split_num:]
        self.ve_train, self.ve_test = ve[:split_num], ve[split_num:]

    def __len__(self):
        if self.split == 'train':
            return len(self.pc_train)
        else:
            return len(self.pc_test)

    def _get_item(self, index):
        if self.split == 'train':
            return self.pc_train[index], self.ve_train[index]
        else:
            return self.pc_test[index], self.ve_test[index]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = MyDataset(root='dataset/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in tqdm(DataLoader):
        print(point.shape)
        print(label.shape)
