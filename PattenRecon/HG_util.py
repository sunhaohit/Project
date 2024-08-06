import torch
from sklearn.cluster import KMeans
import torch.nn as nn
import numpy as np
import dhg


def kmeans_cluster(x: torch.Tensor, K_patch: int) -> torch.Tensor:
    x = x.transpose(1, 2)  # x.shape([B, N, 6])
    x = x.view(-1, 6)
    kmeans = KMeans(n_clusters=K_patch, random_state=0).fit(
        x.detach().cpu().numpy())
    cluster_indices = kmeans.labels_  # 获取聚类后对应的GS点云参数
    # usage e.g. clusters[0].size() == torch.Size([15972, 58])
    cluster_indices_dict = {}
    for cluster_idx in range(K_patch):
        cluster_indices_dict[cluster_idx] = np.where(cluster_indices == cluster_idx)[0].tolist()
    clusters = {}
    for cluster_idx, data_indices in cluster_indices_dict.items():
        clusters[cluster_idx] = x[data_indices]

    mean_clusters = []
    for cluster in clusters:
        mean_cluster = clusters[cluster].mean(dim=0)
        mean_clusters.append(mean_cluster)
    assert len(mean_clusters) == K_patch
    mean_clusters = torch.stack(mean_clusters)  # torch.Size([K_patch, C_g=58])
    return mean_clusters


def construct_hypergraph(mean_clusters: torch.Tensor):
    hg_space = dhg.Hypergraph.from_feature_kNN(mean_clusters.detach().cpu(), k=K_space)

    def fix_iso_v(G: dhg.Hypergraph):
        # fix isolated vertices
        iso_v = np.array(G.deg_v) == 0
        if np.any(iso_v):
            extra_e = [tuple([e, ]) for e in np.where(iso_v)[0]]
            G.add_hyperedges(extra_e)
        return G

    hg_space = fix_iso_v(hg_space)
    global_hg = hg_space
    opted_combined = global_hg.e2v(global_hg.v2e(
        X=combined))  # hg_space.smoothing_with_HGNN(hg_space.e2v(hg_space.v2e(X=combined))) # 空域v-e-v + 频域
    # assert opted_combined.shape == torch.Size([K_patch, C_g]) #
    opted_combined_sliced = opted_combined[:, :C_g]  # 取出第二维度中前面的 58 个元素
    assert opted_combined_sliced.shape == torch.Size([K_patch, C_g])

    return g


class HyperGraphEncoder(nn.Module):
    def __init__(self, k_patch, global_feat=True, feature_transform=False):
        super(HyperGraphEncoder, self).__init__()
        #     self.stn = STN3d(channel)
        #     self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        #     self.conv2 = torch.nn.Conv1d(64, 128, 1)
        #     self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #     self.bn1 = nn.BatchNorm1d(64)
        #     self.bn2 = nn.BatchNorm1d(128)
        #     self.bn3 = nn.BatchNorm1d(1024)
        #     self.global_feat = global_feat
        #     self.feature_transform = feature_transform
        #     if self.feature_transform:
        #         self.fstn = STNkd(k=64)
        self.k_patch = k_patch

    #
    def forward(self, x):
        x = kmeans_cluster(x, self.k_patch)
        g = construct_hypergraph(x)
