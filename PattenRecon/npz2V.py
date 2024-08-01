"""
将原数据集的.npz文件进行处理，处理为V矩阵
"""

import os
import pickle

import numpy as np
import torch.nn.functional as F
import torch
import json


def axis_angle_to_matrix(axis_angle):
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def compact_pack(V, K=8, permute=False):
    # print(len(V), K)
    if len(V) > K:
        print(f"Warning, extend {K} to {len(V)}")
        K = len(V)
    num_v = len(V)
    n_empty = K - num_v

    # Nodes
    # v_mask = np.zeros(K, dtype=np.bool)
    v_mask = np.zeros(K)
    v_mask[: len(V)] = True
    if permute:
        # in the origin index, first num_v are object
        v_map = np.random.permutation(K).tolist()  # stores the original id
    else:
        v_map = np.arange(K).tolist()
    v_mask = [v_mask[i] for i in v_map]

    _v_bbox = [v["bbox_L"] for v in V] + [np.zeros(3)] * n_empty
    v_bbox = [_v_bbox[i] for i in v_map]
    v_bbox = torch.from_numpy(np.stack(v_bbox, axis=0)).float()
    # p_global = T_gl @ p_local
    _v_t_gl = [v["abs_center"] for v in V] + [np.zeros(3)] * n_empty
    v_t_gl = [_v_t_gl[i] for i in v_map]
    v_t_gl = torch.from_numpy(np.stack(v_t_gl, axis=0)).float()
    # ! Now assume partnet-M all init part R = I
    v_r_gl = torch.zeros(K, 3).float()
    ret_v = torch.cat([torch.LongTensor(v_mask)[..., None], v_bbox, v_r_gl, v_t_gl], -1)

    return ret_v, v_map


# prepare the code book
def prepare_embedding():
    mesh_index_fn = './resource/partnet_m_partkeys.json'
    with open(mesh_index_fn, "r") as f:
        embedding_index = json.load(f)
    embedding_index = embedding_index['train'] + embedding_index['val'] + embedding_index['test']
    print(f'len(embedding_index):{len(embedding_index)}')
    i = embedding_index.index('148_0')

    embedding_precompute_path = './resource/codebook/s1.5_partshape_ae_737.npz'

    data = np.load(embedding_precompute_path, allow_pickle=True)
    embedding = data["embedding"]
    # ! all the embedding codebook is saved in order [train, val, test] !!

    embedding_valid = data["valid_mask"]
    print(
        f"Load embedding from {embedding_precompute_path}, totally valid precompute {embedding_valid.sum()} embeddings"
    )
    return embedding_index, embedding


def add_embedding(V, v_map, filename, embedding_index, embedding):
    code_list = []
    for i in range(8):
        if V[i, 0] < 0.5:
            code_list.append(np.zeros_like(embedding[0]))
        else:
            key = f"{filename}_{i}"
            code_id = embedding_index.index(key)
            code_list.append(embedding[code_id])
    code_list = np.stack(code_list, 0)
    embedding_scale = abs(embedding).max(axis=0) + 1e-8
    std = embedding_scale.copy()
    # ! warning, the code is already divided
    code_list = code_list / std[None, ...]
    precompute_code = torch.from_numpy(code_list).float()

    V_added = torch.cat([V, precompute_code], dim=-1)
    return V_added, v_map


if __name__ == '__main__':
    embedding_index, embedding = prepare_embedding()

    ls_file = os.listdir('./dataset/npz')
    for file in ls_file:
        data = np.load(os.path.join('dataset', 'npz', file), allow_pickle=True)
        print(file)
        V_list, E_list = data["V"].tolist(), data["E"].tolist()
        if len(V_list) > 8:
            continue
        V, v_map = compact_pack(V_list)
        V, v_map = add_embedding(V, v_map, file[:-4], embedding_index, embedding)
        # print(V)
        with open(os.path.join('dataset', 'VE', file[:-4] + '.pkl'), 'wb') as f:
            pickle.dump(V, f)
