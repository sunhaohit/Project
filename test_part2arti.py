import sys, os, os.path as osp

sys.path.append(osp.dirname(os.getcwd()))
from core.models import get_model
from init import setup_seed
import yaml, logging, imageio, torch, os
import os.path as osp
import numpy as np
from tqdm import tqdm
from dataset import get_dataset
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from copy import deepcopy
import trimesh

device = torch.device("cuda:0")


def prepare_nap(nap_version="v6.1", nap_ep="5455"):
    # prepare the model
    # cfg_fn = f"../configs/v5/{nap_version}_diffusion.yaml"
    cfg_fn = f"../configs/nap/{nap_version}_diffusion.yaml"
    ckpt = torch.load(f"../log/{nap_version}_diffusion/checkpoint/{nap_ep}.pt")
    with open(cfg_fn, "r") as f:
        cfg = yaml.full_load(f)
    cfg["logging"]["viz_frame"] = 10
    cfg["root"] = osp.dirname(os.getcwd())
    cfg["modes"] = ["train", "val", "test"]
    cfg["dataset"]["dataset_proportion"] = [1.0, 1.0, 1.0]
    # handle the dir change
    cfg["dataset"]["split_path"] = osp.join("..", cfg["dataset"]["split_path"])
    cfg["dataset"]["embedding_index_file"] = osp.join("..", cfg["dataset"]["embedding_index_file"])
    cfg["dataset"]["embedding_precompute_path"] = osp.join(
        "..", cfg["dataset"]["embedding_precompute_path"]
    )
    cfg["model"]["part_shape_prior"]["pretrained_shapeprior_path"] = osp.join(
        "..", cfg["model"]["part_shape_prior"]["pretrained_shapeprior_path"]
    )

    ModelClass = get_model(cfg["model"]["model_name"])
    nap_model = ModelClass(cfg)
    nap_model.model_resume(ckpt, is_initialization=True, network_name=["all"])
    nap_model.to_gpus()
    nap_model.set_eval()
    # get the scale factor
    dataclass = get_dataset(cfg)
    dataset = dataclass(cfg, "train")
    training_E_scale, training_V_scale = dataset.E_scale.copy(), dataset.V_scale.copy()
    training_V_scale = np.concatenate([training_V_scale, dataset.embedding_scale], axis=-1)
    training_E_scale = torch.from_numpy(training_E_scale).float().cuda()
    training_V_scale = torch.from_numpy(training_V_scale).float().cuda()
    training_scale = (training_V_scale, training_E_scale)

    database = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(
        dataset.training_embedding
    )  # it's train embedding
    mesh_names = dataset.training_embedding_index
    return nap_model, training_scale, database, mesh_names, cfg


def gen(model, V_scale, E_scale, N=100):
    net = model.network
    K = model.K
    noise_V = torch.randn(N, K, 1 + 6 + net.shapecode_dim).cuda()
    noise_E = torch.randn(N, (K * (K - 1)) // 2, 13).cuda()
    V, E = net.generate(noise_V, noise_E, V_scale, E_scale)
    # if net.use_hard_v_mask:
    #     random_n = torch.randint(low=2, high=K + 1, size=(N,))
    #     V_mask = torch.arange(K)[None, :] < random_n[:, None]
    #     V_mask = V_mask.float().to(noise_V.device)
    #     noise_V[..., 0] = V_mask  # set the noisy first channel to gt v_mask
    # else:
    #     V_mask = None
    # V, E = net.generate(noise_V, noise_E, V_scale, E_scale, V_mask=V_mask)
    return V, E, noise_V, noise_E


if __name__ == '__main__':
    setup_seed(12345)
    VERSION, EP = "v6.1", "5455"
    # VERSION, EP = "v5.1.5", "5455"
    # VERSION, EP = "v5.1.6", "5455"
    nap_model, training_scale, database, mesh_names, cfg = prepare_nap(VERSION, EP)
    cates = cfg["dataset"]["cates"]
    K = cfg["dataset"]["max_K"]
    DST = f"../log/test/G/K_{K}_cate_{''.join(cates)}_{VERSION}_{EP}"
    DST_viz = f"../log/test/Viz/K_{K}_cate_{''.join(cates)}_{VERSION}_{EP}"
    DST_retrieval = f"../log/test/G/K_{K}_cate_{''.join(cates)}_{VERSION}_{EP}_retrieval"
    DST_retrieval_viz = f"../log/test/Viz/K_{K}_cate_{''.join(cates)}_{VERSION}_{EP}_retrieval"
    GT_MESH_ROOT = "../data/partnet_mobility_graph_mesh"

    N = 449
    # N = 2
    gen_V, gen_E, noise_V, noise_E = gen(nap_model, *training_scale, N=N)

    from save_viz_utils import save

    save(
        gen_V,
        gen_E,
        mesh_extraction_fn=nap_model.generate_mesh,
        dst=DST,
        dst_retrieval=DST_retrieval,
        database=database,
        mesh_names=mesh_names,
        gt_mesh_dir=GT_MESH_ROOT,
    )

    from save_viz_utils import viz_dir

    viz_dir(DST, DST_viz, max_viz=40)
    viz_dir(DST_retrieval, DST_retrieval_viz, max_viz=40)
