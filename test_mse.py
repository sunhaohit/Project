import pickle
import torch.nn.functional as F
import os

xpath = 've_result/'
ypath = 'PattenRecon/dataset/VE/'
ls_filename = os.listdir(xpath)
for i in ls_filename:
    with open(os.path.join(xpath, i), 'rb') as f:
        x = pickle.load(f)
        # print(x[:, :5])
    with open(os.path.join(ypath, i), 'rb') as f:
        y = pickle.load(f)
        # print(y[:, :5])
    print(F.mse_loss(x, y))
