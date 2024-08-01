import os
import json
import pandas as pd
import torch
import pickle

data = torch.load('mydataset_pt/148.pt')
print(list(data[:, 1]), data.shape)
#
# file_names = os.listdir('mydataset')
# for file in file_names:
#     ls = []
#     with open(os.path.join('mydataset',file)) as f:
#         data = f.readlines()
#     data = data[10:]
#     for i in data:
#         ls_tem = i[:-2].split(' ')
#         try:
#             ls_tem.remove("")
#             ls_tem.append("0")
#         except:
#             pass
#         # try:
#         ls.append([float(k) for k in ls_tem])
#         # except:
#         #     print(f'Error: {i}')
#
#     # print(ls)
#     result = torch.tensor(ls).T
#     print(result.shape)
#     torch.save(result, os.path.join('mydataset_pt',file[:-3]+"pt"))
# break

# index = torch.randint(1, 10001, (1024,))
# print(index)
# torch.save(index, 'my_index.pt')
