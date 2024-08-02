import pickle

import numpy as np
import torch

from pointnet_cls import get_model, get_loss

input_file_path = '862.ply'
model_path = 'best_model.pth'
use_cpu = True

with open(input_file_path, 'r', encoding='ascii') as f:
    data = f.readlines()[10:]
    data = [np.array(i.split(' '), dtype=np.float32) for i in data]
    data = torch.tensor(data)

data = data.unsqueeze(dim=0)
data = data.transpose(2, 1)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


'''MODEL LOADING'''
num_class = 8 * 138
classifier = get_model(num_class, normal_channel=True)
criterion = get_loss()
classifier.apply(inplace_relu)

if not use_cpu:
    classifier = classifier.cuda()
    criterion = criterion.cuda()

checkpoint = torch.load(model_path)
start_epoch = checkpoint['epoch']
classifier.load_state_dict(checkpoint['model_state_dict'])
print('Use pretrain model')

classifier = classifier.eval()

total_loss = 0

pred, _ = classifier(data)
pred = pred[0]
print(pred[:, :5])
with open(input_file_path[:-4] + '.pkl', 'wb') as f:
    pickle.dump(pred, f)
