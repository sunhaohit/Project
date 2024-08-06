import pickle

import numpy as np
import torch

from pointnet_cls import get_model, get_loss
import os




def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


'''MODEL LOADING'''
num_class = 8 * 138
classifier = get_model(num_class, normal_channel=True)
criterion = get_loss()
# classifier.apply(inplace_relu)


dir_path = 'test_dataset/'
# model_path = 'log/classification/2024-08-05_16-51/checkpoints/best_model.pth'
model_path = 'log/classification/2024-08-06_11-09/checkpoints/best_model.pth'
save_path = 'NAP/eval/data/'
ls_input_file = os.listdir(dir_path)
ls_data = []
for input_file_path in ls_input_file:
    input_file_path = os.path.join(dir_path, input_file_path)

    use_cpu = True

    with open(input_file_path, 'r', encoding='utf8') as f:
        data = f.readlines()[10:]
        data = [np.array(i.split(' '), dtype=np.float32) for i in data]
        data = torch.tensor(data)

    # data = data.unsqueeze(dim=0)
    data = data.transpose(1, 0)
    ls_data.append(data)

data = torch.stack(ls_data)
print(data.shape)

if not use_cpu:
    classifier = classifier.cuda()
    criterion = criterion.cuda()


checkpoint = torch.load(model_path)
start_epoch = checkpoint['epoch']
classifier.load_state_dict(checkpoint['model_state_dict'])
print('Use pretrain model')

classifier = classifier.eval()
pred_tensor, _ = classifier(data)
for i in range(len(ls_input_file)):
    pred = pred_tensor[i]
    print(pred[:, :5])
    with open(os.path.join(save_path, ls_input_file[i][:-4] + '.pkl'), 'wb') as f:
        pickle.dump(pred, f)
