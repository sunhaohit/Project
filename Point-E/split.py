import pickle
import os
import random

with open('validation_set.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)

file_names = os.listdir('mydataset')
for i, n in enumerate(file_names):
    file_names[i] = n[:-4]

random.shuffle(file_names[:])  # 注意：直接在原列表上操作或复制一份

train, val = file_names[:800], file_names[800:]

with open('mytrainset.pkl', 'wb') as f:
    pickle.dump(train, f)
with open('myvalset.pkl', 'wb') as f:
    pickle.dump(val, f)

with open('mytrainset.pkl', 'rb') as f:
    data1 = pickle.load(f)
with open('myvalset.pkl', 'rb') as f:
    data2 = pickle.load(f)
print(data1)
print(data2)
