import os
import json
import pandas as pd
import random

file_names = os.listdir('./dataset')
csv = []

for i in file_names:
    file_name = os.path.join('./dataset', i, 'result.json')
    with open(file_name, 'r') as f:
        json_file = json.load(f)
    caption = json_file[0]['text']
    print(caption)
    csv.append(f'{i},A 3D Model of {caption}\n')
    # print(json_file)

with open('my_csv.csv', 'w') as f:
    f.writelines(csv)
#
