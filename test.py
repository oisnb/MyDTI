import torch.nn as nn
import torch
import math

import pickle

# 要保存的字典
my_dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}

# 保存字典到文件
with open('dict.pickle', 'wb') as f:
    pickle.dump(my_dict, f)

# 从文件读取字典
with open('dict.pickle', 'rb') as f:
    loaded_dict = pickle.load(f)

# 打印读取的字典
print(loaded_dict)
