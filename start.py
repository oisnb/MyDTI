from data_processing import create_dataset_for_5folds
import sys, os
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from model import *
from utils import *
from metrics import *

dataset = 'davis'
fold = 0
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
NUM_EPOCHS = 20

protein_dim = 100
atom_dim = 34
hid_dim = 200
n_layers = 3
n_heads = 8
pf_dim = 256
dropout = 0.1
batch = 64
lr = 1e-3
weight_decay = 1e-4
iteration = 30
kernel_size = 9
models_dir = 'models'
results_dir = 'results'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward,
                  dropout, device)  # 解码
model = Predictor(encoder, decoder, device)  # 你有啥用？总模型
model.to(device)
model_st = Predictor.__name__
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# train_data, valid_data = create_dataset_for_5folds(dataset, fold)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
#                                            collate_fn=collate)
# valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
#                                            collate_fn=collate)
# train(model, device, train_loader, optimizer, 1) #使用Adam 怎么加进去


train_data, valid_data = create_dataset_for_5folds(dataset, fold) #5折训练 在这个函数里处理数据
train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                           collate_fn=collate)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                           collate_fn=collate)

best_mse = 1000
best_test_mse = 1000
best_epoch = -1
model_file_name = 'models/model_' + model_st + '_' + dataset + '_' + str(fold) + '.model'

for epoch in range(NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch + 1)
    print('predicting for valid data')
    G, P, test_cindex, test_r2 = predicting(model, device, valid_loader)
    val = get_mse(G, P) #获取均方误差
    print('valid result:', val, best_mse)
    if val < best_mse: #若验证集误差小于最优 则替换
        best_mse = val
        best_epoch = epoch + 1
        torch.save(model.state_dict(), model_file_name) #保存状态字典
        print('rmse improved at epoch ', best_epoch, '; best_test_mse', best_mse, model_st, dataset, fold, '; test_cindex ', test_cindex, '; test_r2 ', test_r2)

    else:
        print('No improvement since epoch ', best_epoch, '; best_test_mse', best_mse, model_st, dataset, fold, '; test_cindex ', test_cindex, '; test_r2 ', test_r2)