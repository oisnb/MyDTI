import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Iterator, List, Optional, Union, Tuple
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj, to_dense_batch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


kernel_size = 7
class DCNN(nn.Module):
    def __init__(self, device):
        super(DCNN, self).__init__()
        self.embedding_xt = nn.Embedding(64, 128) #smiles分子共有64种字符 映射为128大小

        self.conv_in = nn.Conv1d(in_channels=80, out_channels=64, kernel_size=1) #输入维度为80 输出维度为64
        self.convs = nn.ModuleList([nn.Conv1d(128, 2 * 128, kernel_size, padding=kernel_size // 2) for _ in range(3)])
        self.pooling = nn.MaxPool1d(2, stride=2)

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.norm = nn.LayerNorm(64)  # 归一化层
        self.do = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1_xt = nn.Linear(64, 200)

    def normalization(self, vector_present, threshold=1):
        vector_present_clone = vector_present.clone()

        vector_present_clone = vector_present_clone - vector_present_clone.min(1, keepdim=True)[0]
        vector_present_clone = vector_present_clone / vector_present_clone.max(1, keepdim=True)[0]
        vector_present_clone *= threshold
        return vector_present_clone

    def forward(self, smile_tensor: List[np.ndarray] = None):
        drug = smile_tensor.to(torch.int64)
        #drug = torch.tensor(drug).to(torch.int64)
        embedded_xt = self.embedding_xt(drug)
        input_nn = self.conv_in(embedded_xt)

        conv_input = input_nn.permute(0, 2, 1)

        for i, conv in enumerate(self.convs):
            conved = self.norm(conv(conv_input))

            conved = F.glu(conved, dim=1)

            conved = conved + self.scale * conv_input

            conv_input = conved

        conved = self.do(self.fc1_xt(conved))


        return conved #返回三维的tensor

class DGNN(torch.nn.Module):
    def __init__(self, device, num_features_mol=78, output_dim=200, dropout=0.2):
        super(DGNN, self).__init__()

        self.device = device
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc = torch.nn.Linear(num_features_mol * 4, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data_mol, max_num):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch  # [604.78]
        x = self.mol_conv1(mol_x, mol_edge_index)  # [635,78] 第一个数字是全部原子数相加？
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv2(x, mol_edge_index)  # [635,156]
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, mol_edge_index)  # [635,312]
        x = self.relu(x)
        x = self.mol_fc(x)
        x, _ = to_dense_batch(x, mol_batch, max_num) #[20,40,312]
        return x #返回三维的数组

class PCNN(torch.nn.Module):
    def __init__(self, num_features_xt=25, n_filters=32, embed_dim=128, output_dim=128):
        super(PCNN, self).__init__()
        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc_xt1 = nn.Linear(121, 200)

        # activation and regularization
        self.relu = nn.ReLU()

    def forward(self, pro_seq):
        # protein input feed-forward:
        target = pro_seq.to(torch.int64)
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt1(embedded_xt) #embedded_xt大小？
        conv_xt = self.relu(conv_xt) #看看这个大小是不是三维的 [8,32,121]
        conv_xt = self.fc_xt1(conv_xt)
        # flatten 别展开
        # xt = conv_xt.view(-1, 32 * 121)
        # xt = self.fc_xt1(xt)
        return conv_xt

class PCM_CNN(nn.Module):
    def __init__(self):
        super(PCM_CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),  # (30,16,498,498) 输出深度为16，表示有16个神经元
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # (30,32,496,496)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # (30,32,248,248)

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # (30,64,246,246)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  # (30,128,244,244)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=4))  # (30,128,60,60)


        self.do = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1_xt = nn.Linear(60 * 60, 200) #线性层使用的参数过多 容易造成内存不够 有无替代方法

    def normalization(self, vector_present, threshold=1):
        vector_present_clone = vector_present.clone()

        vector_present_clone = vector_present_clone - vector_present_clone.min(1, keepdim=True)[0]
        vector_present_clone = vector_present_clone / vector_present_clone.max(1, keepdim=True)[0]
        vector_present_clone *= threshold
        return vector_present_clone

    def forward(self, x):
        x = self.layer1(x)  # (30,1,500,500)->(30,16,498,498)
        x = self.layer2(x)  # (30,16,498,498)->(30,32,496,496)->(30,32,248,248)
        x = self.layer3(x)  # (30,32,248,248)->(30,64,246,246)
        x = self.layer4(x)  # (30,64,246,246)->(30,128,244,244)->(30,128,60,60)
        #x = x.view(x.size(0), -1)  # (30,128,60,60)->(30,460800) #直接在这一步转成三维 四维转三维？
        x = x.view(x.size(0), x.size(1), -1) #(30,128,60,60)->(30,128,3600)
        x = self.relu(x)
        x = self.do(self.fc1_xt(x))
        #x = self.do(self.relu(self.fc1_xt(self.normalization(x)))) #(30,460800)->(30,1024) 30个蛋白质 1024维特征
        return x

class PGNN(torch.nn.Module):
    def __init__(self, device, num_features_pro=54, output_dim=200, dropout=0.2):
        super(PGNN, self).__init__()

        self.device = device
        # self.pro_conv1 = GCNConv(embed_dim, embed_dim)
        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        # self.pro_conv4 = GCNConv(embed_dim * 4, embed_dim * 8)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


    def forward(self, data_pro, max_num):
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch
        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)
        xt = self.pro_fc_g1(xt)
        xt, _ = to_dense_batch(xt, target_batch, max_num) #[20,40,312]

        return xt #[8,216,200]