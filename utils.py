import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

from metrics import get_cindex, get_rm2

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64} #药物映射dict

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}     #一个数字对应一个字母 蛋白质映射dict

protVec_model = {} #遍历N-gram文件 每三个氨基酸编码 100维 最后共有9048种生物词汇 每个词汇由三个氨基酸组成
with open("../DeepGS-master/dataset/embed/protVec_100d_3grams.csv", encoding='utf8') as f:
    for line in f:
        values = eval(line).rsplit('\t')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        protVec_model[word] = coefs

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg


def encoding(line, MAX_SMI_LEN, smi_ch_ind): #line是字符串 MAX_SMI_LEN是读取的最大字符串长度 smi_ch_ind是大的映射字典
	X = np.zeros(MAX_SMI_LEN)
	for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
		X[i] = smi_ch_ind[ch]

	return X #.tolist()


def seq_to_embeddings(sequence, embedding_dict, embedding_size):
    embeddings = []
    for aa in sequence:
        if aa in embedding_dict:
            embeddings.append(embedding_dict[aa])
        else:
            embeddings.append(np.zeros(embedding_size)) #zero好 还是 random好？
    return np.array(embeddings)


# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, target_key=None, target_graph=None, proteins=None, pro_map=None):

        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.process(xd, target_key, y, smile_graph, target_graph, proteins, pro_map)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, target_key, y, smile_graph, target_graph, proteins, pro_map): #在这里添加新的信息 蛋白质接触图 prot2vec等 fingerprint
        assert (len(xd) == len(target_key) and len(xd) == len(y)), 'The three lists must be the same length!'
        data_list_mol = [] #存药物分子图信息
        data_list_pro = [] #存蛋白质分子图信息
        mol_smiles = [] #存分子对应的矩阵
        fingerprint = []
        data_pro_vector = []
        pro_connectmap = []
        data_len = len(xd)
        data_pro_seq = []
        for i in range(data_len): #一组一组处理 想想那个excel 多加一个列表（字典？）存fingerprint 顺序要对应 还要一个来存prot2vec 蛋白质接触图再看看 蛋白质序列可以直接存？
            smiles = xd[i]
            tar_key = target_key[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit 取出药物图特征
            c_size, features, edge_index = smile_graph[smiles]
            target_size, target_features, target_edge_index = target_graph[tar_key] #取出蛋白质图特征
            # print(np.array(features).shape, np.array(edge_index).shape)
            # print(target_features.shape, target_edge_index.shape)
            # make the graph ready for PyTorch Geometrics GCN algorithms: 数据准备
            GCNData_mol = DATA.Data(x=torch.Tensor(features), #第一个 19个原子 特征维数
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0), #原来是[61,2]代表61条边 每条边两个顶点 transpose一下换顺序
                                    y=torch.FloatTensor([labels])) #所有属性都只代表大小 [1]代表大小为1
            GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))

            GCNData_pro = DATA.Data(x=torch.Tensor(target_features),
                                    edge_index=torch.LongTensor(target_edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]))
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))
            # print(GCNData.target.size(), GCNData.target_edge_index.size(), GCNData.target_x.size())
            data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)

            #添加药物分子序列 用于cnn
            mol_smiles.append(encoding(smiles, 80, CHARISOSMISET)) #用label_smiles进行分子编码 要先将列表转换为字符串

            #添加fingerprint
            mol = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            bits = fp.ToBitString()
            vectorofmol = [int(b) for b in bits]
            fingerprint.append(vectorofmol)

            #添加蛋白质序列 用于cnn
            data_pro_seq.append(encoding(proteins[tar_key], 1000, seq_dict))

            #添加蛋白质接触图特征 用于cnn
            pcm = pro_map[tar_key]
            pcm = pcm.reshape(1, 1, 500, 500)  # 调整为四维array [batch,channel,length,width]
            pcm = torch.tensor(pcm)
            pro_connectmap.append(pcm)

            #添加prot2vec 直接用
            protein = proteins[tar_key] #protein 是字符串形式
            protein = protein[:60].ljust(60,'#')
            trigrams = [protein[i:i + 3] for i in range(0, len(protein) - 2, 3)] #转换为三元组
            prot2vec = seq_to_embeddings(trigrams, protVec_model, 100)
            data_pro_vector.append(prot2vec)



        #mol_smiles = torch.LongTensor(mol_smiles) #需不需要同意转成tensor

        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]

        self.mol_seq = mol_smiles #分子序列对应的向量 用于cnn
        self.data_mol = data_list_mol
        self.data_pro = data_list_pro
        self.mol_fp = fingerprint
        self.pro_pcm = pro_connectmap
        self.pro_vector = data_pro_vector
        self.pro_seq = data_pro_seq


    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx): #在这里新增内容
        return self.data_mol[idx], self.data_pro[idx], self.mol_seq[idx], self.mol_fp[idx], self.pro_seq[idx], self.pro_pcm[idx], self.pro_vector[idx]

def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 200
    TRAIN_BATCH_SIZE = 20
    loss_fn = torch.nn.MSELoss()
    loss_fn = loss_fn.to(device)
    running_cindex = AverageMeter()
    for batch_idx, data in enumerate(train_loader): #遍历每一个batch 一共四十个 每个512个样本 每个data都是个元组 包含分子batch和蛋白质batch data什么形式
        #再把信息拿出来 data[0] - data[6]
        data_mol = data[0].to(device) #药物分子图信息
        data_pro = data[1].to(device) #蛋白质分子图信息
        data_mol_seq = torch.from_numpy(np.asarray(data[2])).to(device)
        data_mol_fp = torch.from_numpy(np.asarray(data[3])).to(device)
        data_pro_seq = torch.from_numpy(np.asarray(data[4])).to(device)
        data_pro_pcm = torch.tensor([item.cpu().detach().numpy() for item in data[5]]).to(device)
       # data_pro_pcm = torch.from_numpy(np.asarray(data[5])).to(device)
        data_pro_vc = torch.from_numpy(np.asarray(data[6])).to(device)

        optimizer.zero_grad()
        output = model(data_mol_seq, data_mol, data_mol_fp, data_pro_seq, data_pro_pcm, data_pro, data_pro_vc) #训练模型
        cindex = get_cindex(data_mol.y.detach().cpu().numpy().reshape(-1), output.detach().cpu().numpy().reshape(-1)) ###
        output = output.to(device)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device)) #计算损失
        loss.backward() #梯度回归
        optimizer.step() #更新参数
        running_cindex.update(cindex, data_mol.y.size(0)) ###

        if batch_idx % LOG_INTERVAL == 0: #每十个batch报告一次进度
            epoch_cindex = running_cindex.get_average()
            running_cindex.reset()

            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tcindex: {:.4f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item(),
                                                                           epoch_cindex))

             ###########剩下model模型建立 Encoder处理蛋白质 连接成一条向量？ 药物先进行处理 结果拉成一条向量 再在Decoder里跟蛋白质进一步处理 Preditior是整个过程

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)  # 药物分子图信息
            data_pro = data[1].to(device)  # 蛋白质分子图信息
            data_mol_seq = torch.from_numpy(np.asarray(data[2])).to(device)
            data_mol_fp = torch.from_numpy(np.asarray(data[3])).to(device)
            data_pro_seq = torch.from_numpy(np.asarray(data[4])).to(device)
            data_pro_pcm = torch.tensor([item.cpu().detach().numpy() for item in data[5]]).to(device)
            # data_pro_pcm = torch.from_numpy(np.asarray(data[5])).to(device)
            data_pro_vc = torch.from_numpy(np.asarray(data[6])).to(device)

            output = model(data_mol_seq, data_mol, data_mol_fp, data_pro_seq, data_pro_pcm, data_pro, data_pro_vc)
            output = output.to(device)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)

    epoch_cindex = get_cindex(total_labels.numpy(), total_preds.numpy())
    epoch_r2 = get_rm2(total_labels.numpy().flatten(), total_preds.numpy().flatten())
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), epoch_cindex, epoch_r2


def collate(data_list): #要不要新加东西 这里只有两个
    batchA = Batch.from_data_list([data[0] for data in data_list]) #药物分子图
    batchB = Batch.from_data_list([data[1] for data in data_list]) #蛋白质分子图
    batchC = [data[2] for data in data_list] #药物序列
    batchD = [data[3] for data in data_list] #药物分子指纹
    batchE = [data[4] for data in data_list] #蛋白质序列
    batchF = [data[5] for data in data_list] #蛋白质接触图
    batchG = [data[6] for data in data_list] #蛋白质向量
    return batchA, batchB, batchC, batchD, batchE, batchF, batchG
