import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
from submodel import *

class SelfAttention(nn.Module): #自注意层
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]

        return x


class Encoder(nn.Module): #编码层 处理蛋白质
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size , dropout, device):
        super().__init__()

        self.PCNN = PCNN() #卷积核大小也是超参数 可以作为超参数传进来
        self.PCM_CNN = PCM_CNN()
        self.PGNN = PGNN(device)
        self.emb1 = nn.Embedding(121 + 1, 200)
        self.emb2 = nn.Embedding(3600 + 1, 200)
        self.emb3 = nn.Embedding(128 + 1, 200)
        self.emb4 = nn.Embedding(100 + 1, 200)
        self.lin_vc = nn.Linear(100, 200)

    def forward(self, data_pro_seq, data_pro, max_num, data_pro_pcm, data_pro_vc, device): #处理蛋白质 还要拼接为一个矩阵
        #pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(self.device)
        #protein = protein + self.pos_embedding(pos)
        #protein = [batch size, protein len,protein_dim]

        #pro_cnn 确保每个都是三维
        pro_cnn = self.PCNN(data_pro_seq) #[20,32,200]
        #pro_cnn = self.emb1(pro_cnn)
        pcm_cnn = self.PCM_CNN(data_pro_pcm) #[20,128,200]
        #pcm_cnn = self.emb2(pcm_cnn)
        pro_gnn = self.PGNN(data_pro, max_num) #[20,1634,200]
        #pro_gnn = self.emb3(pro_gnn)
        #data_pro_vc 可以直接用 已经是三维的
        #data_pro_vc =self.emb4(data_pro_vc)
        data_pro_vc = self.lin_vc(data_pro_vc) #[20,20,200]
        #合并四个量？ 先用embedding缩到一个num_feats上
        src = torch.cat((pro_cnn, pcm_cnn), dim=1)
        src = torch.cat((src, pro_gnn), dim=1)
        src = torch.cat((src, data_pro_vc), dim=1)
        return src



class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)

        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))

        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)

        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, hid dim]

        return x


class DecoderLayer(nn.Module): #解码层
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound sent len]
        # src_mask = [batch size, protein len]

        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg


class Decoder(nn.Module): #解码操作 处理过的蛋白质和处理过的药物进行解码
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.device = device
        self.layers = nn.ModuleList( #有啥用
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.ft = nn.Linear(200, 256) #没用到
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 1) #改成1 变回归任务
        #self.fc_3 = nn.Linear(128, 2)
        self.do_1 = nn.Dropout(0.2)
        # self.gn = nn.GroupNorm(8, 256)

    def forward(self, trg, src, trg_mask=None,src_mask=None): #第一个是处理后的药物 第二个是处理后的蛋白质 第三个是mask的药物
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
       # trg = self.ft(trg) #每次八个八个对进行处理

        # trg = [batch size, compound len, hid dim]

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask) #药物 蛋白质decode处理

        # trg = [batch size, compound len, hid dim]
        """Use norm to determine which atom is significant. """
        norm = torch.norm(trg, dim=2)
        # norm = [batch size,compound len]
        norm = F.softmax(norm, dim=1)
        # norm = [batch size,compound len]
        # trg = torch.squeeze(trg,dim=0)
        # norm = torch.squeeze(norm,dim=0)
        sum = torch.zeros((trg.shape[0], self.hid_dim)).to(self.device)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = trg[i, j, ]
                v = v * norm[i, j]
                sum[i, ] += v
        # sum = [batch size,hid_dim]
        label = self.do_1(F.relu(self.fc_1(sum)))
        # label = self.do_1(F.relu(self.fc_2(label)))
        label = self.fc_2(label)
        return label


class Predictor(nn.Module):
    def __init__(self, encoder, decoder, device, atom_dim=34):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.DCNN = DCNN(device)
        self.DGNN = DGNN(device)
        self.device = device
        self.emb1 = nn.Embedding(64 + 1, 200)
        self.emb2 = nn.Embedding(20 + 1, 200)
        self.emb3 = nn.Embedding(1024 + 1, 200)
        self.mol_fp = torch.nn.Linear(1024, 200)

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        N = len(atom_num)  # batch size
        compound_mask = torch.zeros((N, compound_max_len))
        protein_mask = torch.zeros((N, protein_max_len))
        for i in range(N):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2).to(self.device)
        return compound_mask, protein_mask

    def pack(self, atoms, proteins, device): #蛋白质大小 54
        atoms_len = 0
        proteins_len = 0
        N = len(atoms)
        atom_num = []
        for i in range(N):
            atom_num.append(atoms[i].x.shape[0])
            if atoms[i].x.shape[0] >= atoms_len:
                atoms_len = atoms[i].x.shape[0]

        protein_num = []
        for i in range(N):
            protein_num.append(proteins[i].x.shape[0])
            if proteins[i].x.shape[0] >= proteins_len:
                proteins_len = proteins[i].x.shape[0]

        atoms_new = torch.zeros((N, atoms_len, 78), device=device)
        for i in range(N):
            a_len = atoms[i].x.shape[0]
            atoms_new[i, :a_len, :] = atoms[i].x

        proteins_new = torch.zeros((N, proteins_len, 54), device=device)
        for i in range(N):
            a_len = proteins[i].x.shape[0]
            proteins_new[i, :a_len, :] = proteins[i].x

        adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
        for i in range(N):
            adj = torch.zeros(atoms_len, atoms_len)
            adj[atoms[i].edge_index[0], atoms[i].edge_index[1]] = 1
            adjs_new[i, :atoms_len, :atoms_len] = adj

        adjs_pro = torch.zeros((N, proteins_len, proteins_len), device=device)
        for i in range(N):
            adj = torch.zeros(proteins_len, proteins_len)
            adj[proteins[i].edge_index[0], proteins[i].edge_index[1]] = 1
            adjs_pro[i, :proteins_len, :proteins_len] = adj

        return (atoms_new, adjs_new, proteins_new, adjs_pro, atom_num, protein_num)  # atom_num就是每个药物拥有的原子个数 在MyDTI里把药物

    def forward(self, data_mol_seq, data_mol, data_mol_fp, data_pro_seq, data_pro_pcm, data_pro, data_pro_vc):
        data_pro_pcm = data_pro_pcm.squeeze(1) #删除一维 每一种输入的形式 有

#########在这里pack？ 模仿 通过list增加
        compound, adjs_d, protein, adjs_p, atom_num, protein_num = self.pack(data_mol, data_pro, self.device)
        ####药物处理 记录大小
        drug_cnn = self.DCNN(data_mol_seq).to(torch.long) #已经是三维 [20,128,200]
     #   drug_cnn = self.emb1(drug_cnn)
        drug_gnn = self.DGNN(data_mol, max(atom_num)) #调整为三维 [20,44,200]
     #   drug_gnn = self.emb2(drug_gnn)
        #data_mol_fp是分子指纹 可以直接拿来用 分子指纹调整为[batch,1,num_feats]
        data_mol_fp = data_mol_fp.reshape(data_mol_fp.shape[0], 1, data_mol_fp.shape[1]) #[20,1,1024]
        data_mol_fp = data_mol_fp.float()
        data_mol_fp = self.mol_fp(data_mol_fp) #[20,1,200]

        #根据第二维进行拼接 先embedding
        enc_trg = torch.cat((drug_cnn, drug_gnn), dim=1)
        enc_trg = torch.cat((enc_trg, data_mol_fp), dim=1) #[20,173,200]


        ####蛋白质处理 在encode里面进行 encode里面要拼接成一个三维数组  cnn的不需要pooling gnn的要多加一维 vc直接用
        enc_src = self.encoder(data_pro_seq, data_pro, max(protein_num), data_pro_pcm, data_pro_vc, self.device) #[20,1814,200]

        compound_mask, protein_mask = self.make_masks(atom_num, protein_num, enc_trg.shape[1], enc_src.shape[1])

        ####把处理好的蛋白质和处理好的药物 + mask矩阵丢到decode中 能否让返回值只返回一个值 回归任务
        out = self.decoder(enc_trg, enc_src, compound_mask, protein_mask)
        ####return 这个值 丢到外面去算损失5
        return out

