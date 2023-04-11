import pandas as pd
import numpy as np
import os
import random
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx

from utils import *


# nomarlize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


# print(res_weight_table)


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


# mol atom feature for mol graph
def atom_features(atom): #返回原子特征
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), #属于哪个原子
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + #化合价
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + #H原子数量
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + #原子的隐含价
                    [atom.GetIsAromatic()]) #是否为苯环


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms() #原子数量

    features = [] #分子图的所有原子特征 有几个原子就有几个list
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = [] #分子图的所有边特征 每个list代表边的起点和终点
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed() #将图转换为有向图
    edge_index = []
    mol_adj = np.zeros((c_size, c_size)) #分子的邻接矩阵 对称
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    mol_adj += np.matrix(np.eye(mol_adj.shape[0])) #主对角线变全1
    index_row, index_col = np.where(mol_adj >= 0.5) #记录有边的点 一行一行记录 只记录不为0的地方
    for i, j in zip(index_row, index_col): #打包起来就是边 i为起点 j为终点
        edge_index.append([i, j])
    # print('smile_to_graph')
    # print(np.array(features).shape)
    return c_size, features, edge_index #第一个是原子数量 第二个是所有原子的特征 第三个是边的特征 即所有边 每条边记录下起点和终点 自己跟自己也有边


# target feature for target graph
def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                print('error', len(line), len(pro_seq))
                continue
            count = 0
            for res in line:
                if res not in pro_res_table:
                    count += 1
                    continue
                pfm_mat[pro_res_table.index(res), count] += 1
                count += 1
    # ppm_mat = pfm_mat / float(line_count)
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat
    # k = float(len(pro_res_table))
    # pwm_mat = np.log2(ppm_mat / (1.0 / k))
    # pssm_mat = pwm_mat
    # print(pssm_mat)
    return pssm_mat


def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table))) #每个氨基酸基团的热编码
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq) #pssm是位置权重矩阵 行是氨基酸种类 列是蛋白质序列 计算原蛋白质序列中的某个氨基酸突变为另一个的可能性 值越大越容易突变 跟蛋白质接触图不是一个东西
    other_feature = seq_feature(pro_seq) #蛋白质的其他特征
    # print('target_feature')
    # print(pssm.shape)
    # print(other_feature.shape)

    # print(other_feature.shape)
    # return other_feature
    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1) #返回总特征


# target aln file save in data/dataset/aln
def target_to_feature(target_key, target_sequence, aln_dir):
    # aln_dir = 'data/' + dataset + '/aln'
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    # if 'X' in target_sequence:
    #     print(target_key)
    feature = target_feature(aln_file, target_sequence) #获取蛋白质的全部特征 包含pssm矩阵 以及氨基酸热编码等
    return feature


# pconsc4 predicted contact map save in data/dataset/pconsc4 蛋白质接触图 两个基团靠的近不近
def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    target_edge_index = []
    target_size = len(target_sequence) #蛋白质的氨基酸数目
    # contact_dir = 'data/' + dataset + '/pconsc4'
    contact_file = os.path.join(contact_dir, target_key + '.npy') #先找到那个蛋白质图
    contact_map = np.load(contact_file) #把蛋白质接触图重新加载回来
    contact_map += np.matrix(np.eye(contact_map.shape[0])) #自己跟自己也有接触 明天从这里开始 得到蛋白质接触图 再看看他是怎么从接触图过渡到gnn的
    index_row, index_col = np.where(contact_map >= 0.5) #记录下所有有接触的残基与残基
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_feature = target_to_feature(target_key, target_sequence, aln_dir)
    target_edge_index = np.array(target_edge_index)
    return target_size, target_feature, target_edge_index #第一个是蛋白质拥有的氨基酸数目 第二个是蛋白质的全部特征 包括pssm矩阵 氨基酸热编码等 第三个是哪些基团有接触

def protein_to_matrix(target_key, contact_dir): #把蛋白质转换为自身的蛋白质接触图 以二维矩阵形式存在 一次加载一个蛋白质接触图矩阵
    contact_file = os.path.join(contact_dir, target_key + '.npy') #先找到那个蛋白质接触图路径
    contact_map = np.load(contact_file) #把蛋白质接触图重新加载回来
    contact_map += np.array(np.eye(contact_map.shape[0])) #自己跟自己也有接触
    contact_map.resize(500, 500, refcheck=False) #缩小或扩大为统一维度 方便cnn处理
    return contact_map #返回矩阵

# to judge whether the required files exist
def valid_target(key, dataset):
    contact_dir = 'data/' + dataset + '/pconsc4'
    aln_dir = 'data/' + dataset + '/aln'
    contact_file = os.path.join(contact_dir, key + '.npy')
    aln_file = os.path.join(aln_dir, key + '.aln')
    # print(contact_file, aln_file)
    if os.path.exists(contact_file) and os.path.exists(aln_file):
        return True
    else:
        return False


def data_to_csv(csv_file, datalist):
    with open(csv_file, 'w') as f:
        f.write('compound_iso_smiles,target_sequence,target_key,affinity\n')
        for data in datalist:
            f.write(','.join(map(str, data)) + '\n')


def create_dataset_for_test(dataset):
    # load dataset
    dataset_path = 'data/' + dataset + '/'
    test_fold = json.load(open(dataset_path + 'folds/test_fold_setting1.txt'))
    ligands = json.load(open(dataset_path + 'ligands_can.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(dataset_path + 'Y', 'rb'), encoding='latin1')
    # load contact and aln
    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/pconsc4'
    msa_list = []
    contact_list = []
    for key in proteins:
        msa_list.append(os.path.join(msa_path, key + '.aln'))
        contact_list.append(os.path.join(contac_path, key + '.npy'))

    drugs = []
    prots = []
    prot_keys = []
    drug_smiles = []
    # smiles
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
        drug_smiles.append(ligands[d])
    # seqs
    for t in proteins.keys():
        prots.append(proteins[t])
        prot_keys.append(t)
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    valid_test_count = 0
    rows, cols = np.where(np.isnan(affinity) == False)
    rows, cols = rows[test_fold], cols[test_fold]
    temp_test_entries = []
    for pair_ind in range(len(rows)):
        # if the required files is not exist, then pass
        if not valid_target(prot_keys[cols[pair_ind]], dataset):
            continue
        ls = []
        ls += [drugs[rows[pair_ind]]]
        ls += [prots[cols[pair_ind]]]
        ls += [prot_keys[cols[pair_ind]]]
        ls += [affinity[rows[pair_ind], cols[pair_ind]]]
        temp_test_entries.append(ls)
        valid_test_count += 1
    csv_file = 'data/' + dataset + '_test.csv'
    data_to_csv(csv_file, temp_test_entries)
    print('dataset:', dataset)
    print('test entries:', len(test_fold), 'effective test entries', valid_test_count)

    compound_iso_smiles = drugs
    target_key = prot_keys

    # create smile graph
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    # print(smile_graph['CN1CCN(C(=O)c2cc3cc(Cl)ccc3[nH]2)CC1']) #for test

    # create target graph
    # print('target_key', len(target_key), len(set(target_key)))
    target_graph = {}
    for key in target_key:
        if not valid_target(key, dataset):  # ensure the contact and aln files exists
            continue
        g = target_to_graph(key, proteins[key], contac_path, msa_path)
        target_graph[key] = g

    # count the number of  proteins with aln and contact files
    print('effective drugs,effective prot:', len(smile_graph), len(target_graph))
    if len(smile_graph) == 0 or len(target_graph) == 0:
        raise Exception('no protein or drug, run the script for datasets preparation.')

    # 'data/davis_test.csv' or data/kiba_test.csv'
    df_test = pd.read_csv('data/' + dataset + '_test.csv')
    test_drugs, test_prot_keys, test_Y = list(df_test['compound_iso_smiles']), list(df_test['target_key']), list(
        df_test['affinity'])
    test_drugs, test_prot_keys, test_Y = np.asarray(test_drugs), np.asarray(test_prot_keys), np.asarray(test_Y)
    test_dataset = DTADataset(root='data', dataset=dataset + '_test', xd=test_drugs, y=test_Y,
                              target_key=test_prot_keys, smile_graph=smile_graph, target_graph=target_graph)

    return test_dataset


def create_dataset_for_5folds(dataset, fold=0):
    # load dataset
    dataset_path = 'data/' + dataset + '/'
    train_fold_origin = json.load(open(dataset_path + 'folds/train_fold_setting1.txt'))
    train_fold_origin = [e for e in train_fold_origin]  # for 5 folds

    ligands = json.load(open(dataset_path + 'ligands_can.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    # load contact and aln
    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/pconsc4' #npy是numpy保存的二进制文件 直接load就可以读取数值
    msa_list = []
    contact_list = []
    for key in proteins:
        msa_list.append(os.path.join(msa_path, key + '.aln'))
        contact_list.append(os.path.join(contac_path, key + '.npy'))

    # load train,valid and test entries 训练集分为5部分 选择一部分作为验证集 其他四部分作为训练集
    train_folds = []
    valid_fold = train_fold_origin[fold]  # one fold
    for i in range(len(train_fold_origin)):  # other folds
        if i != fold:
            train_folds += train_fold_origin[i]

    affinity = pickle.load(open(dataset_path + 'Y', 'rb'), encoding='latin1')
    drugs = []
    prots = []
    prot_keys = []
    drug_smiles = []
    pro_contect = {}  # 收集全部蛋白质的二维矩阵

    # smiles
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True) #从smiles转成分子再转成smiles 使其唯一
        drugs.append(lg)
        drug_smiles.append(ligands[d])
    # seqs
    for t in proteins.keys():
        prots.append(proteins[t])
        prot_keys.append(t)
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    opts = ['train', 'valid']
    valid_train_count = 0
    valid_valid_count = 0
    for opt in opts:
        if opt == 'train': #考虑如何把数据对应起来 通过将结合力矩阵展平+顺序查找 找到药物smiles对应的蛋白质及其结合力大小
            rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[train_folds], cols[train_folds]
            train_fold_entries = []
            for pair_ind in range(len(rows)):
                if not valid_target(prot_keys[cols[pair_ind]], dataset):  # ensure the contact and aln files exists
                    continue
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                train_fold_entries.append(ls) #list集合 一个list包括一个药物smiles 蛋白质序列 蛋白质key 药物和蛋白质的结合力
                valid_train_count += 1

            csv_file = 'data/' + dataset + '_' + 'fold_' + str(fold) + '_' + opt + '.csv'
            data_to_csv(csv_file, train_fold_entries) #存储所有数据
        elif opt == 'valid':
            rows, cols = np.where(np.isnan(affinity) == False) #行代表药物位置 列代表蛋白质位置 把二维矩阵拉平 一个个点进行判断
            rows, cols = rows[valid_fold], cols[valid_fold] #valid_fold就是对应点的位置 每个样本都是点
            valid_fold_entries = []
            for pair_ind in range(len(rows)):
                if not valid_target(prot_keys[cols[pair_ind]], dataset):
                    continue
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                valid_fold_entries.append(ls)
                valid_valid_count += 1

            csv_file = 'data/' + dataset + '_' + 'fold_' + str(fold) + '_' + opt + '.csv'
            data_to_csv(csv_file, valid_fold_entries)
    print('dataset:', dataset)
    # print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))

    # entries with protein contact and aln files are marked as effiective
    print('fold:', fold)
    print('train entries:', len(train_folds), 'effective train entries', valid_train_count)
    print('valid entries:', len(valid_fold), 'effective valid entries', valid_valid_count)

    compound_iso_smiles = drugs
    target_key = prot_keys

    # create smile graph 对所有smiles构建唯一的图表示 图的特征包括点的数量 点的全部特征 所有边的起点和终点
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile) #把smiles序列转换成图的相关信息 点 边 构成一个元组
        smile_graph[smile] = g #把smiles和自身对应的图联系起来
    # print(smile_graph['CN1CCN(C(=O)c2cc3cc(Cl)ccc3[nH]2)CC1']) #for test

    # create target graph 和smiles一个道理
    # print('target_key', len(target_key), len(set(target_key)))
    target_graph = {}
    for key in target_key:
        if not valid_target(key, dataset):  # ensure the contact and aln files exists
            continue
        g = target_to_graph(key, proteins[key], contac_path, msa_path) #返回元组 第一个是蛋白质拥有的氨基酸数目 第二个是蛋白质的全部特征 包括pssm矩阵 氨基酸热编码等 第三个是哪些基团有接触
        target_graph[key] = g
        pcm =protein_to_matrix(key, contac_path)
        pro_contect[key] = pcm

    # count the number of  proteins with aln and contact files
    print('effective drugs,effective prot:', len(smile_graph), len(target_graph))
    if len(smile_graph) == 0 or len(target_graph) == 0:
        raise Exception('no protein or drug, run the script for datasets preparation.')

    # 'data/davis_fold_0_train.csv' or data/kiba_fold_0__train.csv'
    train_csv = 'data/' + dataset + '_' + 'fold_' + str(fold) + '_' + 'train' + '.csv'
    df_train_fold = pd.read_csv(train_csv)
    train_drugs, train_prot_keys, train_Y = list(df_train_fold['compound_iso_smiles']), list(
        df_train_fold['target_key']), list(df_train_fold['affinity']) #读回来的时候以list形式读回来
    train_drugs, train_prot_keys, train_Y = np.asarray(train_drugs), np.asarray(train_prot_keys), np.asarray(train_Y) #再转换为numpy
    train_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=train_drugs, target_key=train_prot_keys, #数据都是一组一组的 有多少药就有多少蛋白质
                               y=train_Y, smile_graph=smile_graph, target_graph=target_graph, proteins=proteins, pro_map=pro_contect) #将三个东西封装起来 打包为一个dataset对象 有没有重复的？ prot2vec能不能类似处理？


    df_valid_fold = pd.read_csv('data/' + dataset + '_' + 'fold_' + str(fold) + '_' + 'valid' + '.csv') #对验证集做同样的操作 读回来然后封装起来
    valid_drugs, valid_prots_keys, valid_Y = list(df_valid_fold['compound_iso_smiles']), list(
        df_valid_fold['target_key']), list(df_valid_fold['affinity'])
    valid_drugs, valid_prots_keys, valid_Y = np.asarray(valid_drugs), np.asarray(valid_prots_keys), np.asarray(
        valid_Y)
    valid_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=valid_drugs,
                               target_key=valid_prots_keys, y=valid_Y, smile_graph=smile_graph,
                               target_graph=target_graph, proteins=proteins, pro_map=pro_contect)
    return train_dataset, valid_dataset
