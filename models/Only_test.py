import sys
import argparse


import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error as MSE, mean_absolute_percentage_error as MAPE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, GINConv, TopKPooling
from torch_geometric.nn import (
    global_mean_pool as gap,
    global_max_pool as gmp,
)
from torchinfo import summary as summary

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    GPU_NUM = int(sys.argv[1])
    PATH = str(sys.argv[2])
    suffix = str(sys.argv[3])

    sys.argv.remove(str(GPU_NUM))
    sys.argv.remove(PATH)
    sys.argv.remove(suffix)
except:
    print()
    print("Error!")
    print("GPU number, saving path, and experiment number must be entered as inputs.")
    print("Input is as follows:")
    print("\tUsage : python3 FSGNN.py 3 /nasdata4/pei4/feature_selection_gnn/experiments 2")
    print()
    sys.exit(1)

torch.cuda.set_device(GPU_NUM)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print()
    print(
        f"server: {torch.cuda.get_device_name()} - cuda({torch.cuda.current_device()}) v{torch.version.cuda} is available"
    )
    print(f"Torch version: {torch.__version__}")
    print(f"Count of using GPUs:", torch.cuda.device_count())
    print()
    # print(torch.cuda.memory_summary())
else:
    print("Can not use GPU device!")
    sys.exit(1)

################################################################################
def is_interactive():
    """Return True if all in/outs are tty"""
    # TODO: check on windows if hasattr check would work correctly and add value:
    return sys.stdin.isatty() and sys.stdout.isatty() and sys.stderr.isatty()


def setup_exceptionhook():
    """
    Overloads default sys.excepthook with our exceptionhook handler.
    If interactive, our exceptionhook handler will invoke pdb.post_mortem;
    if not interactive, then invokes default handler.
    """

    def _pdb_excepthook(type, value, tb):
        if is_interactive():
            import traceback
            import pdb

            traceback.print_exception(type, value, tb)
            pdb.post_mortem(tb)
        else:
            lgr.warn("We cannot setup exception hook since not in interactive mode")

    sys.excepthook = _pdb_excepthook


################################################################################
def tester(net, num_splits, test_loader, save_path):
    cls_acc_list = []
    reg_acc_list = []

    with torch.no_grad():
        for fold in range(num_splits):
            saving_file_path = save_path + "/checkpoint_" + str(fold + 1) + "_" + suffix + ".pt"
            net.load_state_dict(torch.load(saving_file_path))
            net.eval()

            pred_cls = []
            pred_reg = []

            true_cls = []
            true_reg = []

            for i, data in enumerate(test_loader, 0):
                true_cls.append(data[0].y.item())
                true_reg.append(data[2].cpu().detach().numpy())

                pred_reg_output, pred_cls_output = net(data)

                pred_cls_output = torch.exp(pred_cls_output)
                pred_cls_output = pred_cls_output.cpu().detach().numpy()
                pred_reg_output = pred_reg_output.cpu().detach().numpy()

                pred_cls_output = np.argmax(pred_cls_output, axis=None)
                pred_cls.append(pred_cls_output)
                pred_reg.append(pred_reg_output)

            true_reg = np.asarray(true_reg)
            pred_reg = np.asarray(pred_reg)
            true_reg = np.squeeze(true_reg)
            pred_reg = np.squeeze(pred_reg)

            cls_acc = accuracy_score(true_cls, pred_cls)
            reg_acc_MSE = MSE(true_reg, pred_reg)

            cls_acc_list.append(cls_acc)
            reg_acc_list.append(reg_acc_MSE)

            print('Fold {}'.format(fold + 1))
            print(f"==>> cls acc: {cls_acc}")
            print(f"==>> reg acc(MSE): {reg_acc_MSE}")
            print(f"")
        print(f"==================")
        print(f"==>> mean cls acc: {np.mean(cls_acc_list):.2f}")
        print(f"==>> mean reg acc(MSE): {np.mean(reg_acc_list):.2f}")

    mean_accuracy = [np.mean(cls_acc_list), np.mean(reg_acc_list)]

    return cls_acc_list, reg_acc_list, mean_accuracy

def printFigure(num_splits, net, save_path):
    fig, axs = plt.subplots(4, 1, figsize=(30, 24))
    for fold in range(num_splits):
        file_load_path = save_path + "/best_model_" + str(fold + 1) + "_" + suffix + ".pt"
        net.load_state_dict(torch.load(file_load_path))
        plt.subplot(4, 1, fold + 1)
        U = net.input.weights.squeeze()
        U = U.cpu().detach().numpy()
        U[np.where(U < 0)] = 0
        sns.heatmap(U)

    plt.savefig(save_path + "/snp_inputlayer_param_" + suffix + ".png", dpi=600)


################################################################################
class my_dataset(Dataset):
    def __init__(self, data_path, node_feature_type="pcor", edge_sparse_type=5, snp_data_type="real", num_fold=5):
        super(my_dataset, self).__init__()
        self.node_feature_type  = node_feature_type
        self.edge_sparse_type   = edge_sparse_type
        self.snp_data_type      = snp_data_type

        snp_Additive    = pd.read_csv(data_path + 'Data01_SNPs_Additive.csv')   # snp additive 정보 읽어오기
        snp_GENEgroups  = pd.read_csv(data_path + 'Data02_SNPs_GENEgroups.csv') # snp gene group 정보 읽어오기
        add_snp_list    = snp_Additive.columns[6:].to_list()                    # csv파일에서 FID, IID, PAT, MAT, SEX, PHENOTYPE 제외한 SNP정보만 추출
        gene_group      = dict(zip(list(snp_GENEgroups.SNPID), list(snp_GENEgroups.GENEgroup))) # {SNP명 : group 번호} 생성
        # gene_group_list = list(snp_GENEgroups.SNPID)    # group 번호 정보를 담은 list data 생성

        self.snp_group = []
        for snp_name in add_snp_list:
            self.snp_group.append(gene_group[snp_name[:-2]])    # SNP명을 key로 받는 dicionary를 이용해서 add_snp_list의 SNP명 순서로 group 번호 할당
        self.snp_group_data = torch.tensor(self.snp_group, dtype=torch.long)

        self.my_Data = []
        self.num_sbj    = 157
        self.num_QT     = 8
        self.num_snp    = 3001

        smaple_data_features = np.load(data_path + 'smaple_data_features.npy', allow_pickle=True)
        for subject_data in smaple_data_features:
            if node_feature_type == "pcor":
                node_feature = subject_data[1]
            else:   # "bold"
                node_feature = subject_data[2]

            if edge_sparse_type == 5:
                edge_idx = subject_data[3]
            elif edge_sparse_type == 10:
                edge_idx = subject_data[4]
            elif edge_sparse_type == 15:
                edge_idx = subject_data[5]
            else:
                edge_idx = subject_data[6]

            graph_label = subject_data[7]

            if self.snp_data_type == "real":
                snp_data = subject_data[8]
            else:
                snp_data = torch.randint(0, 3, size=(157, 3001)).to(device=device)
                torch.save(snp_data, data_path + "/fake_snp_data.pt")

            t1_measure  = subject_data[9]

            # numpy to tensor
            node_feature    = torch.Tensor(node_feature).transpose(0, 1)
            graph_label     = torch.tensor(graph_label, dtype=torch.long)
            edge_idx        = torch.Tensor(edge_idx).transpose(0, 1)
            fc_data         = Data(x=node_feature, edge_index=edge_idx.long(), y=graph_label).to(device=device)

            snp_data    = torch.Tensor(snp_data).to(device=device)
            t1_measure  = torch.Tensor(t1_measure).to(device=device)
            snp_data    = torch.squeeze(snp_data)
            t1_measure  = torch.squeeze(t1_measure)

            self.my_Data.append([fc_data, snp_data, t1_measure])

        rand_state = 42
        self.num_fold = num_fold
        skf = StratifiedKFold(n_splits=self.num_fold, shuffle=True, random_state=rand_state)

        y_idx = []
        for i in range(self.num_sbj):
            y_idx.append(self.my_Data[i][0].y.item())

        self.train_dataset  = []
        self.test_dataset   = []
        self.y_train = []
        for train_idx, test_idx in skf.split(range(self.num_sbj), y_idx):
            for i in train_idx:
                self.train_dataset.append([self.my_Data[i][0], self.my_Data[i][1], self.my_Data[i][2]])
                self.y_train.append(self.my_Data[i][0].y.item())
            for j in test_idx:
                self.test_dataset.append([self.my_Data[j][0], self.my_Data[j][1], self.my_Data[j][2]])
            break

    # 인덱스에 해당되는 데이터를 tensor 형태로 반환
    def __getitem__(self, idx):
        fc_map = self.my_Data[idx][0]
        minor_allele_cnt = self.my_Data[idx][1]
        t1_meausre = self.my_Data[idx][2]

        return fc_map, minor_allele_cnt, t1_meausre

    # 데이터 총 개수 반환
    def __len__(self):
        return len(self.num_sbj)


class Input_layer(nn.Module):
    def __init__(self, n_QT=10, n_snp=3000):
        super().__init__()
        self.n_QT = n_QT
        self.n_snp = n_snp

        self.weights = torch.rand(1, self.n_QT, self.n_snp).to(device)
        self.weights = nn.Parameter(self.weights, requires_grad=True).cuda()

    def forward(self, x):
        return x * self.weights.mean(dim=1)


class feature_selection_GNN(nn.Module):
    def __init__(self, **kwargs):
        super(feature_selection_GNN, self).__init__()
        # define kwargs
        self.num_features = (
            kwargs["num_features"] if "num_features" in kwargs.keys() else 116
        )
        self.num_classes = (
            kwargs["num_classes"] if "num_classes" in kwargs.keys() else 4
        )
        self.pooling_ratio = (
            kwargs["pooling_ratio"] if "pooling_ratio" in kwargs.keys() else 0.5
        )
        self.n_QT = kwargs["n_QT"] if "n_QT" in kwargs.keys() else 8
        self.n_snp = kwargs["n_snp"] if "n_snp" in kwargs.keys() else 3001

        self.gnn_model = (
            kwargs["gnn_model"] if "gnn_model" in kwargs.keys() else "gcn"
        )

        conv1_out_channels = 50
        conv2_out_channels = 50
        # define layers
        if self.gnn_model == "gcn":
            self.conv1 = GCNConv(in_channels=self.num_features, out_channels=conv1_out_channels)
            self.conv2 = GCNConv(in_channels=conv1_out_channels, out_channels=conv1_out_channels)
        elif self.gnn_model == "gin":
            self.conv1 = GINConv(nn.Sequential(nn.Linear(self.num_features, conv1_out_channels), nn.ReLU(), nn.Linear(conv1_out_channels, conv1_out_channels)))
            self.conv2 = GINConv(nn.Sequential(nn.Linear(conv1_out_channels, conv2_out_channels), nn.ReLU(), nn.Linear(conv2_out_channels, conv2_out_channels)))
            
        self.pool1 = TopKPooling(
            in_channels=conv1_out_channels, ratio=self.pooling_ratio
        )
        self.pool2 = TopKPooling(conv2_out_channels, ratio=self.pooling_ratio)
        self.input = Input_layer(n_QT=self.n_QT, n_snp=self.n_snp)
        self.fc1 = nn.Linear(in_features=self.input.n_snp, out_features=500)
        self.bn1 = nn.BatchNorm1d(self.fc1.out_features)
        self.fc2 = nn.Linear(
            in_features=self.fc1.out_features, out_features=self.input.n_QT
        )
        self.fc3 = nn.Linear(700, out_features=conv1_out_channels)
        self.bn2 = nn.BatchNorm1d(self.fc3.out_features)
        self.fc4 = nn.Linear(
            in_features=self.fc3.out_features, out_features=self.num_classes
        )

        # custom layer initialization
        nn.init.xavier_normal_(self.input.weights)

    def forward(self, data):
        x = data[0].x
        edge_index = data[0].edge_index
        x_snp = data[1]
        batch = (
            data[0].batch
            if hasattr(data[0], "batch")
            else torch.tensor((), dtype=torch.long).new_zeros(data[0].num_nodes)
        )

        # start here
        x1 = self.conv1(x, edge_index)
        x1, edge_index, edge_attr, batch, perm, score1 = self.pool1(
            x1, edge_index, batch=batch
        )
        x2 = self.conv2(x1, edge_index)
        x1_gmp = gmp(x1, batch)
        x1_gap = gap(x1, batch)
        x1 = torch.cat([x1_gmp, x1_gap], dim=1)
        x2, edge_index, edge_attr, batch, perm, score2 = self.pool2(
            x2, edge_index, batch=batch
        )
        x2 = torch.cat([gmp(x2, batch), gap(x2, batch)], dim=1)
        xx = torch.cat([x1, x2], dim=1)
        x_snp = F.relu(self.input(x_snp))
        x_snp = F.relu(self.bn1(self.fc1(x_snp)))
        xx = torch.cat([x_snp, xx], dim=1)
        xx = self.bn2(self.fc3(xx))

        reg = F.relu(self.fc2(x_snp))
        cls = F.log_softmax(self.fc4(xx), dim=-1)

        return reg, cls


class U_matrix_similarity(object):
    def __init__(self, LD_group):
        self.LD_group = LD_group

    def __call__(self, num_features, num_classes, num_snp, save_path, num_fold, gnn_model):
        U_list = []
        for i in range(num_fold):
            temp_net_path = save_path + "/checkpoint_" + str(i + 1) + "_" + suffix + ".pt"
            temp_net = feature_selection_GNN(num_features=num_features, num_classes=num_classes, n_snp=num_snp, gnn_model=gnn_model).to(device)
            temp_net.load_state_dict(torch.load(temp_net_path))
            U = temp_net.input.weights.squeeze()
            U = U.cpu().detach().numpy()
            # U[np.where(U < 0)] = 0
            U_hat = self.group_count(self.binarization(self.row_wise_normalization(U), 0.9), self.LD_group)
            U_list.append(U_hat)
        sim_mat = np.ones((num_fold, num_fold))
        for i, mat_A in enumerate(U_list):
            for j, mat_B in enumerate(U_list):
                sim_mat[i][j] = np.round(self.row_wise_inner_product(mat_A, mat_B), 3)
        self.print_figure(sim_mat, save_path)

    def NormalizeData(self, data):
        if (np.max(data) - np.min(data)) == 0:
            return 0
        else:
            return (data - np.min(data)) / (np.max(data) - np.min(data))

    def row_wise_normalization(self, data):
        temp = np.empty(data.shape)
        for i, row in enumerate(data):
            temp[i] = self.NormalizeData(row)
        return temp
    
    def binarization(self, data, threshold=0):
        data[np.where(data > threshold)] = 1
        return data

    def group_count(self, data, Group):
        # Group_min = Group[np.argmin(Group)]  # 1
        Group_max = Group[np.argmax(Group)]  # 29
        temp = np.zeros((data.shape[0], Group_max.item())) # 10 by 29
        for i, row in enumerate(data):
            data[i] = row * Group.T
            unique_val_arr, cnt_arr = np.unique(data[i], return_counts=True)
            unique_val_arr = np.asarray(unique_val_arr, dtype=int)
            unique_val_arr = np.delete(unique_val_arr, 0, axis=0)
            cnt_arr = np.delete(cnt_arr, 0, axis=0)
            for j, idx in enumerate(unique_val_arr):
                temp[i][idx-1] = cnt_arr[j]
        return temp

    def row_wise_inner_product(self, mat_A, mat_B):
        sim_vec = []
        for i, (A, B) in enumerate(zip(mat_A, mat_B)):
            row_wise_similarity = np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B))
            sim_vec.append(row_wise_similarity)
        return np.mean(sim_vec)

    def print_figure(self, data, save_path):
        plt.figure(figsize=(10, 10))
        sns.heatmap(data, annot=True, fmt=".2f", cmap='Blues')
        plt.title("U matrix similarity")
        plt.savefig(save_path + "/U matrix similarity_" + suffix + ".png", dpi=600)

###################################################################
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)


###################################################################
def main():
    parser = MyParser()
    # define arguments
    parser.add_argument(
        "--feature_type", type=str, required=True, help="type of node feature"
    )
    parser.add_argument(
        "--edge_type", type=int, required=True, help="type of edge sparsity"
    )
    parser.add_argument(
        "--snp_type", type=str, required=True, help="type of snp data type"
    )
    parser.add_argument(
        "--gnn", type=str, required=False, help="baseline gnn model"
    )

    ###################################################################
    if len(sys.argv) <= 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    saving_path = PATH + "/experiment_" + suffix

    node_feature_type = args.feature_type
    edge_sparse_type = args.edge_type
    snp_data_type = args.snp_type
    gnn_model = args.gnn

    data_path = '/nasdata4/pei4/feature_selection_gnn/sample_data/'
    dataset = my_dataset(data_path, node_feature_type=node_feature_type, edge_sparse_type=edge_sparse_type, snp_data_type=snp_data_type)
    class_label = {0: "CN", 1: "MCI", 2: "AD"}
    num_classes = len(class_label)
    num_features = dataset.train_dataset[0][0].x.shape[1]

    net = feature_selection_GNN(num_features=num_features, num_classes=num_classes, n_snp=dataset.num_snp, gnn_model=gnn_model).to(device)
    num_splits = 4
    test_loader = DataLoader(dataset.test_dataset)

    folds_cls_acc, folds_reg_acc, mean_acc = tester(net, num_splits, test_loader, saving_path)
    printFigure(num_splits, net, saving_path)
    gene_group = dataset.snp_group_data.cpu().detach().numpy()
    calc_ld = U_matrix_similarity(LD_group=gene_group)
    calc_ld(num_features, num_classes, dataset.num_snp, saving_path, num_splits, gnn_model)

if __name__ == "__main__":
    main()