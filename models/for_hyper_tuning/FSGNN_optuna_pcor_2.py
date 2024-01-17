import os
import sys
import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error as MSE, mean_absolute_percentage_error as MAPE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, GINConv, TopKPooling
from torch_geometric.nn import (
    global_mean_pool as gap,
    global_max_pool as gmp,
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

try:
    GPU_NUM = int(sys.argv[1])
    PATH = str(sys.argv[2])
    suffix = str(sys.argv[3])

    sys.argv.remove(str(GPU_NUM))
    sys.argv.remove(PATH)
    sys.argv.remove(suffix)

    saving_path = PATH + "/experiment_" + suffix
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
def regularity(U, LD, num_QT, a, b, c):
    abc_sum = np.round(a + b + c, 2)
    if (a >= 0.0) and (b >= 0.0) and (c >= 0.0) and abc_sum == 1.0:
        pass
    else:
        raise Exception("Value error! The sum of a,b,c, must be 1.")

    LD_group = torch.unique(LD)

    # 빈 dictionary 생성
    indexed_U = {}
    for i in LD_group:
        indexed_U[i.item()] = []

    for i, key in enumerate(LD):
        temp = torch.tensor(0, dtype=float, requires_grad=True)
        for t in range(num_QT):
            temp_for_grad = temp.clone()
            temp_for_grad = temp + U[i, t] ** 2
        indexed_U[key.item()].append(temp_for_grad)

    group_sparsity = torch.tensor(0, dtype=float, requires_grad=True)
    for k in LD_group:
        temp = torch.sum(torch.tensor(indexed_U[k.item()]))
        temp_for_grad = temp.clone()
        temp_2 = torch.sqrt(temp_for_grad)
        temp_for_grad_2 = temp_2.clone()
        group_sparsity_for_grad = group_sparsity.clone
        group_sparsity_for_grad = group_sparsity + temp_for_grad_2

    group_sparsity = group_sparsity_for_grad

    individual_sparsity = torch.square(U)  # 3000 by 10
    individual_sparsity = torch.sum(
        individual_sparsity, 1
    )  # 1은 한 행의 모든 값들을 더함. -> 3000 by 1
    individual_sparsity = torch.sqrt(individual_sparsity)
    individual_sparsity = torch.sum(individual_sparsity, 0)  # 0은 한 열의 모든 값들을 더함.

    element_sparsity = torch.abs(U)
    element_sparsity = torch.sum(element_sparsity, 1)
    element_sparsity = torch.sum(element_sparsity, 0)

    return (a * group_sparsity) + (b * individual_sparsity) + (c * element_sparsity)


def trainer(dataset, skf2, patience=5, batch_size=16, num_features=2, num_classes=3, learning_rate=0.001, epochs=500, alpha=0.4, beta=0.3, gamma=0.3, save_path="/nasdata4/pei4/feature_selection_gnn", gnn_model="gcn"):
    history = []
    epoch_flag = []
    for fold, (train_idx, val_idx) in enumerate(skf2.split(dataset.train_dataset, dataset.y_train)):
        saving_file_path = save_path + "/checkpoint_" + str(fold + 1) + "_" + suffix + ".pt"
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=saving_file_path)

        print()
        print("Fold {}".format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        if len(val_sampler) <= batch_size:
            batch_size_val = len(val_sampler)
        else:
            batch_size_val = batch_size

        train_loader = DataLoader(dataset.train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        val_loader = DataLoader(dataset.train_dataset, batch_size=batch_size_val, sampler=val_sampler, drop_last=True)
        net = feature_selection_GNN(num_features=num_features, num_classes=num_classes, n_snp=dataset.num_snp, gnn_model=gnn_model).to(device)
        constraints = WeightConstraint()

        reg_criterion = nn.MSELoss()
        cls_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        history.append({"train_losses": [], "val_losses": [], "avg_train_losses": [], "avg_val_losses": []})

        for epoch in range(epochs):
            # clear lists to track next epoch
            train_loss = []
            val_loss = []

            ###################
            # train the model #
            ###################
            net.train()
            for i, data in enumerate(train_loader, 0):
                train_clinical_score = data[2]
                optimizer.zero_grad()
                train_reg_output, train_cls_output = net(data)

                U = net.input.weights.squeeze()
                U = torch.transpose(U, 0, 1)
                regularity_loss = regularity(U, dataset.snp_group_data, dataset.num_QT, alpha, beta, gamma)
                train_loss = (reg_criterion(train_reg_output.squeeze(), train_clinical_score) + regularity_loss)

                train_loss += cls_criterion(train_cls_output, data[0].y) * 0.5
                train_loss = train_loss / len(train_loader.sampler)

                train_loss.backward()
                optimizer.step()

                net._modules["input"].apply(constraints)  # Input_layer parameter만 제한
                history[fold]["train_losses"].append(train_loss.item())  # record training loss

            ######################
            # validate the model #
            ######################
            net.eval()
            for i, data in enumerate(val_loader, 0):
                val_clinical_score = data[2]
                val_reg_output, val_cls_output = net(data)

                U = net.input.weights.squeeze()
                U = torch.transpose(U, 0, 1)
                regularity_loss = regularity(U, dataset.snp_group_data, dataset.num_QT, alpha, beta, gamma)
                val_loss = (reg_criterion(val_reg_output.squeeze(), val_clinical_score) + regularity_loss)

                val_loss += cls_criterion(val_cls_output, data[0].y) * 0.5
                val_loss = val_loss / len(val_loader.sampler)
                history[fold]["val_losses"].append(val_loss.item())  # record validation loss

            # print 학습/검증 statistics, epoch당 평균 loss 계산
            train_loss = torch.mean(train_loss)
            val_loss = torch.mean(val_loss)
            history[fold]["avg_train_losses"].append(train_loss.cpu().detach().numpy())
            history[fold]["avg_val_losses"].append(val_loss.cpu().detach().numpy())

            epoch_len = len(str(epochs))
            print_msg = (
                f"[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] "
                + f"\ttrain_loss: {train_loss:.5f} "
                + f"\tvalid_loss: {val_loss:.5f}"
            )

            print(print_msg)

            # early_stopping는 validation loss가 감소하였는지 확인이 필요하며, 만약 감소하였을경우 현제 모델을 checkpoint로 만든다.
            early_stopping(val_loss, net, epoch)

            if early_stopping.early_stop:
                print("Early stopping")
                epoch_flag.append(early_stopping.flag)
                break

            # best model이 저장되어있는 last checkpoint를 로드
            net.load_state_dict(torch.load(saving_file_path))

        print("Fold {} training is done.".format(fold + 1))
    print("Finished Training")
    print()

    return net, history, epoch_flag


def tester(net, num_splits, test_loader, save_path):
    cls_acc_list = []
    reg_acc_list = []

    with torch.no_grad():
        for fold in range(num_splits):
            saving_file_path = (
                save_path + "/checkpoint_" + str(fold + 1) + "_" + suffix + ".pt"
            )
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

            cls_acc = 1 - accuracy_score(true_cls, pred_cls)
            reg_acc_MSE = MSE(true_reg, pred_reg)

            cls_acc_list.append(cls_acc)
            reg_acc_list.append(reg_acc_MSE)

            print("Fold {}".format(fold + 1))
            print(f"==>> cls acc: {cls_acc:.2f}")
            print(f"==>> reg acc(MSE): {reg_acc_MSE:.2f}")
            print(f"")
        print(f"==================")
        print(f"==>> mean cls acc: {np.mean(cls_acc_list):.2f}")
        print(f"==>> mean reg acc(MSE): {np.mean(reg_acc_list):.2f}")

    mean_accuracy = [np.mean(cls_acc_list), np.mean(reg_acc_list)]

    return cls_acc_list, reg_acc_list, mean_accuracy


def objective(trial: optuna.Trial):
    class_label = {0: "CN", 1: "MCI", 2: "AD"}
    combi = [
        # [0.1, 0.2, 0.7],
        # [0.1, 0.3, 0.6],
        # [0.1, 0.4, 0.5],
        # [0.1, 0.5, 0.4],
        # [0.1, 0.6, 0.3],
        # [0.1, 0.7, 0.2],
        # [0.1, 0.8, 0.1],
        # [0.2, 0.1, 0.7],
        [0.2, 0.2, 0.6],
        [0.2, 0.3, 0.5],
        [0.2, 0.4, 0.4],
        [0.2, 0.5, 0.3],
        [0.2, 0.6, 0.2],
        # [0.2, 0.7, 0.1],
        # [0.3, 0.1, 0.6],
        [0.3, 0.2, 0.5],
        [0.3, 0.3, 0.4],
        [0.3, 0.4, 0.3],
        [0.3, 0.5, 0.2],
        # [0.3, 0.6, 0.1],
        # [0.4, 0.1, 0.5],
        [0.4, 0.2, 0.4],
        [0.4, 0.3, 0.3],
        [0.4, 0.4, 0.2],
        # [0.4, 0.5, 0.1],
        # [0.5, 0.1, 0.4],
        [0.5, 0.2, 0.3],
        [0.5, 0.3, 0.2],
        # [0.5, 0.4, 0.1],
        # [0.6, 0.1, 0.3],
        [0.6, 0.2, 0.2],
        # [0.6, 0.3, 0.1],
        # [0.7, 0.1, 0.2],
        # [0.7, 0.2, 0.1],
        # [0.8, 0.1, 0.1],
    ]

    # dynamic hyperparameter
    # QT_type = trial.suggest_categorical(
    #     "QT_type", ["subcor_vol_QT", "Normed_QT", "zscore_QT"]
    # )
    edge_sparse_type = trial.suggest_discrete_uniform("edge_sparse_type", 5, 20, 5)
    temp = trial.suggest_categorical(name="regul_ratio", choices=combi)
    alpha = temp[0]
    beta = temp[1]
    gamma = temp[2]

    # static hyperparameter
    learning_rate = 0.004208
    batch_size = 15
    snp_data_type = "real"
    epochs = 2000
    patience = 5
    num_splits = 4
    rand_state = 42
    node_feature_type = "pcor"
    gnn_model = "gcn"

    data_path = '/nasdata4/pei4/feature_selection_gnn/sample_data/'
    dataset = my_dataset(data_path, node_feature_type=node_feature_type, edge_sparse_type=edge_sparse_type, snp_data_type=snp_data_type)
    num_classes = len(class_label)
    num_features = dataset.train_dataset[0][0].x.shape[1]
    num_classes = len(class_label)
    skf2 = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=rand_state)
    test_loader = DataLoader(dataset.test_dataset)

    print(f"")
    print(f"=============================")
    print(f"==== Dataset Information ====")
    print(f"=============================")
    print(f"number of subjects : {dataset.num_sbj}")
    print(f"number of QT : {dataset.num_QT}")
    print(f"number of SNPs : {dataset.num_snp}")
    print(f"number of classes : {num_classes}")
    print(f"number of node features : {num_features}")
    print(f"node feature type : {dataset.node_feature_type}")
    print(f"edge sparsity type : {dataset.edge_sparse_type}")
    print(f"snp data type : {dataset.snp_data_type}")
    # print(f"QT data type : {QT_type}")
    print(f"")
    print(f"==============================")
    print(f"====== Hyper parameters ======")
    print(f"==============================")
    print(f"learning rate : {learning_rate}")
    print(f"number of epochs : {epochs}")
    print(f"size of batch : {batch_size}")
    print(f"patience : {patience}")
    print(f"number of folds : {num_splits}")
    print(f"regularity term ratio (a) : {alpha}")
    print(f"regularity term ratio (b) : {beta}")
    print(f"regularity term ratio (c) : {gamma}")
    print(f"baseline gnn model : {gnn_model}")
    print(f"")

    net, history, epoch_flag = trainer(
        dataset=dataset,
        skf2=skf2,
        patience=patience,
        batch_size=batch_size,
        num_features=num_features,
        num_classes=num_classes,
        learning_rate=learning_rate,
        epochs=epochs,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        save_path=saving_path,
        gnn_model=gnn_model
    )
    folds_cls_acc, folds_reg_acc, mean_acc = tester(net, num_splits, test_loader, saving_path)
    trial.report(mean_acc[0], epochs)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return mean_acc[0] + mean_acc[1]


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


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.flag = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.flag = epoch
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class WeightConstraint(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, "weights"):
            w = module.weights.data
            w = w.clamp(1e-9, None)
            module.weights.data = w


class U_matrix_similarity(object):
    def __init__(self):
        self.LD_group = pd.read_pickle(
            "/nasdata4/pei4/feature_selection_gnn/new_LD.pkl"
        )
        self.LD_group = np.array(self.LD_group)

    def __call__(self, num_features, num_classes, num_snp, save_path, num_fold):
        U_list = []
        for i in range(num_fold):
            temp_net_path = (
                save_path + "/checkpoint_" + str(i + 1) + "_" + suffix + ".pt"
            )
            temp_net = feature_selection_GNN(
                num_features=num_features, num_classes=num_classes, n_snp=num_snp
            ).to(device)
            temp_net.load_state_dict(torch.load(temp_net_path))
            U = temp_net.input.weights.squeeze()
            U = U.cpu().detach().numpy()
            U = self.group_count(
                self.binarization(self.row_wise_normalization(U), 0), self.LD_group
            )
            U_list.append(U)
        sim_mat = np.ones((num_fold, num_fold))
        for i, mat_A in enumerate(U_list):
            for j, mat_B in enumerate(U_list):
                sim_mat[i][j] = np.round(self.row_wise_inner_product(mat_A, mat_B), 3)
        self.print_figure(sim_mat, save_path)

    def NormalizeData(self, data):
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
        temp = np.zeros((data.shape[0], Group_max.item()))  # 10 by 29
        for i, row in enumerate(data):
            data[i] = row * Group.T
            unique_val_arr, cnt_arr = np.unique(data[i], return_counts=True)
            unique_val_arr = np.asarray(unique_val_arr, dtype=int)
            unique_val_arr = np.delete(unique_val_arr, 0, axis=0)
            cnt_arr = np.delete(cnt_arr, 0, axis=0)
            for j, idx in enumerate(unique_val_arr):
                temp[i][idx - 1] = cnt_arr[j]
        return temp

    def row_wise_inner_product(self, mat_A, mat_B):
        sim_vec = []
        for i, (A, B) in enumerate(zip(mat_A, mat_B)):
            row_wise_similarity = np.sum(A * B) / (
                np.linalg.norm(A) * np.linalg.norm(B)
            )
            sim_vec.append(row_wise_similarity)
        return np.mean(sim_vec)

    def print_figure(self, data, save_path):
        plt.figure(figsize=(10, 10))
        sns.heatmap(data, annot=True, fmt=".2f", cmap="Blues")
        plt.title("U matrix similarity")
        plt.savefig(save_path + "/U matrix similarity_" + suffix + ".png", dpi=600)

def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            "Trial {} finished with best value: {} and parameters: {}. ".format(
            frozen_trial.number,
            frozen_trial.value,
            frozen_trial.params,
            )
        )
###################################################################
def main():
    os.popen("mkdir " + saving_path)

    # Hyperparameter tuning step
    sampler = TPESampler(**TPESampler.hyperopt_parameters())
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=200, callbacks=[logging_callback])

    # print result
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
