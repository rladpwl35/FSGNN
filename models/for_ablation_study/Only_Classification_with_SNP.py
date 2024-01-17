import os
import sys
import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, GINConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

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
def regularity(U, LD, num_QT, a, b, c):
    # abc_sum = np.round(a + b + c, 2)
    # if (a >= 0.0) and (b >= 0.0) and (c >= 0.0) and abc_sum == 1.0:
    #     pass
    # else:
    #     raise Exception("Value error! The sum of a,b,c, must be 1.")

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

    return group_sparsity, individual_sparsity, element_sparsity

def trainer(dataset, skf2, batch_size=16, num_features=2, num_classes=3, learning_rate=0.001, epochs=500, a=0.4, b=0.3, c=0.3, save_path="/nasdata4/pei4/feature_selection_gnn", gnn_model="gcn", decay=0.1):
    history = []
    epoch_flag = []
    for fold, (train_idx, val_idx) in enumerate(
        skf2.split(dataset.train_dataset, dataset.y_train)
    ):
        saving_file_path = save_path + "/best_model_" + str(fold + 1) + "_" + suffix + ".pt"
        early_stopping = EarlyStopping(
            verbose=True, path=saving_file_path
        )

        print()
        print("Fold {}".format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        if len(val_sampler) <= batch_size:
            batch_size_val = len(val_sampler)
        else:
            batch_size_val = batch_size

        train_loader = DataLoader(
            dataset.train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            drop_last=True,
        )
        val_loader = DataLoader(
            dataset.train_dataset,
            batch_size=batch_size_val,
            sampler=val_sampler,
            drop_last=True,
        )
        net = feature_selection_GNN(
            num_features=num_features, num_classes=num_classes, n_snp=dataset.num_snp, gnn_model=gnn_model
        ).to(device)
        constraints = WeightConstraint()

        criterion_2 = nn.CrossEntropyLoss()
        optimizer       = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=decay)

        train_losses = []
        valid_losses = []
        for epoch in range(epochs):
            # clear lists to track next epoch
            run_tr_cls_loss = 0
            run_tr_r1_loss  = 0
            run_tr_r2_loss  = 0
            run_tr_r3_loss  = 0

            run_vl_cls_loss = 0
            run_vl_r1_loss  = 0
            run_vl_r2_loss  = 0
            run_vl_r3_loss  = 0

            ###################
            # train the model #
            ###################
            net.train()
            for i, data in enumerate(train_loader, 0):
                train_cls_output = net(data)

                # calculate regularity loss
                U = net.input.weights.squeeze()
                U = torch.transpose(U, 0, 1)
                group_sparsity, individual_sparsity, element_sparsity = regularity(U, dataset.snp_group_data, dataset.num_QT, a, b, c)
                regularity_loss = a*group_sparsity + b*individual_sparsity + c*element_sparsity

                train_loss = criterion_2(train_cls_output, data[0].y) + regularity_loss

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                run_tr_cls_loss += train_loss
                run_tr_r1_loss  += a*group_sparsity
                run_tr_r2_loss  += b*individual_sparsity
                run_tr_r3_loss  += c*element_sparsity

            run_tr_cls_loss = run_tr_cls_loss / len(train_loader.sampler)
            run_tr_r1_loss  = run_tr_r1_loss  / len(train_loader.sampler)
            run_tr_r2_loss  = run_tr_r2_loss  / len(train_loader.sampler)
            run_tr_r3_loss  = run_tr_r3_loss  / len(train_loader.sampler)
            train_losses.append([run_tr_cls_loss.item(),
                                 run_tr_r1_loss.item(),
                                 run_tr_r2_loss.item(),
                                 run_tr_r3_loss.item()])

            ######################
            # validate the model #
            ######################
            net.eval()
            for i, data in enumerate(val_loader, 0):
                val_cls_output = net(data)

                U = net.input.weights.squeeze()
                U = torch.transpose(U, 0, 1)
                group_sparsity, individual_sparsity, element_sparsity = regularity(U, dataset.snp_group_data, dataset.num_QT, a, b, c)
                regularity_loss = a*group_sparsity + b*individual_sparsity + c*element_sparsity
                valid_loss = criterion_2(val_cls_output, data[0].y) + regularity_loss

                run_vl_cls_loss += valid_loss
                run_vl_r1_loss  += a*group_sparsity
                run_vl_r2_loss  += b*individual_sparsity
                run_vl_r3_loss  += c*element_sparsity


            run_vl_cls_loss = run_vl_cls_loss / len(val_loader.sampler)
            run_vl_r1_loss  = run_vl_r1_loss  / len(val_loader.sampler)
            run_vl_r2_loss  = run_vl_r2_loss  / len(val_loader.sampler)
            run_vl_r3_loss  = run_vl_r3_loss  / len(val_loader.sampler)
            valid_losses.append([run_vl_cls_loss.item(),
                                 run_vl_r1_loss.item(),
                                 run_vl_r2_loss.item(),
                                 run_vl_r3_loss.item()])

            # print 학습/검증 statistics, epoch당 평균 loss 계산
            epoch_len = len(str(epochs))
            print_msg = (
                f"[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] "
                + f"\ttrain_loss: {train_loss:.5f} "
                + f"\tvalid_loss: {valid_loss:.5f}"
            )

            print(print_msg)

            # early_stopping는 validation loss가 감소하였는지 확인이 필요하며, 만약 감소하였을경우 현제 모델을 checkpoint로 만든다.
            early_stopping(valid_loss, net)
            torch.save(net.state_dict(), save_path + "/checkpoint_" + str(fold + 1) + "_" + suffix + ".pt")
            net.load_state_dict(torch.load(save_path + "/checkpoint_" + str(fold + 1) + "_" + suffix + ".pt"))

        epoch_flag.append(early_stopping.flag)
        history.append([train_losses, valid_losses])
        print("Fold {} training is done.".format(fold + 1))

    print("Finished Training")
    print()

    return net, history, epoch_flag


def tester(net, num_splits, test_loader, save_path):
    cls_acc_list = []

    with torch.no_grad():
        for fold in range(num_splits):
            saving_file_path = save_path + "/checkpoint_" + str(fold + 1) + "_" + suffix + ".pt"
            net.load_state_dict(torch.load(saving_file_path))
            net.eval()

            pred_cls = []
            true_cls = []

            for i, data in enumerate(test_loader, 0):
                true_cls.append(data[0].y.item())

                pred_cls_output = net(data)

                pred_cls_output = torch.exp(pred_cls_output)
                pred_cls_output = pred_cls_output.cpu().detach().numpy()

                pred_cls_output = np.argmax(pred_cls_output, axis=None)
                pred_cls.append(pred_cls_output)

            print(true_cls)
            print(pred_cls)
            cls_acc = accuracy_score(true_cls, pred_cls)
            cls_acc_list.append(cls_acc)

            print('Fold {}'.format(fold + 1))
            print(f"==>> cls acc: {cls_acc:.2f}")
            print(f"")
        print(f"==================")
        print(f"==>> mean cls acc: {np.mean(cls_acc_list):.2f}")
            

    mean_accuracy = np.mean(cls_acc_list)

    return cls_acc_list, mean_accuracy

def printFigure(num_splits, history, net, save_path):
    history = np.asarray(history)
    fig, axs = plt.subplots(4, 5, figsize=(15, 15))
    for fold in range(num_splits):
        train_cls_loss  = history[fold][0][:, 0]
        train_r1_loss   = history[fold][0][:, 1]
        train_r2_loss   = history[fold][0][:, 2]
        train_r3_loss   = history[fold][0][:, 3]

        valid_cls_loss  = history[fold][1][:, 0]
        valid_r1_loss   = history[fold][1][:, 1]
        valid_r2_loss   = history[fold][1][:, 2]
        valid_r3_loss   = history[fold][1][:, 3]

        x_axix          = np.linspace(1, len(train_cls_loss), len(train_cls_loss))

        # train & valid losses
        plt.subplot(4, 5, 5*fold + 1)
        X1 = train_cls_loss + train_r1_loss + train_r2_loss + train_r3_loss
        X2 = valid_cls_loss + valid_r1_loss + valid_r2_loss + valid_r3_loss
        plt.plot(x_axix, X1, label="train")
        plt.plot(x_axix, X2, label="valid")
        plt.legend()

        # training -> regress & cls & regul losses
        plt.subplot(4, 5, 5*fold + 2)
        X2 = train_cls_loss
        X3 = train_r1_loss + train_r2_loss + train_r3_loss
        plt.plot(x_axix, X2, label="train_cls")
        plt.plot(x_axix, X3, label="train_regul")
        plt.legend()

        # validation -> regress & cls & regul losses
        plt.subplot(4, 5, 5*fold + 3)
        X2 = valid_cls_loss
        X3 = valid_r1_loss + valid_r2_loss + valid_r3_loss
        plt.plot(x_axix, X2, label="valid_cls")
        plt.plot(x_axix, X3, label="valid_regul")
        plt.legend()

        # training -> regularity term losses
        plt.subplot(4, 5, 5*fold + 4)
        X1 = train_r1_loss
        X2 = train_r2_loss
        X3 = train_r3_loss
        plt.plot(x_axix, train_r1_loss, label="train_r1")
        plt.plot(x_axix, train_r2_loss, label="train_r2")
        plt.plot(x_axix, train_r3_loss, label="train_r3")
        plt.legend()

        # validation -> regularity term losses
        plt.subplot(4, 5, 5*(fold + 1))
        X1 = valid_r1_loss
        X2 = valid_r2_loss
        X3 = valid_r3_loss
        plt.plot(x_axix, valid_r1_loss, label="valid_r1")
        plt.plot(x_axix, valid_r2_loss, label="valid_r2")
        plt.plot(x_axix, valid_r3_loss, label="valid_r3")
        plt.legend()

    plt.savefig(save_path + "/train_valid_loss_" + suffix + ".png", dpi=400)

    fig, axs = plt.subplots(4, 1, figsize=(30, 24))
    for fold in range(num_splits):
        file_load_path = save_path + "/checkpoint_" + str(fold + 1) + "_" + suffix + ".pt"
        net.load_state_dict(torch.load(file_load_path))
        plt.subplot(4, 1, fold + 1)
        U = net.input.weights.squeeze()
        U = U.cpu().detach().numpy()
        U[np.where(U < 0)] = 0
        sns.heatmap(U)

    plt.savefig(save_path + "/snp_inputlayer_param_" + suffix + ".png", dpi=400)


################################################################################
class my_dataset(Dataset):
    def __init__(self, data_path, node_feature_type="pcor", edge_sparse_type=5, snp_data_type="real", qt_data_type="raw", num_fold=5):
        super(my_dataset, self).__init__()
        self.node_feature_type  = node_feature_type
        self.edge_sparse_type   = edge_sparse_type
        self.snp_data_type      = snp_data_type
        self.qt_data_type       = qt_data_type

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

            if self.qt_data_type == "raw":
                t1_measure  = subject_data[9]
            else:
                t1_measure  = subject_data[10]

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
        # self.fc2 = nn.Linear(
        #     in_features=self.fc1.out_features, out_features=self.input.n_QT
        # )
        self.fc3 = nn.Linear(700, out_features=conv1_out_channels)
        self.bn2 = nn.BatchNorm1d(self.fc3.out_features)
        self.fc4 = nn.Linear(
            in_features=self.fc3.out_features, out_features=self.num_classes
        )


        # custom layer initialization
        nn.init.kaiming_normal_(self.input.weights)

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

        cls = F.log_softmax(self.fc4(xx), dim=-1)

        return cls


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(self, verbose=False, delta=0, path="checkpoint.pt", trace_func=print):
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
        self.verbose = verbose
        self.flag = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.flag += self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            pass
        else:
            self.best_score = score
            self.flag += self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        return 1


class WeightConstraint(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, "weights"):
            w = module.weights.data
            w = w.clamp(1e-9, None)
            module.weights.data = w


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
        "--qt_type", type=str, required=False, help="type of QT data type"
    )
    parser.add_argument("--lr", type=float, required=True, help="learning rate")
    parser.add_argument("--epochs", type=int, required=True, help="number of epochs")
    parser.add_argument("--batch_size", type=int, required=True, help="")
    parser.add_argument("--folds", type=int, required=True, help="number of folds")
    parser.add_argument(
        "--a", type=float, required=False, help="regularity term ratio (a)"
    )
    parser.add_argument(
        "--b", type=float, required=False, help="regularity term ratio (b)"
    )
    parser.add_argument(
        "--c", type=float, required=False, help="regularity term ratio (b)"
    )
    parser.add_argument(
        "--gnn", type=str, required=False, help="baseline gnn model"
    )
    parser.add_argument(
        "--decay", type=float, required=True, help="L2 regularization. (weight_decay)"
    )

    ###################################################################
    if len(sys.argv) <= 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # default value
    a   = 0.4
    b    = 0.3
    c   = 0.3

    node_feature_type   = args.feature_type
    edge_sparse_type    = args.edge_type
    snp_data_type       = args.snp_type
    QT_type             = args.qt_type
    learning_rate       = args.lr
    epochs              = args.epochs
    batch_size          = args.batch_size
    num_splits          = args.folds
    a                   = args.a
    b                   = args.b
    c                   = args.c
    gnn_model           = args.gnn
    decay               = args.decay

    saving_path = PATH + "/experiment_" + suffix
    os.popen('mkdir ' + saving_path)

    data_path = '/nasdata4/pei4/feature_selection_gnn/sample_data/'
    dataset = my_dataset(data_path, node_feature_type=node_feature_type, edge_sparse_type=edge_sparse_type, snp_data_type=snp_data_type, qt_data_type=QT_type)
    rand_state = 42
    class_label = {0: "CN", 1: "MCI", 2: "AD"}

    num_classes = len(class_label)
    num_features = dataset.train_dataset[0][0].x.shape[1]

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
    print(f"QT data type : {QT_type}")
    print(f"")
    print(f"==============================")
    print(f"====== Hyper parameters ======")
    print(f"==============================")
    print(f"learning rate : {learning_rate}")
    print(f"number of epochs : {epochs}")
    print(f"size of batch : {batch_size}")
    print(f"number of folds : {num_splits}")
    print(f"regularity term ratio (a) : {a}")
    print(f"regularity term ratio (b) : {b}")
    print(f"regularity term ratio (c) : {c}")
    print(f"baseline gnn model : {gnn_model}")
    print(f"")

    skf2 = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=rand_state)
    test_loader = DataLoader(dataset.test_dataset)

    net, history, epoch_flag = trainer(
        dataset=dataset,
        skf2=skf2,
        batch_size=batch_size,
        num_features=num_features,
        num_classes=num_classes,
        learning_rate=learning_rate,
        epochs=epochs,
        a=a,
        b=b,
        c=c,
        save_path=saving_path,
        gnn_model=gnn_model,
        decay=decay
    )
    folds_cls_acc, mean_acc = tester(net, num_splits, test_loader, saving_path)
    printFigure(num_splits, history, net, saving_path)

    training_info = pd.DataFrame(
        {
            'Experiment No.' : [suffix],
            'Node feature' : [dataset.node_feature_type],
            'Edge sparsity' : [dataset.edge_sparse_type],
            'SNP data type' : [dataset.snp_data_type],
            'QT data type' : [dataset.qt_data_type],
        'Regularity term ratio (A)' : [a],
        'Regularity term ratio (B)' : [b],
        'Regularity term ratio (C)' : [c],
            'Batch size' : [batch_size],
            'Learning Rate' : [learning_rate],
            'Fold 1 epochs' : [epoch_flag[0]],
            'Fold 1 cls accuracy' : [folds_cls_acc[0]],
            'Fold 2 epochs' : [epoch_flag[1]],
            'Fold 2 cls accuracy' : [folds_cls_acc[1]],
            'Fold 3 epochs' : [epoch_flag[2]],
            'Fold 3 cls accuracy' : [folds_cls_acc[2]],
            'Fold 4 epochs' : [epoch_flag[3]],
            'Fold 4 cls accuracy' : [folds_cls_acc[3]],
            'Mean cls accuracy' : mean_acc,
        }
    )

    training_info.to_csv(saving_path + '/training_info.csv')

if __name__ == "__main__":
    main()