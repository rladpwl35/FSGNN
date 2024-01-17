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
def trainer(dataset, skf2, batch_size=16, num_features=2, num_classes=3, learning_rate=0.001, epochs=500, save_path="/nasdata4/pei4/feature_selection_gnn", gnn_model="gcn", decay=0.1):
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

        criterion_2 = nn.CrossEntropyLoss()
        optimizer   = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=decay)

        train_losses = []
        valid_losses = []
        for epoch in range(epochs):
            # clear lists to track next epoch
            run_tr_cls_loss = 0
            run_vl_cls_loss = 0

            ###################
            # train the model #
            ###################
            net.train()
            for i, data in enumerate(train_loader, 0):
                train_cls_output = net(data)

                train_loss = criterion_2(train_cls_output, data[0].y)


                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                run_tr_cls_loss += train_loss
            run_tr_cls_loss = run_tr_cls_loss / len(train_loader.sampler)
            train_losses.append(run_tr_cls_loss.item())
            ######################
            # validate the model #
            ######################
            net.eval()
            for i, data in enumerate(val_loader, 0):
                val_cls_output = net(data)

                valid_loss = criterion_2(val_cls_output, data[0].y)
                run_vl_cls_loss += valid_loss
            run_vl_cls_loss = run_vl_cls_loss / len(val_loader.sampler)
            valid_losses.append(run_vl_cls_loss.item())

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
        train_cls_loss  = history[fold][0]
        valid_cls_loss  = history[fold][1]
        x_axix          = np.linspace(1, len(train_cls_loss), len(train_cls_loss))

        # train & valid losses
        plt.subplot(4, 1, fold + 1)
        X1 = train_cls_loss
        X2 = valid_cls_loss
        plt.plot(x_axix, X1, label="train")
        plt.plot(x_axix, X2, label="valid")
        plt.legend()

    plt.savefig(save_path + "/train_valid_loss_" + suffix + ".png", dpi=400)

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

        self.fc4 = nn.Linear(
            in_features=200, out_features=self.num_classes
        )

    def forward(self, data):
        x = data[0].x
        edge_index = data[0].edge_index
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
    parser.add_argument("--snp_type", type=str, required=True, help="type of snp data type")
    parser.add_argument(
        "--qt_type", type=str, required=False, help="type of QT data type"
    )
    parser.add_argument("--lr", type=float, required=True, help="learning rate")
    parser.add_argument("--epochs", type=int, required=True, help="number of epochs")
    parser.add_argument("--batch_size", type=int, required=True, help="")
    parser.add_argument("--folds", type=int, required=True, help="number of folds")
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

    node_feature_type = args.feature_type
    edge_sparse_type = args.edge_type
    snp_data_type = args.snp_type
    QT_type = args.qt_type
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    num_splits = args.folds
    gnn_model = args.gnn
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