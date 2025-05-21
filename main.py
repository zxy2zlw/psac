import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from Model import Deepnet
from sklearn.model_selection import KFold
import gc
from index import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, cutoff, AUPRC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc as sk_auc, precision_recall_curve, average_precision_score

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def encode(DNA_sequence):
    torch_sq = []
    encode_ = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for base in DNA_sequence:
        base = encode_[base]
        torch_sq.append(base)
    x = torch.tensor(torch_sq, device=device)
    x = x.flatten()
    return x


def dataProcessing(path):
    file = open(path, "r")
    l1 = len(open(path).readlines())
    count = 0
    Training = [0] * l1
    for line in file:
        Data = line.strip('\n')
        Training[count] = encode(Data)
        count += 1
    return Training


def prepareData(PositiveCSV, NegativeCSV):
    Positive = dataProcessing(PositiveCSV)
    Negative = dataProcessing(NegativeCSV)

    len_data1 = len(Positive)
    len_data2 = len(Negative)

    Positive_y = torch.ones(len_data1, dtype=torch.float32, device=device)
    Negative_y = torch.zeros(len_data2, dtype=torch.float32, device=device)

    for num in range(len(Positive)):
        Positive[num] = tuple((Positive[num], Positive_y[num]))
        Negative[num] = tuple((Negative[num], Negative_y[num]))
    Dataset = Positive + Negative
    return Dataset


def ModelTrainingWithCrossValidation(PositiveCSV, NegativeCSV, bs, net, lr, epochs, PATH, modelname, metrics_dict,
                                     n_splits=5):
    AllData = prepareData(PositiveCSV, NegativeCSV)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=520)

    auc_scores = []
    fpr_list, tpr_list = [], []
    precision_list, recall_list, thresholds_list = [], [], []
    avg_precision_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(AllData)):
        print(f"Starting fold {fold + 1}")
        net = Deepnet(feature=128, dropout=0.3, filter_num=128, seq_len=41).to(device)

        train_dataset = [AllData[i] for i in train_idx]
        test_dataset = [AllData[i] for i in test_idx]

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, drop_last=True)

        criterion = nn.BCELoss()
        opt = torch.optim.Adadelta(net.parameters(), lr=lr, rho=0.9)

        train_loss_list = []
        val_loss_list = []

        for epoch in range(epochs):
            train_pre, val_pre, train_labels, val_labels = [], [], [], []

            net.train()
            for num, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad(set_to_none=True)
                yhat = net.forward(x)
                yhat = yhat.flatten()
                loss = criterion(yhat, y)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 3, norm_type=2)
                opt.step()

                train_pre.extend(yhat.cpu().detach().numpy().flatten().tolist())
                train_labels.extend(y.cpu().detach().numpy().astype('int32').flatten().tolist())

            # Store train loss
            train_loss_list.append(loss.item())

            # Calculate metrics for the training set
            print(f"epoch = {epoch + 1}")
            for key in metrics_dict.keys():
                if key != "auc" and key != "AUPRC":
                    metrics = metrics_dict[key](train_labels, train_pre, thresh=0.5)
                    print("train_" + key + ": " + str(metrics))
                else:
                    metrics = metrics_dict[key](train_labels, train_pre)
                    print("train_" + key + ": " + str(metrics))

            del x, y, yhat, train_labels, train_pre
            gc.collect()

            # Validation phase
            net.eval()
            for num, (x, y) in enumerate(test_loader):
                with torch.no_grad():
                    x = x.to(device)
                    y = y.to(device)
                    yhat = net(x)
                    yhat = yhat.flatten()
                    val_pre.extend(yhat.cpu().detach().numpy().flatten().tolist())
                    val_labels.extend(y.cpu().detach().numpy().astype('int32').flatten().tolist())

            val_loss = criterion(torch.tensor(val_pre, dtype=torch.float32),
                                 torch.tensor(val_labels, dtype=torch.float32))
            val_loss_list.append(val_loss.item())

            # Calculate metrics for the validation set
            fpr, tpr, _ = roc_curve(val_labels, val_pre)
            auc_score = roc_auc_score(val_labels, val_pre)
            auc_scores.append(auc_score)
            fpr_list.append(fpr)
            tpr_list.append(tpr)

            # Calculate precision-recall AUPRC
            precision_vals, recall_vals, thresholds = precision_recall_curve(val_labels, val_pre)
            avg_precision = average_precision_score(val_labels, val_pre)
            precision_list.append(precision_vals)
            recall_list.append(recall_vals)
            thresholds_list.append(thresholds)
            avg_precision_scores.append(avg_precision)

            del x, y, yhat, val_labels, val_pre
            gc.collect()

        # Calculate the average AUC and AUPRC scores for the fold
        print(f"Fold {fold + 1} - AUC: {np.mean(auc_scores):.4f} | AUPRC: {np.mean(avg_precision_scores):.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm']
    for fold in range(n_splits):
        auc_score = auc_scores[fold]
        fpr = fpr_list[fold]
        tpr = tpr_list[fold]
        plt.plot(fpr, tpr, color=colors[fold], label=f"Fold {fold + 1} (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for 5-Fold Cross Validation')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()

    # Plot AUPRC Curve
    plt.figure(figsize=(8, 6))
    for fold in range(n_splits):
        precision_vals = precision_list[fold]
        recall_vals = recall_list[fold]
        avg_precision = avg_precision_scores[fold]
        plt.plot(recall_vals, precision_vals, label=f"Fold {fold + 1} (AUPRC = {avg_precision:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for 5-Fold Cross Validation')
    plt.legend(loc="lower left")
    plt.savefig('pr_curve.png')
    plt.show()

    # Plot loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss Curves')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()


torch.manual_seed(520)
PositivePath = "train_pos.txt"
NegativePath = "train_neg.txt"
net = Deepnet(feature=128, dropout=0.3, filter_num=128, seq_len=41).to(device)
Path = "./DNA_model"
modelname = "my_model"

metrics_dict = {"sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy,
                "mcc": mcc, "auc": auc, "precision": precision, "recall": recall,
                "f1": f1, "AUPRC": AUPRC}

ModelTrainingWithCrossValidation(PositiveCSV=PositivePath, NegativeCSV=NegativePath, bs=128, net=net, lr=0.1, epochs=50,
                                 PATH=Path, modelname=modelname, metrics_dict=metrics_dict)
print("Finish!")
