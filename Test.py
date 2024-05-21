# -*- coding: utf-8 -*-
# @Time    : 2022/7/5 14:45
# @Author  : 谢冬梅
# @FileName: Test.py

import numpy as np
import torch
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,matthews_corrcoef

import warnings
warnings.filterwarnings("ignore")


def test_evaluate(model_path, test_dataloader,device):

    model = torch.load(model_path)
    model.eval()
    test_accuracy = []
    predict = []
    y_true = []
    for batch in test_dataloader:
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits, _ = model(b_input_ids)
            logits = logits.squeeze()
        preds = torch.round(logits)
        predict += preds.tolist()
        y_true += b_labels.tolist()
    test_accuracy = np.mean(test_accuracy)
    cm = confusion_matrix(y_true, predict)
    TN = cm[0, 0]
    FP = cm[0, 1]
    specificity = TN / (TN + FP)

    # Recall
    print(f'Recall: {recall_score(y_true, predict):.4f}')

    #sp
    print(f'SP: {specificity:.4f}')


    # Accuracy
    print(f'Accuracy: {accuracy_score(y_true, predict):.4f}')

    # MCC
    print(f'Mcc: {matthews_corrcoef(y_true, predict):.4f}')