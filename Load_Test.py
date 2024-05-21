# -*- coding: utf-8 -*-
# @Time    : 2023/3/3 10:15
# @Author  : 谢冬梅
# @FileName: Load_Test.py

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

def one_hot_coding(data):
    '''
    :param data: DNA序列
    :return: one_hot编码的DNA序列
    '''
    word_enbeding = {"A": [1, 0, 0, 0],
                     "T": [0, 1, 0, 0],
                     "G": [0, 0, 1, 0],
                     "C": [0, 0, 0, 1]
                     }
    data_x = []
    for i in data:
        data_x_one = []
        for j in i:
            data_x_one.append(word_enbeding[j])
        while len(data_x_one) < 300:
            data_x_one.append([0, 0, 0, 0])
        data_x.append(data_x_one[:300])
    return data_x
def Load_TestData(path):
    data = pd.read_csv(path)
    data_text = data['squeue']
    data_label = data["label"]
    train_inputs = torch.tensor( one_hot_coding(data_text), dtype=torch.float)

    train_labels = torch.tensor(data_label)
    batch_size = 32

    train_data = TensorDataset(train_inputs, train_labels)
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size)

    return train_dataloader