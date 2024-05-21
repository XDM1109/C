# -*- coding: utf-8 -*-
# @Time    : 2022/7/5 14:44
# @FileName: main.py

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from Load_Test import Load_TestData
from Test import test_evaluate
import torch

temprature = 0.1
lam = 0.8

if __name__=='__main__':
    device = torch.device("cuda:0")

    # test_data = Load_TestData('data/Eexp_150.csv')
    hum_data = Load_TestData('data/Hum_7111.csv')
    mou_data = Load_TestData('data/Mou_7385.csv')
    ara_data = Load_TestData('data/Ara_2125.csv')
    pro6_data = Load_TestData('data/Pro_6318.csv')


    scl_model_path = r"model\Pro1228_CLdna_model.pt"
    print('-=-=-=-=-=-=-=-scl-=-=-=-=-=-=-=-')
    print('***************************** CL hum ***************************** ')
    test_evaluate(scl_model_path, hum_data, device)
    print('***************************** CL ara ***************************** ')
    test_evaluate(scl_model_path, ara_data, device)
    print('***************************** CL mou ***************************** ')
    test_evaluate(scl_model_path, mou_data, device)
    print('***************************** CL pro ***************************** ')
    test_evaluate(scl_model_path, pro6_data, device)
    print('***************************** CL test ***************************** ')
    # test_evaluate(scl_model_path, test_data, device)

