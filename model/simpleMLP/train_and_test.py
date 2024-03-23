import pickle

import jqdatasdk as jq
import numpy as np
import pandas as pd
import configparser

import torch
from torch.utils.data import DataLoader, TensorDataset
from rich.progress import Progress
from tqdm import tqdm

from model.simpleMLP.SimpleMLP import SimpleMLP
from model.utils import load_feature_and_label

import os
os.chdir('D:\\src\\Stock-Research')

import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Training on CPU.")
    device = torch.device("cpu")

# 设置参数

model = SimpleMLP()
loss_function = torch.nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters())

LOAD_FROM_CACHE = False
# 当 LOAD_FROM_CACHE 为 False 时，以下有用
stock_list = [code[2:8] for code in pd.read_csv('data/PreProcess-000300/stock_list.csv')['code'].to_list()]
# stock_list = ['600519', '601318']

# 以上为参数

print(model)
model = model.to(device)

cache_folder = 'model/cache/linear'
cache_file_list = [f'{cache_folder}/train.pkl', f'{cache_folder}/test.pkl', f'{cache_folder}/valid.pkl']

all_cache_file_exists = True
for cache_file in cache_file_list:
    if not os.path.exists(cache_file):
        all_cache_file_exists = False
        break

if not LOAD_FROM_CACHE or not all_cache_file_exists:
    if not os.path.exists(cache_folder):
        os.makedirs('')
    result = load_feature_and_label('data/PreProcess-000985', stock_list=stock_list, split_date=['2020-07-01', '2021-07-01'])
    for idx in range(len(cache_file_list)):
        with open(cache_file_list[idx], 'wb') as f:
            pickle.dump(result[idx], f)
    del result

with open(cache_file_list[0], 'rb') as f:
    train_data = pickle.load(f)

with open(cache_file_list[1], 'rb') as f:
    test_data = pickle.load(f)

feature_train = torch.Tensor(train_data['feature_list'])
label_train = torch.Tensor(train_data['label_list'])
train_loader = DataLoader(TensorDataset(feature_train, label_train), batch_size=64, shuffle=True)

feature_test = torch.Tensor(test_data['feature_list'])
label_test = torch.Tensor(test_data['label_list'])
test_loader = DataLoader(TensorDataset(feature_test, label_test), batch_size=1000, shuffle=True)

for epoch in tqdm(range(20)):
    model.train()
    train_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 清空梯度
        output = model(data)  # 前向传播
        loss = loss_function(output, target.view(-1, 1))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        train_loss += loss.item() * data.size(0)  # 累加损失

    train_loss = train_loss / len(train_loader)  # 计算平均损失

    model.eval()  # 设置模型为评估模式
    test_loss = 0.0
    with torch.no_grad():  # 不计算梯度，减少内存消耗
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target.view(-1, 1))
            test_loss += loss.item() * data.size(0)  # 累加损失

    test_loss = test_loss / len(test_loader)  # 计算平均损失

    tqdm.write(f'Epoch: {epoch + 1}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

del train_data
del test_data
del feature_train
del label_train
del train_loader
del feature_test
del label_test
del test_loader

# 评估模型
with open(cache_file_list[2], 'rb') as f:
    valid_data = pickle.load(f)

feature_valid = torch.Tensor(valid_data['feature_list'])
label_valid = torch.Tensor(valid_data['label_list'])
valid_loader = DataLoader(TensorDataset(feature_valid, label_valid), batch_size=1000, shuffle=True)

model.eval()  # 设置模型为评估模式
valid_loss = 0.0
with torch.no_grad():  # 不计算梯度，减少内存消耗
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_function(output, target.view(-1, 1))
        valid_loss += loss.item() * data.size(0)  # 累加损失

valid_loss = valid_loss / len(valid_loader)  # 计算平均损失

print(f'Valid Loss: {valid_loss:.4f}')
