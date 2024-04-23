import pickle

import jqdatasdk as jq
import numpy as np
import pandas as pd
import configparser

import torch
from torch.utils.data import DataLoader, TensorDataset
from rich.progress import Progress
from tqdm import tqdm

from model.simpleMLP_compare.DfDataset import DfDataset
from model.simpleMLP_compare.SimpleMLP_compare import SimpleMLPCompare
from model.utils import load_feature_and_label, load_obj

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

model = SimpleMLPCompare()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 以上为参数

print(model)
model = model.to(device)

FEATURE_CNT = 200
EPOCH_CNT = 30
STOCK_LIST_TYPE = 'most_index_2000'

df = pd.read_csv(f"data/PreProcess-mostIndex-Merged/Merged/feature_and_label_{STOCK_LIST_TYPE}.csv")

column_names = load_obj("data/PreProcess-000300-Merged/column_names.pkl")[:FEATURE_CNT]
column_names = ["stock_code", "date"] + column_names + ["ret_next_close_alpha_1000"]
df = df.filter(column_names)

df['date'] = pd.to_datetime(df['date'])
split_date = pd.Timestamp(2021, 6, 30, 23, 59, 59)
split_date2 = pd.Timestamp(2023, 12, 31, 23, 59, 59)

train_df = df[df['date'] < split_date]
test_df = df[(df['date'] >= split_date) & (df['date'] < split_date2)]
valid_df = df[split_date2 <= df['date']]
del df

print(f"train len: {len(train_df)}, test len: {len(test_df)}, valid len: {len(valid_df)}")

train_loader = DataLoader(DfDataset(train_df), batch_size=128, shuffle=True)
test_loader = DataLoader(DfDataset(test_df), batch_size=1000, shuffle=True)
valid_loader = DataLoader(DfDataset(valid_df), batch_size=1000, shuffle=True)


for epoch in tqdm(range(EPOCH_CNT)):
    train_loader.dataset.prepare_data(reuse=2)
    test_loader.dataset.prepare_data()

    model.train()
    train_loss = 0.0
    train_correct_cnt = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 清空梯度
        output = model(data)  # 前向传播
        loss = loss_function(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        train_loss += loss.item() * data.size(0)  # 累加损失
        train_correct_cnt += torch.eq(torch.max(output, dim=1)[1], target).sum().item()

    train_loss = train_loss / len(train_loader.dataset)  # 计算平均损失
    train_corr_rate = train_correct_cnt / len(train_loader.dataset)

    model.eval()  # 设置模型为评估模式
    test_loss = 0.0
    test_correct_cnt = 0
    with torch.no_grad():  # 不计算梯度，减少内存消耗
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target)
            test_loss += loss.item() * data.size(0)  # 累加损失
            test_correct_cnt += torch.eq(torch.max(output, dim=1)[1], target).sum().item()

    test_loss = test_loss / len(test_loader.dataset)  # 计算平均损失
    test_corr_rate = test_correct_cnt / len(test_loader.dataset)

    tqdm.write(f'Epoch: {epoch + 1}, '
               f'Training Loss: {train_loss:.4f}, Training Correct Rate: {train_corr_rate:.4f}, '
               f'Test Loss: {test_loss:.4f}, Test Correct Rate: {test_corr_rate:.4f}')


valid_loader.dataset.prepare_data()
model.eval()  # 设置模型为评估模式
valid_loss = 0.0
valid_correct_cnt = 0
with torch.no_grad():  # 不计算梯度，减少内存消耗
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_function(output, target)
        valid_loss += loss.item() * data.size(0)  # 累加损失
        valid_correct_cnt += torch.eq(torch.max(output, dim=1)[1], target).sum().item()

valid_loss = valid_loss / len(valid_loader.dataset)  # 计算平均损失
valid_corr_rate = valid_correct_cnt / len(valid_loader.dataset)

print(f'Valid Loss: {valid_loss:.4f}, Valid Correct Rate: {valid_corr_rate:.4f}')

torch.save(model, f"model/simpleMLP_compare/model/stock({STOCK_LIST_TYPE})_feat{FEATURE_CNT}_epoch{EPOCH_CNT}_corr{valid_corr_rate:.4f}.pth")
