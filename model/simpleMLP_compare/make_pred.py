import pickle
from collections import Counter

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
from model.utils import load_feature_and_label, load_obj, save_obj

import os
os.chdir('D:\\src\\Stock-Research')

config = configparser.ConfigParser()
config.read('config.ini')

jq.auth(config['DEFAULT']['username'], config['DEFAULT']['password'])

import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Training on CPU.")
    device = torch.device("cpu")

# 设置参数
FEATURE_CNT = 200
MODEL_FILE = f"model/simpleMLP_compare/model/feat200_epoch30_corr0.5910.pth"

model = torch.load(MODEL_FILE)
print(model)
model.to(device)
model.eval()

df = pd.read_csv("data/PreProcess-mostIndex-Merged/Merged/feature_and_label.csv")

column_names = load_obj("data/PreProcess-000300-Merged/column_names.pkl")[:FEATURE_CNT]
column_names = ["stock_code", "date"] + column_names + ["ret_next_close_alpha_1000"]
df = df.filter(column_names)

df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] > pd.Timestamp(2021, 6, 30, 23, 59, 59)]

df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)

sub_pred_list = []

grouped = df.groupby('date')
for _, group in tqdm(grouped):
    date = group.iloc[0, 1]
    time = "10:00:00"

    lines = len(group)
    left_codes_list = []
    left_good_labels_list = []

    for i in range(1, lines):
        group_shifted = group.shift(-i)
        group_shifted.iloc[-i:] = group.iloc[0:i]
        left_codes_list.append(group['stock_code'])
        feat = (torch.tensor(group.iloc[:, 2:2 + FEATURE_CNT].to_numpy(), dtype=torch.float32) -
                torch.tensor(group_shifted.iloc[:, 2:2 + FEATURE_CNT].to_numpy(), dtype=torch.float32))
        feat = feat.to(device)
        with torch.no_grad():  # 确保不会计算梯度
            out = model(feat)  # 添加批次维度
        left_good_label = torch.max(out, dim=1)[1].cpu()
        left_good_labels_list.append(left_good_label.numpy())

    if len(left_codes_list) == 0:
        continue

    one_day_result = pd.DataFrame({
        'left': np.concatenate(left_codes_list),
        'left_good': np.concatenate(left_good_labels_list)
    })

    filtered_day_result = one_day_result[one_day_result['left_good'] > 0]
    counter = Counter(one_day_result['left'].tolist() + filtered_day_result['left'].tolist())
    pre_val = 0.2
    code_list = []
    ret_next_close_alpha_list = []
    for (stock_code, cnt) in counter.most_common():
        pre_val -= 10e-4
        code_list.append(stock_code)
        ret_next_close_alpha_list.append(pre_val)
    sub_pred = pd.DataFrame({
        'date': [date] * len(code_list),
        'time': [time] * len(code_list),
        'code': [jq.normalize_code(code) for code in code_list],
        'ret_next_close_alpha': ret_next_close_alpha_list
    })
    sub_pred_list.append(sub_pred)


pred_df = pd.concat(sub_pred_list)
pred_df.to_csv("model/simpleMLP_compare/pred/feat200_epoch30_corr0.5910_submit_result.csv")
