import pickle

import jqdatasdk as jq
import numpy as np
import pandas as pd
import configparser

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, TensorDataset
from rich.progress import Progress
from tqdm import tqdm

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

df = pd.read_csv("data/PreProcess-000300-Merged/Merged/feature_and_label.csv")

df['date'] = pd.to_datetime(df['date'])
split_date = pd.Timestamp(2022, 12, 31, 23, 59, 59)
split_date2 = pd.Timestamp(2025, 12, 31, 23, 59, 59)

train_df = df[df['date'] < split_date]
test_df = df[(df['date'] >= split_date) & (df['date'] < split_date2)]
valid_df = df[split_date2 <= df['date']]


def prepare_data(df):
    grouped = df.groupby('date')
    feature_list = []
    label_list = []

    for rnd in range(2):
        for _, group in grouped:
            original_group = group.reset_index(drop=True)
            shuffled_group = group.sample(frac=1).reset_index(drop=True)

            feat_orig = original_group.filter(regex='factor.*').values
            feat_shuffled = shuffled_group.filter(regex='factor.*').values
            labels_orig = original_group['ret_next_close_alpha_1000'].values
            labels_shuffled = shuffled_group['ret_next_close_alpha_1000'].values

            combined_features = feat_orig - feat_shuffled
            label_differences = np.where((labels_orig - labels_shuffled) > 0, 1, 0)

            feature_list.append(combined_features)
            label_list.append(label_differences)

    # 一次性合并所有特征和标签
    return np.vstack(feature_list), np.concatenate(label_list)


train_feature, train_label = prepare_data(train_df)
test_feature, test_label = prepare_data(test_df)

print("start to compare")

clf_map = {}
for max_d in [10]:
    # 实例化模型，可以通过参数调整模型，例如 n_estimators 表示树的数量
    clf_map[f"{max_d}"] = RandomForestClassifier(n_estimators=100, max_depth=max_d, verbose=1)

    # 训练模型
    clf_map[f"{max_d}"].fit(train_feature, train_label)

    # 对测试集进行预测
    predicted_labels = clf_map[f"{max_d}"].predict(test_feature)

    # 计算准确率
    accuracy = accuracy_score(test_label, predicted_labels)
    print(f"max_d: {max_d}, 分类准确率: {accuracy:.4f}")
