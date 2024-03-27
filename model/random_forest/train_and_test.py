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

# 设置参数

LOAD_FROM_CACHE = False
# 当 LOAD_FROM_CACHE 为 False 时，以下有用
# jq_l1 = load_obj(f"data/IndustryData/jq_l1.pkl")
# stock_list = None
# stock_list = [str(item)[:6] for item in jq_l1['HY002']]     # HY003是工业
stock_list = [code[2:8] for code in pd.read_csv('data/PreProcess-000300/stock_list.csv')['code'].to_list()]
# stock_list = ['600519', '601318']

# 以上为参数

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


train_features = train_data['feature_list']
train_labels = train_data['label_list']

test_features = test_data['feature_list']
test_labels = test_data['label_list']

all_labels = np.concatenate([train_labels, test_labels])
all_labels_categorical, bins = pd.qcut(all_labels, 4, labels=False, retbins=True)

train_labels_cate = all_labels_categorical[:len(train_labels)]
test_labels_cate = all_labels_categorical[len(train_labels):]

for max_d in [5]:
    # 实例化模型，可以通过参数调整模型，例如 n_estimators 表示树的数量
    clf = RandomForestClassifier(n_estimators=100, max_depth=max_d, verbose=1)

    # 训练模型
    clf.fit(train_features, train_labels_cate)

    # 对测试集进行预测
    predicted_labels = clf.predict(test_features)

    # 计算准确率
    accuracy = accuracy_score(test_labels_cate, predicted_labels)
    print(f"max_d: {max_d}, 分类准确率: {accuracy:.4f}")
