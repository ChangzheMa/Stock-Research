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

from model.utils import load_feature_and_label, load_obj, save_obj

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

# 当 LOAD_FROM_CACHE 为 False 时，以下有用
# jq_l1 = load_obj(f"data/IndustryData/jq_l1.pkl")
# stock_list = None
# stock_list = [str(item)[:6] for item in jq_l1['HY002']]     # HY003是工业
# stock_list = [code[2:8] for code in pd.read_csv('data/PreProcess-000300/stock_list.csv')['code'].to_list()]
# stock_list = ['600519', '601318']


def test_by_stock_list(train_data, test_data, stock_label, test_result=None, max_deep=5, qcut=4):
    """
    {
        'jq_l1_HY001_maxdeep5_qcut4': {
            'accuracy': 0.33,
            'feature_importance': [('factor000_1000', 1.06070869e-02), ...]
        },
        ...
    }
    """

    if test_result is None:
        test_result = {}

    result_key = f"{stock_label}_maxdeep{max_deep}_qcut{qcut}"
    if result_key in test_result.keys():
        print(f"stock_label: {stock_label}, max_deep: {max_deep}, qcut: {qcut}, 分类准确率: {test_result[result_key]['accuracy']:.4f} / {1/qcut:.4f} (已有数据)")
        return test_result

    train_features = train_data['feature_list']
    train_labels = train_data['label_list']
    test_features = test_data['feature_list']
    test_labels = test_data['label_list']

    all_labels = np.concatenate([train_labels, test_labels])
    all_labels_categorical, bins = pd.qcut(all_labels, qcut, labels=False, retbins=True)

    train_labels_cate = all_labels_categorical[:len(train_labels)]
    test_labels_cate = all_labels_categorical[len(train_labels):]

    # 实例化模型，可以通过参数调整模型，例如 n_estimators 表示树的数量
    clf = RandomForestClassifier(n_estimators=100, max_depth=max_deep, verbose=1)

    # 训练模型
    clf.fit(train_features, train_labels_cate)

    # 对测试集进行预测
    predicted_labels = clf.predict(test_features)

    # 计算准确率
    accuracy = accuracy_score(test_labels_cate, predicted_labels)
    print(f"stock_label: {stock_label}, max_deep: {max_deep}, qcut: {qcut}, 分类准确率: {accuracy:.4f} / {1/qcut:.4f}")

    feature_import = clf.feature_importances_
    feature_list = [f"factor{('000'+str(i))[-3:]}_1000" for i in range(500)] + [f"factor{('000'+str(j))[-3:]}_1330" for j in range(500)]

    test_result[result_key] = {
        'accuracy': accuracy,
        'accuracy_up_rate': accuracy / (1/qcut),
        'feature_importance': [item for item in sorted([(feature_list[k], feature_import[k]) for k in range(1000)], key=lambda x: x[1], reverse=True) if item[1] > 0]
    }
    return test_result


if __name__ == '__main__':
    test_result = load_obj("data/RandomForestCompare/compare_result.pkl")
    industry_file_name_list = ['jq_l1', 'sw_l1', 'zjw']
    for industry_file_name in industry_file_name_list:
        industry_list = load_obj(f"data/IndustryData/{industry_file_name}.pkl")
        for sub_indu_name in industry_list.keys():
            stock_list = [str(item)[:6] for item in industry_list[sub_indu_name]]
            if len(stock_list) == 0:
                continue
            [train_data, test_data, valid_data] = load_feature_and_label('data/PreProcess-000985',
                                                                         stock_list=stock_list,
                                                                         split_date=['2020-07-01', '2021-07-01'])
            del valid_data
            for max_deep in [5, 10]:
                for qcut in [2, 4, 10, 100]:
                    test_result = test_by_stock_list(train_data, test_data, f"{industry_file_name}_{sub_indu_name}", test_result, max_deep, qcut)
                    save_obj(test_result, "data/RandomForestCompare/compare_result.pkl")
