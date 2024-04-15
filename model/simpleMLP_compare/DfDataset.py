import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class DfDataset(Dataset):
    def __init__(self, dataframe, shuffle_every_n=10e5):
        self.df = dataframe
        self.shuffle_every_n = shuffle_every_n
        self.counter = 0
        self.processed_features = []
        self.processed_labels = []
        # self.prepare_data()

    def prepare_data(self):
        grouped = self.df.groupby('date')
        feature_list = []
        label_list = []
        for _, group in grouped:
            # 保持原始顺序的 DataFrame
            original_group = group.reset_index(drop=True)
            # 随机打乱后的 DataFrame
            shuffled_group = group.sample(frac=1).reset_index(drop=True)

            # 提取特征和标签
            feat_orig = original_group.filter(regex='factor.*').values
            feat_shuffled = shuffled_group.filter(regex='factor.*').values
            labels_orig = original_group['ret_next_close_alpha_1000'].values
            labels_shuffled = shuffled_group['ret_next_close_alpha_1000'].values

            # 拼接特征
            combined_features = feat_orig - feat_shuffled
            # 计算标签差异
            label_differences = np.where((labels_orig - labels_shuffled) > 0, 1, 0)

            # 收集所有特征和标签
            feature_list.append(combined_features)
            label_list.append(label_differences)

        # 一次性合并所有特征和标签
        self.processed_features = np.vstack(feature_list)
        self.processed_labels = np.concatenate(label_list)

    def __getitem__(self, index):
        # Shuffle data if counter hits shuffle_every_n
        # if self.counter >= self.shuffle_every_n:
        #     self.prepare_data()
        #     self.counter = 0
        # self.counter += 1

        # Randomly select one sample
        features = torch.tensor(self.processed_features[index], dtype=torch.float32)
        label = torch.tensor(self.processed_labels[index], dtype=torch.int64)

        return features, label

    def __len__(self):
        return len(self.df)
