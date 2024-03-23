import numpy as np
import pandas as pd

from rich.progress import Progress
from tqdm import tqdm

import os
os.chdir('D:\\src\\Stock-Research')


def split_df_by_dates(df, split_dates):
    """
    按给定的时间点分割 DataFrame。

    参数:
    df: DataFrame，需要被分割的 DataFrame。
    split_dates: list，包含分割点的字符串列表。

    返回:
    sub_df_list: 包含分割后的子 DataFrame 的列表。
    """
    df['date'] = pd.to_datetime(df['date'])
    split_dates = pd.to_datetime(split_dates).tolist()
    start_date = df['date'].min() - pd.Timedelta(days=1)
    end_date = df['date'].max() + pd.Timedelta(days=1)
    split_dates = [start_date] + split_dates + [end_date]

    sub_df_list = []
    for idx in range(len(split_dates) - 1):
        mask = (df['date'] >= split_dates[idx]) & (df['date'] < split_dates[idx+1])
        sub_df = df.loc[mask]
        sub_df_list.append(sub_df)

    return sub_df_list


def load_feature_and_label(base_path, label_column_name='ret_next_close_alpha_1000', stock_list=None, split_date=None):
    """
    :param base_path: 从哪个目录读取数据
    :param label_column_name: label列的名字
    :param stock_list: 读取哪些股票
    :param split_date: 分割日期
    :return: [{'feature_list': [], 'label_list': []}, {'feature_list': [], 'label_list': []}]，有几段取决于分割日期有几个
    """
    if split_date is None:
        split_date = []

    # 构建形状正确的返回结果，里面数据是空的
    result = [
        {'feature_list': [], 'label_list': []}
    ]
    for i in range(len(split_date)):
        result.append({'feature_list': [], 'label_list': []})

    folder_list = os.listdir(base_path) if stock_list is None else stock_list
    for folder in tqdm(folder_list):
        file_path = os.path.join(base_path, folder, 'feature_and_label.csv')
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        df = df.drop(['Unnamed: 0', 'Unnamed: 0_1000_x', 'Unnamed: 0_1330_x', 'Unnamed: 0_1000_y', 'Unnamed: 0_1330_y'], axis=1)
        sub_df_list = split_df_by_dates(df, split_date)

        for idx in range(len(sub_df_list)):
            sub_df = sub_df_list[idx]
            result[idx]['feature_list'].append(sub_df.iloc[:, 1:1001].to_numpy())
            result[idx]['label_list'].append(sub_df[label_column_name].to_list())

    for result_obj in result:
        result_obj['feature_list'] = np.concatenate(result_obj['feature_list'], axis=0)
        result_obj['label_list'] = np.concatenate(result_obj['label_list'], axis=0)

    return result

