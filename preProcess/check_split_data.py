import jqdatasdk as jq
import numpy as np
import pandas as pd
import configparser

from rich.progress import Progress

import os
os.chdir('D:\\src\\Stock-Research')

config = configparser.ConfigParser()
config.read('config.ini')

jq.auth(config['DEFAULT']['username'], config['DEFAULT']['password'])
print(jq.get_query_count())

base_dir = f"data/PreProcess-000985"

missing_count = 0

with Progress() as progress:

    task1 = progress.add_task("[red]生成 feature_and_label.csv...", total=len(os.listdir(base_dir)))

    for sub_folder in os.listdir(base_dir):
        folder = os.path.join(base_dir, sub_folder)
        # if os.path.exists(os.path.join(folder, 'missing.log')):
        #     os.remove(os.path.join(folder, 'missing.log'))

        # 文件已存在则不重复处理
        if os.path.exists(os.path.join(folder, 'feature_and_label.csv')):
            progress.update(task1, advance=1)
            continue

        miss_file = False
        for file in ['1000.csv', '1330.csv', 'label_1000.csv', 'label_1330.csv']:
            if not os.path.exists(os.path.join(folder, file)):
                miss_file = True
                break
        # 文件不全则跳过不处理
        if miss_file:
            missing_count += 1
            progress.update(task1, advance=1)
            continue

        # 正常处理并写入文件
        feature_1000 = pd.read_csv(os.path.join(folder, '1000.csv'))
        feature_1330 = pd.read_csv(os.path.join(folder, '1330.csv'))
        feature_1000.drop('code', axis=1, inplace=True)
        feature_1330.drop('code', axis=1, inplace=True)
        feature_merged = pd.merge(feature_1000, feature_1330, on='date', how='inner', suffixes=('_1000', '_1330'))

        label_1000 = pd.read_csv(os.path.join(folder, 'label_1000.csv'))
        label_1330 = pd.read_csv(os.path.join(folder, 'label_1330.csv'))
        label_1000.drop(['time', 'code'], axis=1, inplace=True)
        label_1330.drop(['time', 'code'], axis=1, inplace=True)
        label_merged = pd.merge(label_1000, label_1330, on='date', how='inner', suffixes=('_1000', '_1330'))

        all_merged = pd.merge(feature_merged, label_merged, on='date', how='inner')
        all_merged.to_csv(os.path.join(folder, 'feature_and_label.csv'))

        progress.update(task1, advance=1)


print(f"missing count: {missing_count}")
