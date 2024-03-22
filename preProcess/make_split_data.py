import jqdatasdk as jq
import numpy as np
import pandas as pd
import configparser

import os
os.chdir('D:\\src\\Stock-Research')

config = configparser.ConfigParser()
config.read('config.ini')

jq.auth(config['DEFAULT']['username'], config['DEFAULT']['password'])
print(jq.get_query_count())

stock_list_unnormed = pd.read_csv('data/PreProcess-000300/stock_list.csv')
stock_list = jq.normalize_code(stock_list_unnormed['code'].to_list())

data_prefix_list = [
    '2018',
    '2019',
    '2020',
    '2021',
    '2022',
    '2023',
    '20231201_20240208',
]

file_name_list_obj = {
    '1000': [f"{prefix}_1000.csv" for prefix in data_prefix_list],
    '1330': [f"{prefix}_1330.csv" for prefix in data_prefix_list],
    'label': ['label.csv', 'label_20231201_20240208.csv']
}

for data_type in file_name_list_obj.keys():
    processed_obj = {}

    for raw_file in file_name_list_obj[data_type]:
        print(f"loading {raw_file}")
        raw_df = pd.read_csv(f"data/RawData-000985/{raw_file}")
        print(f"length: {len(raw_df)}")

        grouped = raw_df.groupby('code')
        dfs = {code: group for code, group in grouped}
        print(f"done grouped")

        for code in dfs.keys():
            if code not in processed_obj.keys():
                processed_obj[code] = pd.DataFrame()
            processed_obj[code] = pd.concat([processed_obj[code], dfs[code]], ignore_index=True)
        print(f"done concat data")

    for key in processed_obj.keys():
        print(f"saving {key}")
        folder = os.path.join(f"data/PreProcess-000985", key[0:6])
        if not os.path.exists(folder):
            os.makedirs(folder)

        if data_type in ['1000', '1330']:
            processed_obj[key].to_csv(os.path.join(folder, f"{data_type}.csv"))
        elif data_type in ['label']:
            split_label = {code: group for code, group in processed_obj[key].groupby('time')}
            split_label['10:00:00'].to_csv(os.path.join(folder, f"{data_type}_1000.csv"))
            split_label['13:30:00'].to_csv(os.path.join(folder, f"{data_type}_1330.csv"))
