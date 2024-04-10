import jqdatasdk as jq
import numpy as np
import pandas as pd
import configparser

import os

from tqdm import tqdm

os.chdir('D:\\src\\Stock-Research')

config = configparser.ConfigParser()
config.read('config.ini')

jq.auth(config['DEFAULT']['username'], config['DEFAULT']['password'])
print(jq.get_query_count())

base_dir = f"data/PreProcess-000985"


def make_merged_data(stock_list):
    stock_df_map = {}
    for stock_code in stock_list:
        stock_df_map[stock_code] = pd.read_csv(f"data/PreProcess-000985/{stock_code}/feature_and_label.csv", index_col=0)
    for code, df in stock_df_map.items():
        df.set_index('date', inplace=True)
    combined_df = pd.concat(stock_df_map, names=['stock_code', 'date'])
    if not os.path.exists("data/PreProcess-000300-Merged/Merged"):
        os.mkdir("data/PreProcess-000300-Merged/Merged")
    combined_df.to_csv(f"data/PreProcess-000300-Merged/Merged/feature_and_label.csv")
    return combined_df


if __name__ == '__main__':
    stock_list = [item[2:8] for item in pd.read_csv("data/PreProcess-000300-Merged/stock_list.csv")['code']]
    print(stock_list)
    combined_df = make_merged_data(stock_list)
