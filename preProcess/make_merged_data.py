import jqdatasdk as jq
import numpy as np
import pandas as pd
import configparser

import os

from tqdm import tqdm

from model.utils import load_obj

os.chdir('D:\\src\\Stock-Research')

config = configparser.ConfigParser()
config.read('config.ini')

jq.auth(config['DEFAULT']['username'], config['DEFAULT']['password'])
print(jq.get_query_count())

base_dir = f"data/PreProcess-000985"


def make_merged_data(stock_list):
    folder_path = "data/PreProcess-mostIndex-Merged"
    stock_df_map = {}
    for stock_code in stock_list:
        stock_df_map[stock_code] = pd.read_csv(f"data/PreProcess-000985/{stock_code}/feature_and_label.csv", index_col=0)
    for code, df in stock_df_map.items():
        df.set_index('date', inplace=True)
    combined_df = pd.concat(stock_df_map, names=['stock_code', 'date'])
    if not os.path.exists(f"{folder_path}/Merged"):
        os.mkdir(f"{folder_path}/Merged")
    combined_df.to_csv(f"{folder_path}/Merged/feature_and_label.csv")
    pd.DataFrame({'code': stock_list}).to_csv(f"{folder_path}/stock_list.csv")
    return combined_df


if __name__ == '__main__':
    stock_list = load_obj("data/indexData/index_result.pkl")['most_index_4000']['stock_list']
    print(f"stock_list: {stock_list[:5]}")
    print(f"len(stock_list): {len(stock_list)}")
    combined_df = make_merged_data(stock_list)
