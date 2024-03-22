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

stock_list = pd.read_csv('data/PreProcess-000300/stock_list.csv')

