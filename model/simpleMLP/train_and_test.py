import jqdatasdk as jq
import numpy as np
import pandas as pd
import configparser

from rich.progress import Progress
from model.utils import load_feature_and_label

import os

os.chdir('D:\\src\\Stock-Research')

[train_data, test_data, valid_data] = load_feature_and_label('data/PreProcess-000985',
                                                             split_date=['2020-07-01', '2021-07-01'])
