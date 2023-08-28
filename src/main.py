# Python libraries
import os
from pathlib import Path

# Third-party packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Internal modules
from src.utils import add_to_log


def main():
    """
    This model...
    """
    # Read the input file
    current_dir = Path.cwd()
    Path('outputs').mkdir(parents=True, exist_ok=True)

    data_path1 = os.path.join(current_dir, '21co_drs_ex2_btc_usd_data.csv')
    data_path2 = os.path.join(current_dir, '21co_drs_ex2_eth_usd_data.csv')

    # Directly read the Excel files into a DataFrame
    with open(data_path1, "r", encoding="utf-8-sig") as read_file1:
        btc_usd = pd.read_csv(read_file1)
    df_btc_usd = btc_usd.copy()
    df_btc_usd.timestamp = pd.to_datetime(df_btc_usd.timestamp).dt.date
    df_btc_usd.set_index("timestamp", inplace=True)

    with open(data_path2, "r", encoding="utf-8-sig") as read_file2:
        eth_usd = pd.read_csv(read_file2)
    df_eth_usd = eth_usd.copy()
    df_eth_usd.timestamp = pd.to_datetime(df_eth_usd.timestamp).dt.date
    df_eth_usd.set_index("timestamp", inplace=True)

    add_to_log("The given data is read.")

    # # Clean and sort the data
    # df = data_raw.sort_values(by="Date")
    # df.set_index("Date", inplace=True)
    # df.index = df.index.date
    # print(df.isna().sum())  # No NaN values in the dataset
    # print(df.isnull().sum())  # No null values in the dataset
    # # df.describe().to_csv(path_or_buf="./outputs/data_summary.csv", sep=",", columns=df.columns)
    # # There is one negative value in "Adj Close" column, let's clean that outlier
    # df = df[df["Adj Close"] >= 0]
    # df.describe().to_csv(path_or_buf="./outputs/data_summary.csv", sep=",", columns=df.columns)

    # Visualization

    a = 5
