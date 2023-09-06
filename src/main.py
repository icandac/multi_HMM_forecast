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
from src.HMM_func import calculate_volatility, split_time_series_data, \
    decode_hmm, calculate_hypothetical_returns, prepare_evaluation_data, \
    evaluate_model_performance, model_performance_metrics, fit_hmm_with_multi_start


def main():
    """
    This is the main function for the cryptocurrency price prediction project.
    The function performs the following steps:

    1. Data Preprocessing: Load and preprocess historical cryptocurrency data.
    2. Model Training: Fit a Hidden Markov Model (HMM) to the log-returns of the data.
       - Bayesian Information Criterion (BIC) is used for model selection.
       - Multi-start strategy is employed for parameter initialization.
    3. Model Evaluation: Backtest the model on a validation dataset.
       - Performance metrics such as accuracy, F1 score, and confusion matrix are calculated.
    4. Forecasting: Predict the hidden states for a test dataset.
    5. Visualization: Generate plots to visualize performance and predictions.

    The function doesn't take any arguments or return any values. All results, including
    performance metrics and visualizations, are printed out during the function's execution.
    """
    ############################# Data preparation and EDA ########################################
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

    # Calculate the log returns
    df_btc_usd['Log_Return'] = np.log(df_btc_usd['close'] / df_btc_usd['close'].shift(1))
    df_btc_usd.dropna(subset=['Log_Return'], inplace=True) # drop the first NaN return row
    df_btc_usd.describe().to_csv(path_or_buf="./outputs/data_summary_btcusd.csv", sep=",",
                                 columns=df_btc_usd.columns)
    df_eth_usd['Log_Return'] = np.log(df_eth_usd['close'] / df_eth_usd['close'].shift(1))
    df_eth_usd.dropna(subset=['Log_Return'], inplace=True)  # drop the first NaN return row
    df_eth_usd.describe().to_csv(path_or_buf="./outputs/data_summary_ethusd.csv", sep=",",
                                 columns=df_btc_usd.columns)
    # Create a new dataframe merging the 'Log_Return' columns of both
    df_log_returns = pd.DataFrame({
        'Log_Return_BTC': df_btc_usd['Log_Return'],
        'Log_Return_ETH': df_eth_usd['Log_Return']
    })
    print(df_btc_usd.Log_Return.min(), df_eth_usd.Log_Return.min(), df_btc_usd.Log_Return.max(),
          df_eth_usd.Log_Return.max(), df_btc_usd.Log_Return.mean(), df_eth_usd.Log_Return.mean(),
          df_btc_usd.Log_Return.std(), df_eth_usd.Log_Return.std())
    # Min-Max scaling for BTC
    btc_min = df_btc_usd['close'].min()
    btc_max = df_btc_usd['close'].max()
    df_btc_usd['close_normalized'] = (df_btc_usd['close'] - btc_min) / (btc_max - btc_min)

    # Min-Max scaling for ETH
    eth_min = df_eth_usd['close'].min()
    eth_max = df_eth_usd['close'].max()
    df_eth_usd['close_normalized'] = (df_eth_usd['close'] - eth_min) / (eth_max - eth_min)

    # Assuming df_btc_usd['Log_Return'] and df_eth_usd['Log_Return'] contain the log-returns for
    # BTC and ETH
    btc_volatility = calculate_volatility(df_btc_usd['Log_Return'])
    eth_volatility = calculate_volatility(df_eth_usd['Log_Return'])

    print(f"BTC Volatility: {btc_volatility}")
    print(f"ETH Volatility: {eth_volatility}")

    # log_returns = df_log_returns.to_numpy()

    train_data, val_data, test_data = split_time_series_data(df_log_returns)
    train_data, val_data, test_data = train_data.to_numpy(), val_data.to_numpy(), \
                                      test_data.to_numpy()

    ################################# Fit HMM #####################################################
    n_iter = 50
    n_starts = 100

    # Fit the HMM model to the training data
    model, bic_score, best_k = fit_hmm_with_multi_start(train_data, n_starts, n_iter)
    print(f"startprob = {model.startprob_}\n transmat = {model.transmat_}\n means = "
          f"{model.means_}\n covars = {model.covars_}")
    print(f"The multivariate HM model predicts to have {best_k} different market behaviours.")

    # For decoding the most likely sequence of hidden states, use the Viterbi algorithm
    logprob, best_sequence = decode_hmm(model, train_data)
    print("Log probability of the best sequence:", logprob)
    print("Best sequence of hidden states:", best_sequence)

    # Validation
    logprob_val, best_sequence_val = decode_hmm(model, val_data)
    hypothetical_returns = calculate_hypothetical_returns(best_sequence_val, val_data)
    print("Log probability of the best sequence for the validation data:", logprob_val)
    print("Best sequence of hidden states for the validation data:", best_sequence_val)

    true_val, predicted_val = prepare_evaluation_data(val_data, best_sequence_val)
    metrics_val = evaluate_model_performance(true_val, predicted_val)
    print("Performance Metrics on Validation Data:", metrics_val)

    # Test
    best_sequence_test = model.predict(test_data)
    true_test, predicted_test = prepare_evaluation_data(test_data, best_sequence_test)
    metrics_test = evaluate_model_performance(true_test, predicted_test)
    print("Performance Metrics on Test Data:", metrics_test)
    metrics = model_performance_metrics(true_test, predicted_test)
    print(metrics)

    ################################### Visualization #############################################
    # Replication of Figure 1 in the article
    time = df_btc_usd.index
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # Plotting the BTC and ETH price
    sns.lineplot(x=time, y=df_eth_usd["close_normalized"], ax=axes[0], label="ETH-normalized")
    sns.lineplot(x=time, y=df_btc_usd["close_normalized"], ax=axes[0], label="BTC-normalized")
    axes[0].set_title('Price Over Time')
    axes[0].set_xlabel('Time [days]')
    axes[0].set_ylabel('Value')
    axes[0].grid(True)

    # Plotting the log-returns of BTC and ETH
    sns.lineplot(x=time, y=df_eth_usd["Log_Return"], ax=axes[1], label="ETH")
    sns.lineplot(x=time, y=df_btc_usd["Log_Return"], ax=axes[1], label="BTC")
    axes[1].set_title('Log-Returns Over Time')
    axes[1].set_xlabel('Time [days]')
    axes[1].set_ylabel('Value')
    axes[1].grid(True)

    # Show the plot
    plt.tight_layout()
    # plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(df_btc_usd["Log_Return"], bins=50, kde=True, label='BTC', ax=ax)
    sns.histplot(df_eth_usd["Log_Return"], bins=50, kde=True, label='ETH', ax=ax)

    ax.set_title('Distribution of Log-Returns for BTC and ETH')
    ax.set_xlabel('Log-Returns')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    # Calculate the rolling standard deviation (volatility)
    df_btc_usd['Rolling_Volatility'] = df_btc_usd["Log_Return"].rolling(window=21).std()
    df_eth_usd['Rolling_Volatility'] = df_eth_usd["Log_Return"].rolling(window=21).std()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting the rolling volatilities
    sns.lineplot(x=df_eth_usd.index, y=df_eth_usd['Rolling_Volatility'], ax=ax, label='ETH')
    sns.lineplot(x=df_btc_usd.index, y=df_btc_usd['Rolling_Volatility'], ax=ax, label='BTC')

    ax.set_title(f'Rolling Volatility Over Time (Window: {21} days)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Rolling Volatility')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

