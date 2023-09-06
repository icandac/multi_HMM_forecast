import numpy as np
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, \
    confusion_matrix
from math import sqrt


def calculate_volatility(log_returns):
    """
    Calculate the volatility of log-returns.

    Parameters:
        log_returns (np.ndarray or pd.Series): The log-returns for which to calculate the volatility.

    Returns:
        float: The calculated volatility.
    """
    mean_return = np.mean(log_returns)
    volatility = np.sqrt(np.mean((log_returns - mean_return) ** 2))
    return volatility


def split_time_series_data(df, train_ratio=0.7, val_ratio=0.15):
    total_size = len(df)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size: train_size + val_size]
    test_data = df.iloc[train_size + val_size:]

    return train_data, val_data, test_data


def initialize_hmm_parameters(data, n_components, random_state=None):
    """
    Initialize the parameters for the Hidden Markov Model.

    Parameters:
        data (ndarray): The observed data (log-returns).
        n_components (int): The number of hidden states.
        random_state (int or None): Random state for reproducibility.

    Returns:
        startprob (ndarray): Initial state occupation distribution.
        transmat (ndarray): State transition matrix.
        means (ndarray): Means of the Gaussian distributions.
        covars (ndarray): Covariances of the Gaussian distributions.
    """
    # Fit a Gaussian Mixture Model to initialize means and covariances
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(data)
    means = gmm.means_
    covars = gmm.covariances_

    # Initialize start probabilities uniformly
    startprob = np.ones(n_components) / n_components

    # Initialize transition probabilities
    # Using the rule from the article: πv|u = (h + 1)/(h + k) when v = u and πv|u = 1/(h + k)
    # when v != u
    h = 1.0  # A suitable positive constant (can be tuned)
    transmat = np.ones((n_components, n_components)) * (1 / (h + n_components))
    np.fill_diagonal(transmat, (h + 1) / (h + n_components))

    return startprob, transmat, means, covars


def fit_hmm_with_multi_start(train_data, n_starts, n_iter, min_states=2, max_states=7,
                             random_state=None, h=1):
    """
    Fit a Gaussian Hidden Markov Model to the training data using multiple initializations.

    Parameters:
        train_data (ndarray): The observed data (log-returns).
        n_starts (int): Number of different initializations.
        n_iter (int): Number of iterations for the EM algorithm.
        min_states (int): Minimum number of states for BIC selection.
        max_states (int): Maximum number of states for BIC selection.
        random_state (int or None): Random state for reproducibility.
        h (int): A suitable positive constant for initializing transition probabilities.

    Returns:
        best_model (GaussianHMM object): The best fitted HMM model.
        best_score (float): The log probability of the best sequence.
    """
    # Determine the best number of states using BIC
    bic_scores = bic_hmm_selection(train_data, n_iter, min_states, max_states)
    best_k = min(bic_scores, key=bic_scores.get)

    best_score = float('-inf')
    best_model = None

    # Deterministic Initialization
    startprob = np.full(best_k, 1.0 / best_k)
    transmat = np.full((best_k, best_k), 1.0 / (h + best_k))
    np.fill_diagonal(transmat, (h + 1.0) / (h + best_k))
    gmm = GaussianMixture(n_components=best_k)
    gmm.fit(train_data)
    means = gmm.means_
    covars = gmm.covariances_
    model = hmm.GaussianHMM(n_components=best_k, covariance_type="full", n_iter=n_iter,
                            random_state=random_state)
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars
    model.fit(train_data)
    logprob = model.score(train_data)
    if logprob > best_score:
        best_score = logprob
        best_model = model

    # Random Initialization
    for i in range(n_starts - 1):
        # Initialize HMM parameters randomly
        startprob = np.random.dirichlet(np.ones(best_k), size=1)[0]
        transmat = np.random.dirichlet(np.ones(best_k), size=best_k)
        means = np.random.randn(best_k, train_data.shape[1])
        covars = np.random.randn(best_k, train_data.shape[1], train_data.shape[1])
        covars = np.array([np.dot(C, C.T) for C in covars])  # Make sure it's positive-definite

        # Create and fit HMM model
        model = hmm.GaussianHMM(n_components=best_k, covariance_type="full", n_iter=n_iter,
                                random_state=random_state)
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.means_ = means
        model.covars_ = covars
        model.fit(train_data)

        # Get the log probability of the best sequence
        logprob = model.score(train_data)

        if logprob > best_score:
            best_score = logprob
            best_model = model

    return best_model, best_score, best_k


def calculate_bic(log_likelihood, num_params, num_samples):
    """
    Calculate the Bayesian Information Criterion (BIC) for a given model.

    Parameters:
    - log_likelihood: float, log-likelihood of the fitted model
    - num_params: int, number of parameters in the model
    - num_samples: int, number of samples (observations)

    Returns:
    - float, BIC value
    """
    return -2 * log_likelihood + num_params * np.log(num_samples)


def bic_hmm_selection(data, n_iter, min_states=2, max_states=5):
    bic_scores = {}
    for n_states in range(min_states, max_states + 1):
        model = hmm.GaussianHMM(
            n_components=n_states, covariance_type="full", n_iter=n_iter
        )
        model.fit(data)
        log_prob = model.score(data)
        n_params = (
            n_states  # number of states
            + n_states * (n_states - 1)  # transition probabilities
            + 2 * n_states * data.shape[1]  # means and covariances
        )
        bic = -2 * log_prob + np.log(len(data)) * n_params
        bic_scores[n_states] = bic
    return bic_scores


def decode_hmm(hmm_model, data):
    """
    Decodes the most likely sequence of hidden states using the Viterbi algorithm.

    Parameters:
    - hmm_model: The trained HMM model from hmmlearn
    - data: np.ndarray, The sequence of observed data

    Returns:
    - logprob: float, The log probability of the produced state sequence
    - best_sequence: np.ndarray, The most likely sequence of states
    """
    logprob, best_sequence = hmm_model.decode(data)
    return logprob, best_sequence


def calculate_hypothetical_returns(best_sequence, val_data):
    """
    Calculate hypothetical returns based on the best sequence of hidden states.

    Parameters:
        best_sequence (ndarray): An array containing the best sequence of hidden states.
        val_data (ndarray): A 2D array containing the log-returns for BTC and ETH.

    Returns:
        hypothetical_returns (ndarray): An array containing the hypothetical returns based
                                         on the best sequence of hidden states.
    """
    # Initialize an array to hold the hypothetical returns
    hypothetical_returns = np.zeros(len(val_data))

    # Loop through the data
    for i in range(1, len(val_data)):
        if best_sequence[i] == 1:  # If the state is 1 (bullish)
            hypothetical_returns[i] = val_data[
                i, 0]  # Use the BTC log-returns as hypothetical returns
        else:  # If the state is 0 (bearish)
            hypothetical_returns[i] = -val_data[
                i, 1]  # Use the negative of ETH log-returns as hypothetical returns

    return hypothetical_returns


def calculate_directional_accuracy(true_values, predicted_values):
    """
    Calculate the directional accuracy of the model's predictions.

    Parameters:
    - true_values: np.array, true labels
    - predicted_values: np.array, predicted labels by the model

    Returns:
    - directional_accuracy: float, the directional accuracy metric
    """
    correct_count = 0
    for true, pred in zip(true_values, predicted_values):
        if (true > 0 and pred > 0) or (true <= 0 and pred <= 0):
            correct_count += 1
    directional_accuracy = correct_count / len(true_values) * 100
    return directional_accuracy


def model_performance_metrics(true_values, predicted_values):
    """
    Evaluate the model's performance using various metrics.

    Parameters:
    - true_values: np.array, true labels
    - predicted_values: np.array, predicted labels by the model

    Returns:
    - metrics: dict, a dictionary containing various performance metrics
    """
    metrics = {}
    metrics['MAE'] = mean_absolute_error(true_values, predicted_values)
    metrics['MSE'] = mean_squared_error(true_values, predicted_values)
    metrics['RMSE'] = sqrt(metrics['MSE'])
    # Exclude zero true_values for MAPE calculation
    non_zero_true_values = true_values[true_values != 0]
    non_zero_predicted_values = predicted_values[true_values != 0]

    if len(non_zero_true_values) == 0:
        metrics['MAPE'] = np.nan  # All true values are zero
    else:
        metrics['MAPE'] = np.mean(np.abs(
            (non_zero_true_values - non_zero_predicted_values) / non_zero_true_values)) * 100
    metrics['Directional Accuracy'] = calculate_directional_accuracy(true_values, predicted_values)

    return metrics


# Metric functions
def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error
    """
    return mean_squared_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_model_performance(true_values, predicted_values):
    """
    Evaluate the performance of the model based on various metrics.

    Parameters:
    - true_values: numpy array, true labels (increase: 1, decrease: 0)
    - predicted_values: numpy array, predicted labels by the model (increase: 1, decrease: 0)

    Returns:
    A dictionary containing various performance metrics.
    """
    accuracy = accuracy_score(true_values, predicted_values)

    unique_labels = np.unique(np.concatenate((true_values, predicted_values)))

    if len(unique_labels) > 2:
        f1 = f1_score(true_values, predicted_values, average='weighted')
    else:
        f1 = f1_score(true_values, predicted_values, average='binary')

    conf_matrix = confusion_matrix(true_values, predicted_values, labels=unique_labels)

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix: {conf_matrix}")

    return {
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Confusion Matrix': conf_matrix
    }


# Function to prepare the true_values and predicted_values arrays for evaluation
def prepare_evaluation_data(array_data, best_sequence):
    """
    Prepare the true and predicted values for evaluation.

    Parameters:
        array_data (ndarray): A 2D array containing the log-returns for BTC and ETH.
        best_sequence (ndarray): An array containing the best sequence of hidden states.

    Returns:
        true_values (ndarray): An array containing the true values based on BTC log-returns.
        predicted_values (ndarray): An array containing the predicted values based on the
                                    best sequence of hidden states.
    """
    # True values: 1 if log-return > 0; 0 otherwise
    true_values = (array_data[:, 0] > 0).astype(
        int)  # Assuming BTC log-returns are in the first column

    # Predicted values: Use the 'best_sequence' to index into the means of the hidden states
    predicted_values = best_sequence  # Here, the best_sequence itself serves as the predicted values

    return true_values, predicted_values
