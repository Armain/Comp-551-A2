import numpy as np
import pandas as pd
import os

RANDOM_SEED = 2026

FEATURE_NAMES = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!",
    "char_freq_$", "char_freq_#",
    "capital_run_length_average", "capital_run_length_longest",
    "capital_run_length_total",
]


def load_data(data_dir="data"):
    """Load spambase dataset. Returns X (n, 57) and y (n,) as numpy arrays."""
    path = os.path.join(data_dir, "spambase.data")
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def train_test_split(X, y, train_frac=0.05, seed=RANDOM_SEED):
    """Shuffle and split into train/test. Returns X_train, X_test, y_train, y_test."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train, X_test=None):
    """Standardize using train statistics. Returns standardized arrays, mean, std."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero for constant features
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std if X_test is not None else None
    return X_train_std, X_test_std, mean, std


def get_kfold_indices(n_samples, k=5, seed=RANDOM_SEED):
    """Return list of (train_indices, val_indices) for K-fold CV."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    fold_size = n_samples // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n_samples
        val_idx = idx[start:end]
        train_idx = np.concatenate([idx[:start], idx[end:]])
        folds.append((train_idx, val_idx))
    return folds
