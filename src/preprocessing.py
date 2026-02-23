import numpy as np
import pandas as pd
from pathlib import Path
from utils import config

RANDOM_SEED = config.random_seed


def _parse_feature_names(names_path):
    """Parse feature names from spambase.names file."""
    names = []
    with open(names_path, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line and not line.startswith("|") and not line.startswith("1,"):
                name = line.split(":")[0].strip()
                names.append(name)
    return names


FEATURE_NAMES = _parse_feature_names(
    Path(__file__).parent.parent / "data" / "spambase.names"
)


def load_data(data_dir="data"):
    """Load spambase dataset. Returns X as DataFrame (n, 57) and y as Series (n,)."""
    raw = pd.read_csv(Path(data_dir) / "spambase.data", header=None)
    X = raw.iloc[:, :-1]
    X.columns = FEATURE_NAMES
    y = raw.iloc[:, -1]
    y.name = "spam"
    return X, y


def train_test_split(X, y, train_frac=0.05, seed=RANDOM_SEED):
    """Shuffle and split into train/test. Returns DataFrames/Series."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def standardize(X_train, X_test=None):
    """Standardize using train statistics. Accepts DataFrame or ndarray, returns ndarrays."""
    X_tr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std == 0] = 1.0
    X_train_std = (X_tr - mean) / std
    X_test_std = None
    if X_test is not None:
        X_te = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        X_test_std = (X_te - mean) / std
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
