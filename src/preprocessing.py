import numpy as np
import pandas as pd
from pathlib import Path
from utils import config

RANDOM_SEED = config.random_seed


def _parse_feature_names(names_path: Path) -> list[str]:
    """Parse ordered feature names from spambase.names.

    Args:
        names_path: Path to the spambase.names file.

    Returns:
        List of feature name strings in dataset column order.
    """
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


def load_data(data_dir: Path | str = "data") -> tuple[pd.DataFrame, pd.Series]:
    """Load the spambase CSV and return labelled feature matrix and target.

    Args:
        data_dir: Directory containing spambase.data.

    Returns:
        Tuple of (X DataFrame of shape (n, 57), y spam label Series).
    """
    raw = pd.read_csv(Path(data_dir) / "spambase.data", header=None)
    X = raw.iloc[:, :-1]
    X.columns = FEATURE_NAMES
    y = raw.iloc[:, -1]
    y.name = "spam"
    return X, y


def train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float = 0.05,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Randomly shuffle and split into train/test subsets.

    Args:
        X: Feature matrix.
        y: Target labels.
        train_frac: Fraction of data used for training.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def standardize(
    X_train: np.ndarray | pd.DataFrame,
    X_test: np.ndarray | pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Standardize features using train-set mean and std (prevents data leakage).

    Args:
        X_train: Training features as ndarray or DataFrame.
        X_test: Optional test features to transform with train statistics.

    Returns:
        Tuple of (X_train_std, X_test_std, mean, std). X_test_std is None if
        X_test is not provided.
    """
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


def get_kfold_indices(
    n_samples: int, k: int = 5, seed: int = RANDOM_SEED
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate shuffled K-fold train/validation index splits.

    Args:
        n_samples: Total number of samples.
        k: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        List of k (train_indices, val_indices) tuples.
    """
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
