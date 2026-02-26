import numpy as np
import pandas as pd
from preprocessing import standardize, get_kfold_indices, RANDOM_SEED
from models import LogisticRegressionSGD


def cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean binary cross-entropy with clipping for numerical stability.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities in (0, 1).

    Returns:
        Scalar mean cross-entropy loss.
    """
    p = np.clip(y_prob, 1e-12, 1 - 1e-12)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly classified samples.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Scalar accuracy in [0, 1].
    """
    return np.mean(y_true == y_pred)


def kfold_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int,
    lr: float,
    batch_size: int,
    lam: float = 0,
    init_scale: float = 0.0,
    max_epochs: int = 200,
    patience: int = 10,
    seed: int = RANDOM_SEED,
) -> pd.Series:
    """Run K-fold CV with per-fold standardization and early stopping on val loss.

    Args:
        X_train: Feature matrix.
        y_train: Binary labels.
        k: Number of folds.
        lr: Learning rate.
        batch_size: Mini-batch size.
        lam: L2 regularization strength.
        init_scale: Weight initialization std (0.0 = zero init).
        max_epochs: Maximum training epochs per fold.
        patience: Early stopping patience in epochs.
        seed: Random seed for fold splits and model init.

    Returns:
        pd.Series with mean_ and std_ prefixed metrics across folds.
    """
    folds = get_kfold_indices(len(y_train), k=k, seed=seed)
    X_arr = X_train.values
    y_arr = y_train.values

    fold_metrics = pd.DataFrame(
        index=range(k),
        columns=["train_ce", "val_ce", "train_acc", "val_acc"],
        dtype=float,
    )

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_va = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_va = y_arr[train_idx], y_arr[val_idx]
        X_tr_std, X_va_std, _, _ = standardize(X_tr, X_va)

        rng = np.random.default_rng(seed)
        model = LogisticRegressionSGD(
            n_features=X_tr.shape[1], lr=lr, lam=lam,
            batch_size=batch_size, init_scale=init_scale, rng=rng,
        )
        model.fit(X_tr_std, y_tr, max_epochs=max_epochs, rng=rng,
                  X_val=X_va_std, y_val=y_va, patience=patience)

        fold_metrics.loc[fold_i] = [
            cross_entropy(y_tr, model.predict_proba(X_tr_std)),
            cross_entropy(y_va, model.predict_proba(X_va_std)),
            accuracy(y_tr, model.predict(X_tr_std)),
            accuracy(y_va, model.predict(X_va_std)),
        ]

    return pd.concat([fold_metrics.mean().add_prefix("mean_"),
                      fold_metrics.std().add_prefix("std_")])
