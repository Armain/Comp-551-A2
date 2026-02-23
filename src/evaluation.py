import numpy as np
import pandas as pd
from preprocessing import standardize, get_kfold_indices, RANDOM_SEED
from models import LogisticRegressionSGD


def cross_entropy(y_true, y_prob):
    """Mean binary cross-entropy with clipping for numerical stability."""
    p = np.clip(y_prob, 1e-12, 1 - 1e-12)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def kfold_cv(X_train, y_train, k, lr, batch_size, lam, init_scale=0.0,
             max_epochs=200, patience=10, seed=RANDOM_SEED):
    """Run K-fold CV with per-fold standardization and early stopping.

    Returns pd.Series with mean_/std_ prefixed metrics.
    """
    folds = get_kfold_indices(len(y_train), k=k, seed=seed)
    X_arr = X_train.values if hasattr(X_train, 'values') else X_train
    y_arr = y_train.values if hasattr(y_train, 'values') else y_train

    fold_metrics = pd.DataFrame(index=range(k),
                                columns=["train_ce", "val_ce", "train_acc", "val_acc"],
                                dtype=float)

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

    summary = pd.concat([fold_metrics.mean().add_prefix("mean_"),
                          fold_metrics.std().add_prefix("std_")])
    return summary
