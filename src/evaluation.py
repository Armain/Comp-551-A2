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


def kfold_cv(X_train, y_train, k, lr, batch_size, epochs, lam, seed=RANDOM_SEED):
    """Run K-fold CV with per-fold standardization.

    Returns dict with mean/std of train_ce, val_ce, train_acc, val_acc.
    """
    folds = get_kfold_indices(len(y_train), k=k, seed=seed)
    metrics = {"train_ce": [], "val_ce": [], "train_acc": [], "val_acc": []}

    for train_idx, val_idx in folds:
        X_tr, X_va = X_train[train_idx], X_train[val_idx]
        y_tr, y_va = y_train[train_idx], y_train[val_idx]
        X_tr_std, X_va_std, _, _ = standardize(X_tr, X_va)

        model = LogisticRegressionSGD(
            n_features=X_tr.shape[1], lr=lr, lam=lam, batch_size=batch_size
        )
        model.fit(X_tr_std, y_tr, epochs=epochs, rng=np.random.default_rng(seed))

        metrics["train_ce"].append(cross_entropy(y_tr, model.predict_proba(X_tr_std)))
        metrics["val_ce"].append(cross_entropy(y_va, model.predict_proba(X_va_std)))
        metrics["train_acc"].append(accuracy(y_tr, model.predict(X_tr_std)))
        metrics["val_acc"].append(accuracy(y_va, model.predict(X_va_std)))

    df = pd.DataFrame(metrics)
    return {**df.mean().add_prefix("mean_").to_dict(),
            **df.std().add_prefix("std_").to_dict()}
