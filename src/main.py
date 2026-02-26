import itertools
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

cwd = Path(__file__).parent.parent

from preprocessing import (
    load_data, train_test_split, standardize, get_kfold_indices, RANDOM_SEED, FEATURE_NAMES
)
from models import LogisticRegressionSGD
from evaluation import kfold_cv
from plot import (
    plot_training_curves, plot_hp_grid_par_coords, plot_lambda_sweep,
    plot_lambda_sweep_acc, plot_l1_coef_path, plot_l1_sparsity, plot_l1_cv_performance
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def train_and_plot_curves(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Train models across a hyperparameter grid and save training curve plots.

    Args:
        X_train: Feature matrix of shape (n_samples, n_features).
        y_train: Binary spam labels.
    """
    print("\nTask 1: Training Curves")
    X_tr_std, _, _, _ = standardize(X_train)

    batch_sizes = [1, 16, 64]
    learning_rates = [1, 0.1, 0.01, 0.001, 0.0001]
    lambdas = [0, 1e-3]
    epochs = 200

    configs = list(itertools.product(lambdas, batch_sizes, learning_rates))
    training_curves = pd.DataFrame(configs, columns=["lam", "batch_size", "lr"])
    training_curves["train_loss"] = None

    for i, (lam, bs, lr) in enumerate(configs):
        model = LogisticRegressionSGD(
            n_features=X_tr_std.shape[1], lr=lr, lam=lam, batch_size=bs
        )
        rng = np.random.default_rng(RANDOM_SEED)
        history = model.fit(X_tr_std, y_train.values, max_epochs=epochs, rng=rng)
        training_curves.at[i, "train_loss"] = history["train_loss"]
        print(f"  lam={lam}, B={bs}, lr={lr}: "
              f"loss={history['train_loss'][-1]:.4f}, acc={history['train_acc'][-1]:.4f}")

    return training_curves


def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Run 5-fold CV over a 5x5x5 hyperparameter grid and return results and best config.

    Args:
        X_train: Feature matrix of shape (n_samples, n_features).
        y_train: Binary spam labels.

    Returns:
        Tuple of (hp_results DataFrame with CV metrics, best_params dict).
    """
    print("\nTask 2: Hyperparameter Tuning (K-Fold CV)")

    learning_rates = [1, 0.1, 0.01, 0.001, 0.0001]
    batch_sizes = [1, 4, 16, 32, 64]
    init_scales = [0.0, 0.001, 0.01, 0.1, 1.0]
    k = 5

    hp_grid = list(itertools.product(learning_rates, batch_sizes, init_scales))
    hp_results = pd.DataFrame(hp_grid, columns=["lr", "batch_size", "init_scale"])

    for i, (lr, bs, scale) in enumerate(hp_grid):
        cv_summary = kfold_cv(X_train, y_train, k=k, lr=lr, batch_size=bs, init_scale=scale)
        hp_results.loc[i, cv_summary.index] = cv_summary
        print(f"  lr={lr}, B={bs}, init_scale={scale}: "
              f"val_ce={cv_summary['mean_val_ce']:.4f} +/- {cv_summary['std_val_ce']:.4f}, "
              f"val_acc={cv_summary['mean_val_acc']:.4f}")

    best = hp_results.loc[hp_results["mean_val_ce"].idxmin()]
    print(f"\nBest config: lr={best['lr']}, B={best['batch_size']}, init_scale={best['init_scale']}")
    print(f"  val_ce={best['mean_val_ce']:.4f}, val_acc={best['mean_val_acc']:.4f}")

    best_params = {"lr": best["lr"], "batch_size": best["batch_size"], "init_scale": best["init_scale"]}
    return hp_results, best_params


def sweep_regularization(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> pd.DataFrame:
    """Sweep L2 lambda via 20-fold CV and evaluate the best lambda on the test set.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_test: Test feature matrix.
        y_test: Test labels.

    Returns:
        sweep_results DataFrame with mean/std CV metrics per lambda.
    """
    print("\nTask 3: Lambda Sweep (Bias-Variance Trade-off)")

    lr, bs = 0.1, 16
    lambdas = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    k = 20

    sweep_results = pd.DataFrame({"lam": lambdas})
    for i, lam in enumerate(lambdas):
        cv_summary = kfold_cv(X_train, y_train, k=k, lr=lr, batch_size=bs, lam=lam)
        sweep_results.loc[i, cv_summary.index] = cv_summary
        print(f"  lam={lam:.0e}: train_ce={cv_summary['mean_train_ce']:.4f}, "
              f"val_ce={cv_summary['mean_val_ce']:.4f}, "
              f"val_acc={cv_summary['mean_val_acc']:.4f}")

    best_idx = sweep_results["mean_val_ce"].idxmin()
    best_lam = sweep_results.loc[best_idx, "lam"]
    print(f"\nBest lambda: {best_lam:.0e} (val_ce={sweep_results.loc[best_idx, 'mean_val_ce']:.4f})")

    X_tr_std, X_te_std, _, _ = standardize(X_train, X_test)
    model = LogisticRegressionSGD(
        n_features=X_tr_std.shape[1], lr=lr, lam=best_lam, batch_size=bs
    )
    rng = np.random.default_rng(RANDOM_SEED)
    model.fit(X_tr_std, y_train.values, rng=rng)

    test_ce = model.compute_loss(X_te_std, y_test.values)
    test_acc = np.mean(model.predict(X_te_std) == y_test.values)
    print(f"Final test results (lam={best_lam:.0e}): CE={test_ce:.4f}, Acc={test_acc:.4f}")
    return sweep_results

def fit_and_plot_l1_regularization_path(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> None:
    """Fit L1 regularization path, run fold-safe CV, and evaluate at the best C.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_test: Test feature matrix.
        y_test: Test labels.
    """
    print("\nTask 4: L1 Regularization Path")

    X_tr_std, X_te_std, _, _ = standardize(X_train, X_test)

    Cs = np.logspace(-4, 4, 30)
    coef_matrix = np.zeros((len(Cs), X_tr_std.shape[1]))
    nnz_counts = np.zeros(len(Cs), dtype=int)

    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=10000,
        tol=1e-4,
        warm_start=True,
        random_state=RANDOM_SEED
    )
    for i, C in enumerate(Cs):
        model.set_params(C=C)
        model.fit(X_tr_std, y_train.values)

        coef_matrix[i] = model.coef_[0]
        nnz_counts[i] = int(np.sum(np.abs(coef_matrix[i]) > 0))
        test_acc = model.score(X_te_std, y_test.values)
        print(f"  C={C:.4e}: nnz={nnz_counts[i]}, test_acc={test_acc:.4f}")

    plot_l1_coef_path(Cs, coef_matrix, FEATURE_NAMES)
    plot_l1_sparsity(Cs, nnz_counts)

    print("  Running manual CV for performance plot...")
    k = 20
    X_arr = X_train.values
    y_arr = y_train.values
    folds = get_kfold_indices(len(y_arr), k=k, seed=RANDOM_SEED)

    cv_scores = np.zeros((k, len(Cs)))
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        X_fold_tr, X_fold_va = X_arr[train_idx], X_arr[val_idx]
        y_fold_tr, y_fold_va = y_arr[train_idx], y_arr[val_idx]
        X_fold_tr_std, X_fold_va_std, _, _ = standardize(X_fold_tr, X_fold_va)

        fold_model = LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=10000,
            tol=1e-4,
            warm_start=True,
            random_state=RANDOM_SEED
        )
        for j, C in enumerate(Cs):
            fold_model.set_params(C=C)
            fold_model.fit(X_fold_tr_std, y_fold_tr)
            cv_scores[fold_i, j] = fold_model.score(X_fold_va_std, y_fold_va)

    mean_scores = cv_scores.mean(axis=0)
    std_scores = cv_scores.std(axis=0)
    plot_l1_cv_performance(Cs, mean_scores, std_scores)

    best_C = float(Cs[np.argmax(mean_scores)])
    print(f"  Best C (CV): {best_C:.4e} (lambda={1/best_C:.4e})")

    # Evaluate on test set, at the CV-selected C.
    X_tr_std, X_te_std, _, _ = standardize(X_train, X_test)
    final_model = LogisticRegression(
        penalty="l1",
        C=best_C,
        solver="saga",
        max_iter=10000,
        tol=1e-4,
        random_state=RANDOM_SEED
    )
    final_model.fit(X_tr_std, y_train.values)
    test_acc = final_model.score(X_te_std, y_test.values)
    print(f"  One-time L1 test accuracy at CV-selected C: {test_acc:.4f}")


if __name__ == "__main__":
    X, y = load_data(cwd / "data")
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Train spam ratio: {y_train.mean():.3f}")

    # Task 1: Training curves across hyperparameter grid
    training_curves = train_and_plot_curves(X_train, y_train)
    plot_training_curves(training_curves)

    # Task 2: K-fold CV hyperparameter tuning
    hp_results, best_params = tune_hyperparameters(X_train, y_train)
    plot_hp_grid_par_coords(hp_results)

    # Task 3: Lambda sweep with K-fold CV
    sweep_results = sweep_regularization(X_train, y_train, X_test, y_test)
    plot_lambda_sweep(sweep_results)
    plot_lambda_sweep_acc(sweep_results)

    # Task 4: L1 regularization path (sklearn)
    fit_and_plot_l1_regularization_path(X_train, y_train, X_test, y_test)
