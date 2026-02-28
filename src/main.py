import itertools

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

cwd = Path(__file__).parent.parent

from preprocessing import (
    load_data, train_test_split, standardize, get_kfold_indices, RANDOM_SEED, FEATURE_NAMES
)
from models import LogisticRegressionSGD
from evaluation import kfold_cv
from tuner import RandomizedGridSearchCV, OptunaTunerCV
from utils import config
from plot import (
    plot_training_curves, plot_hp_grid_par_coords, plot_lambda_sweep,
    plot_lambda_sweep_acc, plot_l1_coef_path, plot_l1_sparsity, plot_l1_cv_performance
)


def training_curves(X_train: pd.DataFrame, y_train: pd.Series, use_adam: bool = False) -> pd.DataFrame:
    """Train models across a hyperparameter grid and save training curve plots.

    Args:
        X_train: Feature matrix of shape (n_samples, n_features).
        y_train: Binary spam labels.
        use_adam: If True, use Adam optimizer; otherwise use vanilla SGD.

    Returns:
        training_curves DataFrame with per-config train loss history and final metrics.
    """
    suffix = 'Adam' if use_adam else 'Vanilla SGD'
    print(f"\nTask 1: Training Curves for Logistic Regression ({suffix})")
    X_tr_std, _, _, _ = standardize(X_train)

    batch_sizes = [1, 16, 64]
    learning_rates = [1, 0.1, 0.01, 0.001, 0.0001]
    lambdas = [0, 1e-3]
    epochs = 200

    configs = list(itertools.product(lambdas, batch_sizes, learning_rates))
    training_curves = pd.DataFrame(configs, columns=["lam", "batch_size", "lr"])
    training_curves["train_loss"] = None
    training_curves["final_loss"] = np.nan
    training_curves["final_acc"] = np.nan

    for i, (lam, bs, lr) in tqdm(enumerate(configs), total=len(configs)):
        model = LogisticRegressionSGD(
            n_features=X_tr_std.shape[1], lr=lr, lam=lam, batch_size=bs, use_adam=use_adam
        )
        rng = np.random.default_rng(RANDOM_SEED)
        history = model.fit(X_tr_std, y_train.values, max_epochs=epochs, rng=rng)
        training_curves.at[i, "train_loss"] = history["train_loss"]
        training_curves.at[i, "final_loss"] = history["train_loss"][-1]
        training_curves.at[i, "final_acc"] = history["train_acc"][-1]

    if config.verbose:
        print(training_curves[["lam", "batch_size", "lr", "final_loss", "final_acc"]].to_string(index=False))
    return training_curves


def sweep_regularization(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> pd.DataFrame:
    """Sweep L2 lambda via K-fold CV and evaluate the best lambda on the test set.

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
    for i, lam in tqdm(enumerate(lambdas), total=len(lambdas)):
        cv_summary = kfold_cv(X_train, y_train, k=k, lr=lr, batch_size=bs, lam=lam, epochs=200)
        sweep_results.loc[i, cv_summary.index] = cv_summary

    if config.verbose:
        print(sweep_results[["lam", "mean_train_ce", "std_train_ce", "mean_val_ce", "std_val_ce", "mean_val_acc"]].to_string(index=False))

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
) -> pd.DataFrame:
    """Fit L1 regularization path, run fold-safe CV, and evaluate at the best C.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_test: Test feature matrix.
        y_test: Test labels.

    Returns:
        l1_path_results DataFrame with C, nnz, and test_acc per regularization value.
    """
    print("\nTask 4: L1 Regularization Path")

    X_tr_std, X_te_std, _, _ = standardize(X_train, X_test)

    Cs = np.logspace(-4, 4, 30)
    coef_matrix = np.zeros((len(Cs), X_tr_std.shape[1]))
    nnz_counts = np.zeros(len(Cs), dtype=int)
    l1_path_results = pd.DataFrame({"C": Cs})

    model = LogisticRegression(
        penalty="l1", solver="saga", max_iter=10000,
        tol=1e-4, warm_start=True, random_state=RANDOM_SEED
    )
    for i, C in tqdm(enumerate(Cs), total=len(Cs), desc="L1 Reg Path"):
        model.set_params(C=C)
        model.fit(X_tr_std, y_train.values)
        coef_matrix[i] = model.coef_[0]
        nnz_counts[i] = int(np.sum(np.abs(coef_matrix[i]) > 0))
        l1_path_results.at[i, "nnz"] = nnz_counts[i]
        l1_path_results.at[i, "test_acc"] = model.score(X_te_std, y_test.values)

    if config.verbose:
        print(l1_path_results.to_string(index=False))

    plot_l1_coef_path(Cs, coef_matrix, FEATURE_NAMES)
    plot_l1_sparsity(Cs, nnz_counts)

    k = 20
    X_arr = X_train.values
    y_arr = y_train.values
    folds = get_kfold_indices(len(y_arr), k=k, seed=RANDOM_SEED)

    cv_scores = np.zeros((k, len(Cs)))
    for fold_i, (train_idx, val_idx) in tqdm(enumerate(folds), total=k, desc="L1 Reg CV"):
        X_fold_tr, X_fold_va = X_arr[train_idx], X_arr[val_idx]
        y_fold_tr, y_fold_va = y_arr[train_idx], y_arr[val_idx]
        X_fold_tr_std, X_fold_va_std, _, _ = standardize(X_fold_tr, X_fold_va)

        fold_model = LogisticRegression(
            penalty="l1", solver="saga", max_iter=10000,
            tol=1e-4, warm_start=True, random_state=RANDOM_SEED
        )
        for j, C in enumerate(Cs):
            fold_model.set_params(C=C)
            fold_model.fit(X_fold_tr_std, y_fold_tr)
            cv_scores[fold_i, j] = fold_model.score(X_fold_va_std, y_fold_va)

    mean_scores = cv_scores.mean(axis=0)
    p10_scores = np.percentile(cv_scores, 10, axis=0)
    p90_scores = np.percentile(cv_scores, 90, axis=0)
    plot_l1_cv_performance(Cs, mean_scores, p10_scores, p90_scores)

    best_C = float(Cs[np.argmax(mean_scores)])
    print(f"Best C (CV): {best_C:.4e} (lambda={1/best_C:.4e})")

    X_tr_std, X_te_std, _, _ = standardize(X_train, X_test)
    final_model = LogisticRegression(
        penalty="l1", C=best_C, solver="saga",
        max_iter=10000, tol=1e-4, random_state=RANDOM_SEED
    )
    final_model.fit(X_tr_std, y_train.values)
    test_acc = final_model.score(X_te_std, y_test.values)
    print(f"L1 test accuracy at CV-selected C: {test_acc:.4f}")

    return l1_path_results

if __name__ == "__main__":
    X, y = load_data(cwd / "data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_frac=0.05)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Train spam ratio: {y_train.mean():.3f}")

    # %% Task 1: Training curves across hyperparameter grid
    training_curves_sgd = training_curves(X_train, y_train, use_adam=False)
    plot_training_curves(training_curves_sgd, title="Training Curves (SGD)",
                         figname="training_curves_sgd.png")

    training_curves_adam = training_curves(X_train, y_train, use_adam=True)
    plot_training_curves(training_curves_adam, title="Training Curves (Adam)",
                         figname="training_curves_adam.png")

    # %% Task 2: K-fold CV hyperparameter tuning (Randomized Grid Search or Optuna)
    if config.tuning_method == "optuna":
        tuner = OptunaTunerCV(X_train, y_train)
        figname = "cv_hp_search_optuna.html"
    else:
        tuner = RandomizedGridSearchCV(X_train, y_train)
        figname = "cv_hp_search_random.html"
    hp_results, best_params = tuner.run()
    plot_hp_grid_par_coords(hp_results, figname=figname)

    X_tr_std, X_te_std, _, _ = standardize(X_train, X_test)
    best_model = LogisticRegressionSGD(
        n_features=X_tr_std.shape[1],
        lr=float(best_params["lr"]),
        lam=float(best_params["lam"]),
        batch_size=int(best_params["batch_size"]),
        init_scale=float(best_params["init_scale"]),
        use_adam=bool(best_params["use_adam"]),
        **({} if not best_params["use_adam"] else {
            "beta1": float(best_params["beta1"]),
            "beta2": float(best_params["beta2"]),
        }),
    )
    best_model.fit(X_tr_std, y_train.values, rng=np.random.default_rng(RANDOM_SEED))
    test_ce = best_model.compute_loss(X_te_std, y_test.values)
    test_acc = np.mean(best_model.predict(X_te_std) == y_test.values)
    print(f"\nBest config:\n{best_params.to_string()}")
    print(f"\nTest results:\nCross Entropy Loss={test_ce:.4f}\nAccuracy={test_acc:.4f}")

    # %% Task 3: Lambda sweep with K-fold CV (Vanilla SGD)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, train_frac=0.2)
    sweep_results = sweep_regularization(X_train, y_train, X_test, y_test)
    sweep_results2 = sweep_regularization(X_train2, y_train2, X_test2, y_test2)
    plot_lambda_sweep(sweep_results, sweep_results2, label1=f"n={X_train.shape[0]}", label2=f"n={X_train2.shape[0]}")
    plot_lambda_sweep_acc(sweep_results, sweep_results2, label1=f"n={X_train.shape[0]}", label2=f"n={X_train2.shape[0]}")

    # %% Task 4: L1 regularization path (sklearn)
    l1_path_results = fit_and_plot_l1_regularization_path(X_train, y_train, X_test, y_test)
