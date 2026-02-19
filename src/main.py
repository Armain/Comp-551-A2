import itertools
import warnings
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import load_data, train_test_split, standardize, RANDOM_SEED, FEATURE_NAMES
from models import LogisticRegressionSGD
from evaluation import kfold_cv
from plot import (
    plot_training_curves, plot_cv_heatmap, plot_lambda_sweep,
    plot_lambda_sweep_acc, plot_l1_coef_path, plot_l1_sparsity, plot_l1_cv_performance
)

FIG = Path(__file__).parent.parent / "figures"
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Train spam ratio: {y_train.mean():.3f}")

    # Task 1: Training curves across hyperparameter grid
    train_and_plot_curves(X_train, y_train)

    # Task 2: K-fold CV hyperparameter tuning
    best_params = tune_hyperparameters(X_train, y_train)

    # Task 3: Lambda sweep with K-fold CV
    sweep_regularization(X_train, y_train, X_test, y_test)

    # Task 4: L1 regularization path (sklearn)
    fit_l1_regularization_path(X_train, y_train, X_test, y_test)


def train_and_plot_curves(X_train, y_train):
    """Task 1: Train logistic regression with various hyperparams, plot training curves."""
    print("\nTask 1: Training Curves")
    X_tr_std, _, _, _ = standardize(X_train)

    batch_sizes = [1, 16, 64]
    learning_rates = [1, 0.1, 0.01, 0.001, 0.0001]
    epochs = 200

    all_results = {}
    for lam in [0, 1e-3]:
        results_by_batch = {}
        for bs in batch_sizes:
            results_by_batch[bs] = {}
            for lr in learning_rates:
                model = LogisticRegressionSGD(
                    n_features=X_tr_std.shape[1], lr=lr, lam=lam, batch_size=bs
                )
                rng = np.random.default_rng(RANDOM_SEED)
                history = model.fit(X_tr_std, y_train, epochs=epochs, rng=rng)
                results_by_batch[bs][lr] = history["train_loss"]
                print(f"  lam={lam}, B={bs}, lr={lr}: "
                      f"loss={history['train_loss'][-1]:.4f}, acc={history['train_acc'][-1]:.4f}")
        all_results[lam] = results_by_batch

    plot_training_curves(all_results[0], all_results[1e-3], FIG / "task1" / "training_curves.png")


def tune_hyperparameters(X_train, y_train):
    """Task 2: K-fold CV to select best hyperparameters."""
    print("\nTask 2: Hyperparameter Tuning (K-Fold CV)")

    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [1, 16, 64]
    epoch_values = [50, 100, 200]
    lam = 1e-3
    k = 5

    rows = []
    for lr, bs, ep in itertools.product(learning_rates, batch_sizes, epoch_values):
        result = kfold_cv(X_train, y_train, k=k, lr=lr, batch_size=bs, epochs=ep, lam=lam)
        result["lr"] = lr
        result["batch_size"] = bs
        result["epochs"] = ep
        rows.append(result)
        print(f"  lr={lr}, B={bs}, ep={ep}: "
              f"val_ce={result['mean_val_ce']:.4f} +/- {result['std_val_ce']:.4f}, "
              f"val_acc={result['mean_val_acc']:.4f}")

    df = pd.DataFrame(rows)
    best = df.loc[df["mean_val_ce"].idxmin()]
    print(f"\nBest config: lr={best['lr']}, B={best['batch_size']}, epochs={best['epochs']}")
    print(f"  val_ce={best['mean_val_ce']:.4f}, val_acc={best['mean_val_acc']:.4f}")

    save_path = FIG / "task2" / "cv_results_table.png"
    plot_cv_heatmap(rows, save_path)

    return {"lr": best["lr"], "batch_size": best["batch_size"], "epochs": best["epochs"]}


def sweep_regularization(X_train, y_train, X_test, y_test):
    """Task 3: Lambda sweep with K-fold CV, then final test evaluation."""
    print("\nTask 3: Lambda Sweep (Bias-Variance Trade-off)")

    lr, bs, epochs = 0.1, 16, 200
    lambdas = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    k = 5

    rows = []
    for lam in lambdas:
        result = kfold_cv(X_train, y_train, k=k, lr=lr, batch_size=bs, epochs=epochs, lam=lam)
        result["lam"] = lam
        rows.append(result)
        print(f"  lam={lam:.0e}: train_ce={result['mean_train_ce']:.4f}, "
              f"val_ce={result['mean_val_ce']:.4f}, "
              f"val_acc={result['mean_val_acc']:.4f}")

    df = pd.DataFrame(rows)

    plot_lambda_sweep(
        lambdas,
        df["mean_train_ce"].values, df["mean_val_ce"].values,
        df["std_train_ce"].values, df["std_val_ce"].values,
        FIG / "task3" / "lambda_sweep_ce.png",
    )
    plot_lambda_sweep_acc(
        lambdas,
        df["mean_train_acc"].values, df["mean_val_acc"].values,
        df["std_train_acc"].values, df["std_val_acc"].values,
        FIG / "task3" / "lambda_sweep_acc.png",
    )

    best_lam = df.loc[df["mean_val_ce"].idxmin(), "lam"]
    best_val_ce = df.loc[df["mean_val_ce"].idxmin(), "mean_val_ce"]
    print(f"\nBest lambda: {best_lam:.0e} (val_ce={best_val_ce:.4f})")

    X_tr_std, X_te_std, _, _ = standardize(X_train, X_test)
    model = LogisticRegressionSGD(
        n_features=X_tr_std.shape[1], lr=lr, lam=best_lam, batch_size=bs
    )
    rng = np.random.default_rng(RANDOM_SEED)
    model.fit(X_tr_std, y_train, epochs=epochs, rng=rng)

    test_ce = model.compute_loss(X_te_std, y_test)
    test_acc = np.mean(model.predict(X_te_std) == y_test)
    print(f"Final test results (lam={best_lam:.0e}): CE={test_ce:.4f}, Acc={test_acc:.4f}")


def fit_l1_regularization_path(X_train, y_train, X_test, y_test):
    """Task 4: L1 regularization path using sklearn."""
    print("\nTask 4: L1 Regularization Path")
    X_tr_std, X_te_std, _, _ = standardize(X_train, X_test)

    Cs = np.logspace(-4, 4, 30)
    coef_matrix = np.zeros((len(Cs), X_tr_std.shape[1]))
    nnz_counts = np.zeros(len(Cs), dtype=int)

    prev_model = None
    for i, C in enumerate(Cs):
        model = LogisticRegression(
            C=C, solver="saga", l1_ratio=1, max_iter=10000, tol=1e-4,
            warm_start=True, random_state=RANDOM_SEED
        )
        if prev_model is not None:
            model.coef_ = prev_model.coef_.copy()
            model.intercept_ = prev_model.intercept_.copy()
        model.fit(X_tr_std, y_train)
        prev_model = model

        coef_matrix[i] = model.coef_[0]
        nnz_counts[i] = np.sum(np.abs(coef_matrix[i]) > 0)
        print(f"  C={C:.4e}: nnz={nnz_counts[i]}, test_acc={model.score(X_te_std, y_test):.4f}")

    plot_l1_coef_path(Cs, coef_matrix, FEATURE_NAMES, FIG / "task4" / "coef_path.png", top_k=10)
    plot_l1_sparsity(Cs, nnz_counts, FIG / "task4" / "sparsity.png")

    print("  Running LogisticRegressionCV for CV performance plot...")
    cv_model = LogisticRegressionCV(
        solver="saga", l1_ratios=(1,), Cs=Cs, cv=5,
        max_iter=10000, random_state=RANDOM_SEED, scoring="accuracy"
    )
    cv_model.fit(X_tr_std, y_train)
    mean_scores = cv_model.scores_[1].mean(axis=0)

    plot_l1_cv_performance(Cs, mean_scores, FIG / "task4" / "cv_performance.png")

    best_C = cv_model.C_[0]
    print(f"  Best C (CV): {best_C:.4e} (lambda={1/best_C:.4e})")


if __name__ == "__main__":
    main()
