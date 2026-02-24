import itertools
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

cwd = Path(__file__).parent.parent

from preprocessing import load_data, train_test_split, standardize, RANDOM_SEED, FEATURE_NAMES
from models import LogisticRegressionSGD
from evaluation import kfold_cv
from plot import (
    plot_training_curves, plot_cv_heatmap, plot_lambda_sweep,
    plot_lambda_sweep_acc, plot_l1_coef_path, plot_l1_sparsity, plot_l1_cv_performance
)

warnings.simplefilter(action='ignore', category=FutureWarning)

def train_and_plot_curves(X_train, y_train):
    """Task 1: Train logistic regression with various hyperparams, plot training curves."""
    print("\nTask 1: Training Curves")
    X_tr_std, _, _, _ = standardize(X_train)
    y_arr = y_train.values if hasattr(y_train, 'values') else y_train

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
        history = model.fit(X_tr_std, y_arr, max_epochs=epochs, rng=rng)
        training_curves.at[i, "train_loss"] = history["train_loss"]
        print(f"  lam={lam}, B={bs}, lr={lr}: "
              f"loss={history['train_loss'][-1]:.4f}, acc={history['train_acc'][-1]:.4f}")

    plot_training_curves(training_curves)


def tune_hyperparameters(X_train, y_train):
    """Task 2: K-fold CV to select best hyperparameters."""
    print("\nTask 2: Hyperparameter Tuning (K-Fold CV)")

    learning_rates = [1, 0.1, 0.01, 0.001, 0.0001]
    batch_sizes = [1, 4, 16, 32, 64]
    init_scales = [0.0, 0.001, 0.01, 0.1, 1.0]
    lam = 1e-3
    k = 5

    hp_grid = list(itertools.product(learning_rates, batch_sizes, init_scales))
    hp_results = pd.DataFrame(hp_grid, columns=["lr", "batch_size", "init_scale"])

    for i, (lr, bs, scale) in enumerate(hp_grid):
        cv_summary = kfold_cv(X_train, y_train, k=k, lr=lr, batch_size=bs,
                              lam=lam, init_scale=scale)
        hp_results.loc[i, cv_summary.index] = cv_summary
        print(f"  lr={lr}, B={bs}, init_scale={scale}: "
              f"val_ce={cv_summary['mean_val_ce']:.4f} +/- {cv_summary['std_val_ce']:.4f}, "
              f"val_acc={cv_summary['mean_val_acc']:.4f}")

    best = hp_results.loc[hp_results["mean_val_ce"].idxmin()]
    print(f"\nBest config: lr={best['lr']}, B={best['batch_size']}, init_scale={best['init_scale']}")
    print(f"  val_ce={best['mean_val_ce']:.4f}, val_acc={best['mean_val_acc']:.4f}")

    plot_cv_heatmap(hp_results)
    return {"lr": best["lr"], "batch_size": best["batch_size"], "init_scale": best["init_scale"]}


def sweep_regularization(X_train, y_train, X_test, y_test):
    """Task 3: Lambda sweep with K-fold CV, then final test evaluation."""
    print("\nTask 3: Lambda Sweep (Bias-Variance Trade-off)")

    lr, bs = 0.1, 16
    lambdas = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    k = 5

    sweep_results = pd.DataFrame({"lam": lambdas})
    for i, lam in enumerate(lambdas):
        cv_summary = kfold_cv(X_train, y_train, k=k, lr=lr, batch_size=bs, lam=lam)
        sweep_results.loc[i, cv_summary.index] = cv_summary
        print(f"  lam={lam:.0e}: train_ce={cv_summary['mean_train_ce']:.4f}, "
              f"val_ce={cv_summary['mean_val_ce']:.4f}, "
              f"val_acc={cv_summary['mean_val_acc']:.4f}")

    plot_lambda_sweep(sweep_results)
    plot_lambda_sweep_acc(sweep_results)

    best_idx = sweep_results["mean_val_ce"].idxmin()
    best_lam = sweep_results.loc[best_idx, "lam"]
    print(f"\nBest lambda: {best_lam:.0e} (val_ce={sweep_results.loc[best_idx, 'mean_val_ce']:.4f})")

    X_tr_std, X_te_std, _, _ = standardize(X_train, X_test)
    y_tr_arr = y_train.values if hasattr(y_train, 'values') else y_train
    y_te_arr = y_test.values if hasattr(y_test, 'values') else y_test
    model = LogisticRegressionSGD(
        n_features=X_tr_std.shape[1], lr=lr, lam=best_lam, batch_size=bs
    )
    rng = np.random.default_rng(RANDOM_SEED)
    model.fit(X_tr_std, y_tr_arr, rng=rng)

    test_ce = model.compute_loss(X_te_std, y_te_arr)
    test_acc = np.mean(model.predict(X_te_std) == y_te_arr)
    print(f"Final test results (lam={best_lam:.0e}): CE={test_ce:.4f}, Acc={test_acc:.4f}")


# def fit_l1_regularization_path(X_train, y_train, X_test, y_test):
#     """Task 4: L1 regularization path using sklearn."""
#     print("\nTask 4: L1 Regularization Path")
#     X_tr_std, X_te_std, _, _ = standardize(X_train, X_test)
#     y_tr_arr = y_train.values if hasattr(y_train, 'values') else y_train
#     y_te_arr = y_test.values if hasattr(y_test, 'values') else y_test

#     Cs = np.logspace(-4, 4, 30)
#     coef_matrix = np.zeros((len(Cs), X_tr_std.shape[1]))
#     nnz_counts = np.zeros(len(Cs), dtype=int)

#     prev_model = None
#     for i, C in enumerate(Cs):
#         model = LogisticRegression(
#             C=C, solver="saga", l1_ratio=1, max_iter=10000, tol=1e-4,
#             warm_start=True, random_state=RANDOM_SEED
#         )
#         if prev_model is not None:
#             model.coef_ = prev_model.coef_.copy()
#             model.intercept_ = prev_model.intercept_.copy()
#         model.fit(X_tr_std, y_tr_arr)
#         prev_model = model

#         coef_matrix[i] = model.coef_[0]
#         nnz_counts[i] = np.sum(np.abs(coef_matrix[i]) > 0)
#         print(f"  C={C:.4e}: nnz={nnz_counts[i]}, test_acc={model.score(X_te_std, y_te_arr):.4f}")

#     plot_l1_coef_path(Cs, coef_matrix, FEATURE_NAMES)
#     plot_l1_sparsity(Cs, nnz_counts)

#     print("  Running LogisticRegressionCV for CV performance plot...")
#     cv_model = LogisticRegressionCV(
#         solver="saga", l1_ratios=(1,), Cs=Cs, cv=5,
#         max_iter=10000, random_state=RANDOM_SEED, scoring="accuracy"
#     )
#     cv_model.fit(X_tr_std, y_tr_arr)
#     mean_scores = cv_model.scores_[1].mean(axis=0)

#     plot_l1_cv_performance(Cs, mean_scores)

#     best_C = cv_model.C_[0]
#     print(f"  Best C (CV): {best_C:.4e} (lambda={1/best_C:.4e})")



# Proposed changes to this method 
def fit_l1_regularization_path(X_train, y_train, X_test, y_test):
# Task 4: L1 regularization path using sklearn (no test leakage).
print("\nTask 4: L1 Regularization Path")

# Standardize once using TRAIN stats, then apply to TEST
X_tr_std, X_te_std, _, _ = standardize(X_train, X_test)
y_tr_arr = y_train.values if hasattr(y_train, "values") else y_train
y_te_arr = y_test.values if hasattr(y_test, "values") else y_test

Cs = np.logspace(-4, 4, 30)
coef_matrix = np.zeros((len(Cs), X_tr_std.shape[1]))
nnz_counts = np.zeros(len(Cs), dtype=int)

# Proposed change 1: Use true L1 logistic regression: penalty="l1"
# Proposed change 2: Do not evaluate on the test set during the path (cuz it prevents test leakage)
for i, C in enumerate(Cs):
    model = LogisticRegression(
        penalty="l1",
        C=C,
        solver="saga",
        max_iter=10000,
        tol=1e-4,
        warm_start=True,
        random_state=RANDOM_SEED
    )
    model.fit(X_tr_std, y_tr_arr)

    coef_matrix[i] = model.coef_[0]
    nnz_counts[i] = int(np.sum(np.abs(coef_matrix[i]) > 0))

    #  only report training structure statistics during the path
    print(f"  C={C:.4e}: nnz={nnz_counts[i]}")

# Plots for the regularization path and sparsity
plot_l1_coef_path(Cs, coef_matrix, FEATURE_NAMES)
plot_l1_sparsity(Cs, nnz_counts)

# Proposed change 3: Use CV (on TRAIN ONLY) to select best C
print("  Running LogisticRegressionCV for CV performance plot...")
cv_model = LogisticRegressionCV(
    penalty="l1",
    solver="saga",
    Cs=Cs,
    cv=5,
    max_iter=10000,
    random_state=RANDOM_SEED,
    scoring="accuracy"
)
cv_model.fit(X_tr_std, y_tr_arr)

# For binary classification, scores_ is keyed by class label (0/1)
# Use class 1 if present; otherwise fall back to the last key.
key = 1 if 1 in cv_model.scores_ else list(cv_model.scores_.keys())[-1]
mean_scores = cv_model.scores_[key].mean(axis=0)

plot_l1_cv_performance(Cs, mean_scores)

best_C = float(cv_model.C_[0])
print(f"  Best C (CV): {best_C:.4e} (lambda={1/best_C:.4e})")

# Proposed change 4: Evaluate on TEST only once, at the CV-selected C
final_model = LogisticRegression(
    penalty="l1",
    C=best_C,
    solver="saga",
    max_iter=10000,
    tol=1e-4,
    random_state=RANDOM_SEED
)
final_model.fit(X_tr_std, y_tr_arr)
test_acc = final_model.score(X_te_std, y_te_arr)
print(f"  One-time L1 test accuracy at CV-selected C: {test_acc:.4f}")


if __name__ == "__main__":
    X, y = load_data(cwd / "data")
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
