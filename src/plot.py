import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from utils import config

FIG = Path(__file__).parent.parent / "figures"
SHOW_INLINE_PLOTS = config.show_inline_plots
sns.set_theme(style='whitegrid')

def save_fig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if SHOW_INLINE_PLOTS:
        plt.show()
    plt.close(fig)


def plot_training_curves(training_curves, save_path=FIG / "task1" / "training_curves.png"):
    """Combined 2x3 figure: rows = regularization setting, cols = batch size.

    training_curves: DataFrame with columns [lam, batch_size, lr, train_loss]
    where train_loss holds a list of per-epoch loss values.
    """
    batch_sizes = sorted(training_curves["batch_size"].unique())
    lam_vals = sorted(training_curves["lam"].unique())
    row_labels = {0: "No regularization", 1e-3: r"$\lambda=10^{-3}$"}

    fig, axes = plt.subplots(len(lam_vals), len(batch_sizes), figsize=(5 * len(batch_sizes), 7), sharey=True)

    for row, lam in enumerate(lam_vals):
        for col, bs in enumerate(batch_sizes):
            ax = axes[row][col]
            subset = training_curves[(training_curves["lam"] == lam) & (training_curves["batch_size"] == bs)]
            for _, cfg in subset.sort_values("lr").iterrows():
                ax.plot(cfg["train_loss"], label=f"lr={cfg['lr']}")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Epoch")
            ax.set_title(f"Batch Size={bs}  [{row_labels[lam]}]")
            if col == 0:
                ax.set_ylabel("Training Cross-Entropy")

    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[0][-1].legend(handles[::-1], labels[::-1], fontsize=8, loc="upper right")
    fig.suptitle("Training Curves")
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_cv_heatmap(cv_results, save_path=FIG / "task2" / "cv_results_table.png"):
    """Seaborn heatmap of mean val CE, one subplot per init_scale value."""
    results = cv_results if isinstance(cv_results, pd.DataFrame) else pd.DataFrame(cv_results)
    init_scale_vals = sorted(results["init_scale"].unique())

    fig, axes = plt.subplots(1, len(init_scale_vals), figsize=(4.5 * len(init_scale_vals), 4), sharey=True)
    if len(init_scale_vals) == 1:
        axes = [axes]

    for ax, scale in zip(axes, init_scale_vals):
        subset = results[results["init_scale"] == scale]
        pivot = subset.pivot(index="lr", columns="batch_size", values="mean_val_ce")
        pivot = pivot.sort_index(ascending=False)
        sns.heatmap(
            pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".3f",
            annot_kws={"size": 8}, cbar=False,
            yticklabels=[f"{v:.0e}" if v < 0.01 else str(v) for v in pivot.index],
        )
        ax.set_title(f"init_scale = {scale}")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Learning Rate" if ax is axes[0] else "")

    fig.suptitle("Mean Validation Cross-Entropy (K-Fold CV)")
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_lambda_sweep(sweep_results, save_path=FIG / "task3" / "lambda_sweep_ce.png"):
    """Train vs val CE as function of lambda with shaded std regions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = sweep_results["lam"].values.astype(float)
    for metric, marker, label in [("train_ce", "o", "Train CE"), ("val_ce", "s", "Val CE")]:
        mean = sweep_results[f"mean_{metric}"].values
        std = sweep_results[f"std_{metric}"].values
        ax.plot(x, mean, marker=marker, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xscale("symlog", linthresh=1e-7)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Cross-Entropy")
    ax.set_title(r"Bias-Variance Trade-off: CE vs $\lambda$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_lambda_sweep_acc(sweep_results, save_path=FIG / "task3" / "lambda_sweep_acc.png"):
    """Train vs val accuracy as function of lambda with shaded std regions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = sweep_results["lam"].values.astype(float)
    for metric, marker, label in [("train_acc", "o", "Train Acc"), ("val_acc", "s", "Val Acc")]:
        mean = sweep_results[f"mean_{metric}"].values
        std = sweep_results[f"std_{metric}"].values
        ax.plot(x, mean, marker=marker, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xscale("symlog", linthresh=1e-7)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Accuracy")
    ax.set_title(r"Bias-Variance Trade-off: Accuracy vs $\lambda$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_l1_coef_path(Cs, coef_matrix, feature_names, save_path=FIG / "task4" / "coef_path.png", top_k=10):
    """Coefficient values vs C for top-k features by max absolute value."""
    max_abs = np.max(np.abs(coef_matrix), axis=0)
    top_idx = np.argsort(max_abs)[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in top_idx:
        ax.plot(Cs, coef_matrix[:, i], label=feature_names[i])
    ax.set_xscale("log")
    ax.set_xlabel(r"C ($\frac{1}{\lambda}$)")
    ax.set_ylabel("Coefficient value")
    ax.set_title(f"L1 Regularization Path (top {top_k} features)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_l1_sparsity(Cs, nnz_counts, save_path=FIG / "task4" / "sparsity.png"):
    """Number of non-zero coefficients vs C."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Cs, nnz_counts, marker="o", markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel(r"C ($\frac{1}{\lambda}$)")
    ax.set_ylabel("Number of non-zero coefficients")
    ax.set_title(r"Sparsity: $\|w\|_0$ vs C")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_l1_cv_performance(Cs, mean_scores, save_path=FIG / "task4" / "cv_performance.png"):
    """Mean CV accuracy vs C."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Cs, mean_scores, marker="o", markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel(r"C ($\frac{1}{\lambda}$)")
    ax.set_ylabel("Mean CV Accuracy")
    ax.set_title("L1 Logistic Regression: CV Performance vs C")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path)
