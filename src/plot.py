import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

SHOW_INLINE_PLOTS = False
PALETTE = sns.color_palette("tab10")


def save_fig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if SHOW_INLINE_PLOTS:
        plt.show()
    plt.close(fig)


def plot_training_curves(results_no_reg, results_l2, save_path):
    """Combined 2x3 figure: rows = regularization setting, cols = batch size.

    results_no_reg, results_l2: dict {batch_size: {lr: [loss_per_epoch]}}
    """
    batch_sizes = sorted(results_no_reg.keys())
    learning_rates = sorted(results_no_reg[batch_sizes[0]].keys())
    colors = PALETTE[:len(learning_rates)]

    fig, axes = plt.subplots(2, len(batch_sizes), figsize=(5 * len(batch_sizes), 7), sharey=True)

    row_labels = ["No regularization", r"$\lambda=10^{-3}$"]
    for row, results in enumerate([results_no_reg, results_l2]):
        for col, bs in enumerate(batch_sizes):
            ax = axes[row][col]
            for lr, color in zip(learning_rates, colors):
                ax.plot(results[bs][lr], label=f"lr={lr}", color=color)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Epoch")
            if col == 0:
                ax.set_ylabel("Training Cross-Entropy")
                ax.set_title(f"B={bs}  [{row_labels[row]}]")
            else:
                ax.set_title(f"B={bs}")

    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[0][-1].legend(handles[::-1], labels[::-1], fontsize=8, loc="upper right")
    fig.suptitle("Training Curves")
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_cv_heatmap(cv_results, save_path):
    """Seaborn heatmap of mean val CE, one subplot per epoch value."""
    df = pd.DataFrame(cv_results)
    epoch_vals = sorted(df["epochs"].unique())

    fig, axes = plt.subplots(1, len(epoch_vals), figsize=(4.5 * len(epoch_vals), 4), sharey=True)
    if len(epoch_vals) == 1:
        axes = [axes]

    for ax, ep in zip(axes, epoch_vals):
        subset = df[df["epochs"] == ep]
        pivot = subset.pivot(index="lr", columns="batch_size", values="mean_val_ce")
        pivot = pivot.sort_index(ascending=False)
        sns.heatmap(
            pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".3f",
            annot_kws={"size": 8}, cbar=False,
            yticklabels=[f"{v:.0e}" if v < 0.01 else str(v) for v in pivot.index],
        )
        ax.set_title(f"Epochs = {ep}")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Learning Rate" if ax is axes[0] else "")

    fig.suptitle("Mean Validation Cross-Entropy (K-Fold CV)")
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_lambda_sweep(lambdas, train_ces, val_ces, train_stds, val_stds, save_path):
    """Train vs val CE as function of lambda with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array(lambdas, dtype=float)
    c0, c1 = PALETTE[0], PALETTE[1]
    ax.errorbar(x, train_ces, yerr=train_stds, marker="o", label="Train CE", capsize=3, color=c0)
    ax.errorbar(x, val_ces, yerr=val_stds, marker="s", label="Val CE", capsize=3, color=c1)
    ax.set_xscale("symlog", linthresh=1e-7)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Cross-Entropy")
    ax.set_title(r"Bias-Variance Trade-off: CE vs $\lambda$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_lambda_sweep_acc(lambdas, train_accs, val_accs, train_stds, val_stds, save_path):
    """Train vs val accuracy as function of lambda with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array(lambdas, dtype=float)
    c0, c1 = PALETTE[0], PALETTE[1]
    ax.errorbar(x, train_accs, yerr=train_stds, marker="o", label="Train Acc", capsize=3, color=c0)
    ax.errorbar(x, val_accs, yerr=val_stds, marker="s", label="Val Acc", capsize=3, color=c1)
    ax.set_xscale("symlog", linthresh=1e-7)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Accuracy")
    ax.set_title(r"Bias-Variance Trade-off: Accuracy vs $\lambda$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_l1_coef_path(Cs, coef_matrix, feature_names, save_path, top_k=10):
    """Coefficient values vs C for top-k features by max absolute value."""
    max_abs = np.max(np.abs(coef_matrix), axis=0)
    top_idx = np.argsort(max_abs)[-top_k:][::-1]
    colors = PALETTE[:top_k]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, color in zip(top_idx, colors):
        ax.plot(Cs, coef_matrix[:, i], label=feature_names[i], color=color)
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse regularization strength)")
    ax.set_ylabel("Coefficient value")
    ax.set_title(f"L1 Regularization Path (top {top_k} features)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_l1_sparsity(Cs, nnz_counts, save_path):
    """Number of non-zero coefficients vs C."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Cs, nnz_counts, marker="o", markersize=4, color=PALETTE[0])
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse regularization strength)")
    ax.set_ylabel("Number of non-zero coefficients")
    ax.set_title(r"Sparsity: $\|w\|_0$ vs C")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_l1_cv_performance(Cs, mean_scores, save_path):
    """Mean CV accuracy vs C."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Cs, mean_scores, marker="o", markersize=4, color=PALETTE[0])
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse regularization strength)")
    ax.set_ylabel("Mean CV Accuracy")
    ax.set_title("L1 Logistic Regression: CV Performance vs C")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path)
