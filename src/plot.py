import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
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
            ax.set_title(rf"Batch Size={bs}  [$\lambda={lam}$]")
            if col == 0:
                ax.set_ylabel("Training Cross-Entropy")

    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[0][-1].legend(handles[::-1], labels[::-1], fontsize=8, loc="upper right")
    fig.suptitle("Training Curves")
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_cv_heatmap(cv_results, save_path=FIG / "task2" / "cv_hyperparameter_search.png"):
    """3D scatter of mean val CE over the lr x batch_size x init_scale grid."""
    results = cv_results if isinstance(cv_results, pd.DataFrame) else pd.DataFrame(cv_results)

    init_scale_vals = sorted(results["init_scale"].unique())
    scale_to_idx = {s: i for i, s in enumerate(init_scale_vals)}

    x = np.log10(results["lr"].values.astype(float))
    y = np.log10(results["batch_size"].values.astype(float))
    z = results["init_scale"].map(scale_to_idx).values.astype(float)
    c = np.log10(results["mean_val_ce"].values.astype(float))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=c, cmap="viridis", s=60, alpha=0.9, edgecolors="none")

    lr_vals = sorted(results["lr"].unique())
    bs_vals = sorted(results["batch_size"].unique())

    ax.set_xticks(np.log10(lr_vals))
    ax.set_xticklabels([str(v) for v in lr_vals], fontsize=7)
    ax.set_yticks(np.log10(bs_vals))
    ax.set_yticklabels([str(int(v)) for v in bs_vals], fontsize=7)
    ax.set_zticks(list(range(len(init_scale_vals))))
    ax.set_zticklabels([str(s) for s in init_scale_vals], fontsize=7)

    ax.set_xlabel("Learning Rate", labelpad=10)
    ax.set_ylabel("Batch Size", labelpad=10)
    ax.set_zlabel("Init Scale", labelpad=10)
    ax.set_title("Mean Validation CE (K-Fold CV)")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label(r"$\log_{10}$ (Mean Val CE)")

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


def plot_l1_cv_performance(Cs, mean_scores, std_scores=None, save_path=FIG / "task4" / "cv_performance.png"):
    """Mean CV accuracy vs C with optional shaded std region."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Cs, mean_scores, marker="o", markersize=4, label="Mean CV Acc")
    if std_scores is not None:
        ax.fill_between(Cs, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"C ($\frac{1}{\lambda}$)")
    ax.set_ylabel("CV Accuracy")
    ax.set_title("L1 Logistic Regression: CV Performance vs C")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path)
