import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from pathlib import Path
from utils import config

FIG = Path(__file__).parent.parent / "figures"
SHOW_INLINE_PLOTS = config.show_inline_plots
sns.set_theme(style='whitegrid')


def save_fig(fig: plt.Figure, path: Path | str) -> None:
    """Save figure to disk and optionally display inline.

    Args:
        fig: Matplotlib figure to save.
        path: Destination file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if SHOW_INLINE_PLOTS:
        plt.show()
    plt.close(fig)


def plot_training_curves(
    training_curves: pd.DataFrame,
    save_path: Path = FIG / "task1" / "training_curves.png",
) -> None:
    """Plot training loss curves in a rows-x-cols grid (lambda x batch size).

    Args:
        training_curves: DataFrame with columns [lam, batch_size, lr, train_loss]
            where train_loss holds a list of per-epoch loss values.
        save_path: Destination path for the saved figure.
    """
    batch_sizes = sorted(training_curves["batch_size"].unique())
    lam_vals = sorted(training_curves["lam"].unique())

    fig, axes = plt.subplots(len(lam_vals), len(batch_sizes), figsize=(4 * len(batch_sizes), 8), sharey=True)

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
            ax.legend(fontsize=8, loc='lower right')
    fig.suptitle("Training Curves")
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_hp_grid_par_coords(
    cv_results: pd.DataFrame,
    save_path: Path = FIG / "task2" / "cv_hyperparameter_search.html",
) -> None:
    """Interactive parallel coordinates plot of CV hyperparameter search (saved as HTML).

    Args:
        cv_results: DataFrame with columns [lr, batch_size, init_scale, mean_val_ce, ...].
        save_path: Destination path for the saved HTML file.
    """
    lr_vals_sorted = sorted(cv_results["lr"].unique())
    bs_vals_sorted = sorted(cv_results["batch_size"].unique())
    init_scale_vals = sorted(cv_results["init_scale"].unique())
    scale_to_idx = {s: i for i, s in enumerate(init_scale_vals)}

    lr_log = np.log10(cv_results["lr"].values.astype(float))
    bs_log = np.log10(cv_results["batch_size"].values.astype(float))
    is_ord = cv_results["init_scale"].map(scale_to_idx).values.astype(float)
    ce_log = np.log10(cv_results["mean_val_ce"].values.astype(float))

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=ce_log,
            colorscale="aggrnyl",
            showscale=True,
            colorbar=dict(title="Log Validation Cross Entropy"),
        ),
        dimensions=[
            dict(
                label="Learning Rate",
                values=lr_log,
                tickvals=[np.log10(float(v)) for v in lr_vals_sorted],
                ticktext=[str(v) for v in lr_vals_sorted],
            ),
            dict(
                label="Batch Size",
                values=bs_log,
                tickvals=[np.log10(float(v)) for v in bs_vals_sorted],
                ticktext=[str(int(v)) for v in bs_vals_sorted],
            ),
            dict(
                label="Init Scale",
                values=is_ord,
                tickvals=list(range(len(init_scale_vals))),
                ticktext=[str(s) for s in init_scale_vals],
            ),
            dict(
                label="log Val CE",
                values=ce_log,
            ),
        ],
    ))
    fig.update_layout(
        title="Hyperparameter Search: Parallel Coordinates",
        height=500,
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))


def plot_lambda_sweep(
    sweep_results: pd.DataFrame,
    save_path: Path = FIG / "task3" / "lambda_sweep_ce.png",
) -> None:
    """Plot train vs val CE as a function of lambda with shaded std regions.

    Args:
        sweep_results: DataFrame with columns [lam, mean_train_ce, std_train_ce,
            mean_val_ce, std_val_ce, ...].
        save_path: Destination path for the saved figure.
    """
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
    ax.set_title(r"Bias-Variance Trade-off: Cross Entropy vs $\lambda$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_lambda_sweep_acc(
    sweep_results: pd.DataFrame,
    save_path: Path = FIG / "task3" / "lambda_sweep_acc.png",
) -> None:
    """Plot train vs val accuracy as a function of lambda with shaded std regions.

    Args:
        sweep_results: DataFrame with columns [lam, mean_train_acc, std_train_acc,
            mean_val_acc, std_val_acc, ...].
        save_path: Destination path for the saved figure.
    """
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


def plot_l1_coef_path(
    Cs: np.ndarray,
    coef_matrix: np.ndarray,
    feature_names: list[str],
    save_path: Path = FIG / "task4" / "coef_path.png",
    top_k: int = 10,
) -> None:
    """Plot coefficient values vs C for the top-k features by max absolute value.

    Args:
        Cs: Array of C values (inverse regularization strength).
        coef_matrix: Coefficient matrix of shape (len(Cs), n_features).
        feature_names: Feature name strings in dataset column order.
        save_path: Destination path for the saved figure.
        top_k: Number of top features to display.
    """
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


def plot_l1_sparsity(
    Cs: np.ndarray,
    nnz_counts: np.ndarray,
    save_path: Path = FIG / "task4" / "sparsity.png",
) -> None:
    """Plot number of non-zero coefficients vs C.

    Args:
        Cs: Array of C values (inverse regularization strength).
        nnz_counts: Number of non-zero coefficients at each C.
        save_path: Destination path for the saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Cs, nnz_counts, marker="o", markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel(r"C ($\frac{1}{\lambda}$)")
    ax.set_ylabel("Number of non-zero coefficients")
    ax.set_title(r"Sparsity: $\|w\|_0$ vs C")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_l1_cv_performance(
    Cs: np.ndarray,
    mean_scores: np.ndarray,
    std_scores: np.ndarray | None = None,
    save_path: Path = FIG / "task4" / "cv_performance.png",
) -> None:
    """Plot mean CV accuracy vs C with optional shaded std region.

    Args:
        Cs: Array of C values (inverse regularization strength).
        mean_scores: Mean CV accuracy at each C.
        std_scores: Optional std of CV accuracy for shading.
        save_path: Destination path for the saved figure.
    """
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
