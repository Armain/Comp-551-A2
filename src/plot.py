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
    title: str = "Training Curves",
    figname: str = "training_curves.png",
) -> None:
    """Plot training loss curves in a rows-x-cols grid (lambda x batch size).

    Args:
        training_curves: DataFrame with columns [lam, batch_size, lr, train_loss]
            where train_loss holds a list of per-epoch loss values.
        title: Title for the overall figure.
        figname: Output filename saved under figures/task1/.
    """
    batch_sizes = sorted(training_curves["batch_size"].unique())
    lam_vals = sorted(training_curves["lam"].unique())

    fig, axes = plt.subplots(len(lam_vals), len(batch_sizes), figsize=(3.5 * len(batch_sizes), 8), sharey='row')

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
            ax.set_ylim(1e-5, 1) if row == 0 else ax.set_ylim(1e-2, 10)
            if col == 0:
                ax.set_ylabel("Training Cross-Entropy")
            ax.legend(fontsize=8, loc='lower right')
    fig.suptitle(title)
    fig.tight_layout()
    save_fig(fig, FIG / "task1" / figname)


def plot_hp_grid_par_coords(
    hp_results: pd.DataFrame,
    figname: str = "cv_hp_search.html",
) -> None:
    """Two-panel interactive parallel coordinates plot of randomized CV search (saved as HTML).

    SGD results are shown in the top panel; Adam results (with Beta1/Beta2 dimensions)
    in the bottom panel. Both panels share the same color scale (log Val CE).

    Args:
        hp_results: Combined DataFrame with a boolean 'use_adam' column and columns
            [lr, batch_size, init_scale, lam, mean_val_ce, ...], plus [beta1, beta2]
            for Adam rows.
        figname: Output filename saved under figures/task2/.
    """
    sgd_res  = hp_results[~hp_results["use_adam"]]
    adam_res = hp_results[hp_results["use_adam"]]

    lr_ref_ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    lam_ref_ticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    beta1_ref_ticks = [0.85, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 0.999]
    beta2_ref_ticks = [0.99, 0.992, 0.994, 0.996, 0.998, 0.9999]
    bs_vals_sorted = sorted(hp_results["batch_size"].unique())
    init_scale_vals = sorted(hp_results["init_scale"].unique())
    scale_to_idx = {s: i for i, s in enumerate(init_scale_vals)}

    def _build_dims(cv_results: pd.DataFrame) -> tuple[list, np.ndarray]:
        lr_log  = np.log10(cv_results["lr"].values.astype(float))
        bs_log  = np.log10(cv_results["batch_size"].values.astype(float))
        is_ord  = cv_results["init_scale"].map(scale_to_idx).values.astype(float)
        lam_log = np.log10(cv_results["lam"].values.astype(float))
        ce_log  = np.log10(cv_results["mean_val_ce"].values.astype(float))
        dims = [
            dict(label="Learning Rate", values=lr_log,
                 tickvals=[np.log10(v) for v in lr_ref_ticks],
                 ticktext=[str(v) for v in lr_ref_ticks]),
            dict(label="Batch Size", values=bs_log,
                 tickvals=[np.log10(float(v)) for v in bs_vals_sorted],
                 ticktext=[str(int(v)) for v in bs_vals_sorted]),
            dict(label="Init Scale", values=is_ord,
                 tickvals=list(range(len(init_scale_vals))),
                 ticktext=[str(s) for s in init_scale_vals]),
            dict(label="Lambda", values=lam_log,
                 tickvals=[np.log10(v) for v in lam_ref_ticks],
                 ticktext=[str(v) for v in lam_ref_ticks]),
        ]
        if "beta1" in cv_results.columns and cv_results["beta1"].notna().any():
            dims.append(dict(
                label="Beta1", values=cv_results["beta1"].values.astype(float),
                tickvals=beta1_ref_ticks
                ))
        if "beta2" in cv_results.columns and cv_results["beta2"].notna().any():
            dims.append(dict(
                label="Beta2", values=cv_results["beta2"].values.astype(float),
                tickvals=beta2_ref_ticks
                ))
        dims.append(dict(label="log Val CE", values=ce_log))
        return dims, ce_log

    sgd_dims,  sgd_ce  = _build_dims(sgd_res)
    adam_dims, adam_ce = _build_dims(adam_res)

    def _parcoords_fig(dims: list, ce: np.ndarray, title: str) -> go.Figure:
        f = go.Figure(data=go.Parcoords(
            line=dict(color=ce, colorscale="agsunset", showscale=True,
                      cmin=-0.65, cmax=0.55,
                      colorbar=dict(title="log Val CE")),
            dimensions=dims,
        ))
        f.update_layout(height=450, title=dict(text=title, x=0.5, xanchor="center"))
        return f

    sgd_fig  = _parcoords_fig(sgd_dims,  sgd_ce,  "SGD Hyperparameter Search")
    adam_fig = _parcoords_fig(adam_dims, adam_ce, "Adam Hyperparameter Search")

    html_sgd  = sgd_fig.to_html(full_html=False, include_plotlyjs="cdn")
    html_adam = adam_fig.to_html(full_html=False, include_plotlyjs=False)

    out = FIG / "task2" / figname
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f'<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>'
        f'{html_sgd}{html_adam}</body></html>'
    )


def plot_lambda_sweep(
    sweep_results: pd.DataFrame,
    sweep_results2: pd.DataFrame | None = None,
    label1: str | None = None,
    label2: str | None = None,
    figname: str = "lambda_sweep_ce.png",
) -> None:
    """Plot train vs val CE as a function of lambda with 80% uncertainty bands.

    Args:
        sweep_results: Primary sweep DataFrame with p10_/p90_ prefixed columns.
        sweep_results2: Optional second sweep DataFrame (e.g. larger training set).
        label1: Legend suffix for sweep_results.
        label2: Legend suffix for sweep_results2.
        figname: Output filename saved under figures/task3/.
    """
    suffix1 = f' ({label1})' if label1 else ''
    suffix2 = f' ({label2})' if label1 else ''
    
    datasets = [(sweep_results, suffix1, "-")]
    if sweep_results2 is not None:
        datasets.append((sweep_results2, suffix2, "--"))

    fig, ax = plt.subplots(figsize=(8, 6))
    for results, suffix, ls in datasets:
        x = results["lam"].values.astype(float)
        for metric, marker, base_label in [("train_ce", "o", "Training Cross-Entropy"), ("val_ce", "s", "Validation Cross-Entropy")]:
            mean = results[f"mean_{metric}"].values
            p10  = results[f"p10_{metric}"].values
            p90  = results[f"p90_{metric}"].values
            line, = ax.plot(x, mean, marker=marker, linestyle=ls, label=f"{base_label}{suffix}")
            ax.fill_between(x, p10, p90, alpha=0.15, color=line.get_color())
    ax.set_xscale("symlog", linthresh=1e-7)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Cross-Entropy")
    ax.set_title(r"Bias-Variance Trade-off: Cross-Entropy vs $\lambda$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, FIG / "task3" / figname)


def plot_lambda_sweep_acc(
    sweep_results: pd.DataFrame,
    sweep_results2: pd.DataFrame | None = None,
    label1: str | None = None,
    label2: str | None = None,
    figname: str = "lambda_sweep_acc.png",
) -> None:
    """Plot train vs val accuracy as a function of lambda with 80% uncertainty bands.

    Args:
        sweep_results: Primary sweep DataFrame with p10_/p90_ prefixed columns.
        sweep_results2: Optional second sweep DataFrame (e.g. larger training set).
        label1: Legend suffix for sweep_results.
        label2: Legend suffix for sweep_results2.
        figname: Output filename saved under figures/task3/.
    """
    suffix1 = f' ({label1})' if label1 else ''
    suffix2 = f' ({label2})' if label1 else ''
    
    datasets = [(sweep_results, suffix1, "-")]
    if sweep_results2 is not None:
        datasets.append((sweep_results2, suffix2, "--"))

    fig, ax = plt.subplots(figsize=(8, 6))
    for results, suffix, ls in datasets:
        x = results["lam"].values.astype(float)
        for metric, marker, base_label in [("train_acc", "o", "Training Accuracy"), ("val_acc", "s", "Validation Accuracy")]:
            mean = results[f"mean_{metric}"].values
            p10  = results[f"p10_{metric}"].values
            p90  = results[f"p90_{metric}"].values
            line, = ax.plot(x, mean, marker=marker, linestyle=ls, label=f"{base_label}{suffix}")
            ax.fill_between(x, p10, p90, alpha=0.15, color=line.get_color())
    ax.set_xscale("symlog", linthresh=1e-7)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Accuracy")
    ax.set_title(r"Bias-Variance Trade-off: Accuracy vs $\lambda$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, FIG / "task3" / figname)


def plot_l1_coef_path(
    Cs: np.ndarray,
    coef_matrix: np.ndarray,
    feature_names: list[str],
    figname: str = "coef_path.png",
    top_k: int = 10,
) -> None:
    """Plot coefficient values vs C for the top-k features by max absolute value.

    Args:
        Cs: Array of C values (inverse regularization strength).
        coef_matrix: Coefficient matrix of shape (len(Cs), n_features).
        feature_names: Feature name strings in dataset column order.
        figname: Output filename saved under figures/task4/.
        top_k: Number of top features to display.
    """
    max_abs = np.max(np.abs(coef_matrix), axis=0)
    top_idx = np.argsort(max_abs)[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in top_idx:
        ax.plot(Cs, coef_matrix[:, i], label=feature_names[i])
    ax.set_xscale("log")
    ax.set_xlabel(r"C ($\frac{1}{\lambda}$)")
    ax.set_ylabel("Coefficient value")
    ax.set_title(f"L1 Regularization Path (top {top_k} features)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, FIG / "task4" / figname)


def plot_l1_sparsity(
    Cs: np.ndarray,
    nnz_counts: np.ndarray,
    figname: str = "sparsity.png",
) -> None:
    """Plot number of non-zero coefficients vs C.

    Args:
        Cs: Array of C values (inverse regularization strength).
        nnz_counts: Number of non-zero coefficients at each C.
        figname: Output filename saved under figures/task4/.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Cs, nnz_counts, marker="o", markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel(r"C ($\frac{1}{\lambda}$)")
    ax.set_ylabel("Number of non-zero coefficients")
    ax.set_title(r"Sparsity: $\|w\|_0$ vs C")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, FIG / "task4" / figname)


def plot_l1_cv_performance(
    Cs: np.ndarray,
    mean_scores: np.ndarray,
    p10_scores: np.ndarray | None = None,
    p90_scores: np.ndarray | None = None,
    figname: str = "cv_performance.png",
) -> None:
    """Plot mean CV accuracy vs C with optional shaded 80% uncertainty bound.

    Args:
        Cs: Array of C values (inverse regularization strength).
        mean_scores: Mean CV accuracy at each C.
        p10_scores: Optional 10th percentile of CV accuracy for shading.
        p90_scores: Optional 90th percentile of CV accuracy for shading.
        figname: Output filename saved under figures/task4/.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Cs, mean_scores, marker="o", markersize=4, label="Mean CV Accuracy")
    if p10_scores is not None and p90_scores is not None:
        ax.fill_between(Cs, p10_scores, p90_scores, alpha=0.2, label="80% Uncertainty Bound")
    ax.set_xscale("log")
    ax.set_xlabel(r"C ($\frac{1}{\lambda}$)")
    ax.set_ylabel("CV Accuracy")
    ax.set_title("L1 Logistic Regression: CV Performance vs C")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, FIG / "task4" / figname)
