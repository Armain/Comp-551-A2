# COMP 551 Assignment 2: Training and Evaluation Practices

Logistic regression with mini-batch SGD and Adam on the UCI Spambase dataset. Covers from-scratch SGD/Adam implementation, randomized hyperparameter search via K-fold cross-validation, bias-variance trade-off analysis, and L1 regularization paths.

## Project Structure

```
├── src/
│   ├── main.py            # Orchestrator: runs all 4 tasks sequentially
│   ├── preprocessing.py   # Data loading, train/test split, standardization, K-fold
│   ├── models.py          # LogisticRegressionSGD with SGD and Adam (from-scratch)
│   ├── evaluation.py      # Cross-entropy, accuracy, K-fold CV runner
│   ├── tuner.py           # HP tuning classes (RandomizedGridSearchCV, OptunaTunerCV)
│   ├── plot.py            # All figure generation functions
│   └── utils.py           # Config loading and validation (pydantic)
├── data/
│   ├── spambase.data      # UCI Spambase dataset (4601 samples, 57 features)
│   ├── spambase.names     # Feature descriptions
│   └── spambase.DOCUMENTATION
├── figures/               # Generated plots (created on first run)
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   └── task4/
├── config.ini             # Runtime settings (random seed, plot display)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Setup

### Option 1: uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it first if not already available:

```bash
pip install uv
```

Then install dependencies and create the virtual environment:

```bash
uv sync
```

### Option 2: venv + pip

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Option 3: conda

```bash
conda create -n comp551a2 python=3.12
conda activate comp551a2
pip install -r requirements.txt
```
## Configuration

Runtime settings are stored in `config.ini` at the project root and validated via pydantic in `src/utils.py`:

```ini
[settings]
random_seed = 2026
show_inline_plots = true
verbose = true
tuning_method = random
```

| Setting | Type | Description |
|---------|------|-------------|
| `random_seed` | int | Seed for all random number generators (reproducibility) |
| `show_inline_plots` | bool | Whether to display plots interactively (set to `false` for headless runs) |
| `verbose` | bool | Whether to print per-task result DataFrames after each loop (set to `false` for cleaner output) |
| `tuning_method` | str | Task 2 HP tuning strategy: `"random"` (randomized grid search) or `"optuna"` (TPE) |

## Running

Run from the **project root directory** (not from inside `src/`):

```bash
# With uv:
uv run python src/main.py

# With activated venv or conda:
python src/main.py
```

## Expected Output

All 4 tasks run sequentially with progress printed to stdout:

1. **Task 1** (~1 min): Trains 30 model configurations (3 batch sizes × 5 learning rates × 2 regularization settings) for 200 epochs, once with vanilla SGD and once with Adam. Saves 2 training curve plots.
2. **Task 2** (~5-6 min): Runs 10-fold CV over 100 configurations (50 SGD + 50 Adam); tunes lr, batch size, init scale, lambda, and Adam beta parameters. Sampling strategy depends on `tuning_method`: randomized grid search or Optuna TPE. Reports test CE and accuracy for the best configuration. Saves an interactive two-panel parallel coordinates HTML.
3. **Task 3** (~2 min): Runs 20-fold CV over 8 lambda values at two training set sizes (5% and 20%). Prints best lambda and final test CE/accuracy for each. Saves 2 plots.
4. **Task 4** (~1 min): Fits L1 regularization path over 30 C values using sklearn with 20-fold CV. Reports best C and test accuracy. Saves 3 plots.

Total runtime: approximately 9-11 minutes.

## Output Files

All figures are saved to `figures/` (created automatically):

| File | Description |
|------|-------------|
| `task1/training_curves_sgd.png` | Training CE vs epoch, 2x3 grid, vanilla SGD |
| `task1/training_curves_adam.png` | Training CE vs epoch, 2x3 grid, Adam optimizer |
| `task2/cv_hp_search_random.html` | Interactive two-panel parallel coordinates: SGD (top) and Adam (bottom); produced when `tuning_method=random` |
| `task2/cv_hp_search_optuna.html` | Same format; produced when `tuning_method=optuna` |
| `task3/lambda_sweep_ce.png` | Train vs val CE as function of lambda |
| `task3/lambda_sweep_acc.png` | Train vs val accuracy as function of lambda |
| `task4/coef_path.png` | Top-10 L1 coefficient values vs C |
| `task4/sparsity.png` | Number of non-zero coefficients vs C |
| `task4/cv_performance.png` | Mean CV accuracy vs C |

## Implementation Notes

- **Random seed**: `np.random.default_rng(2026)` used throughout for reproducibility
- **Data split**: 5% train (~230 samples), 95% test (Tasks 1, 2, 4); Task 3 additionally runs a 20% split (~920 samples) for dataset size comparison
- **L2 convention**: `lambda * ||w||_2^2` with gradient term `2*lambda*w` (bias excluded from regularization)
- **Adam optimizer**: Bias-corrected moment estimates (`m_hat`, `v_hat`); `beta1` and `beta2` are tunable hyperparameters in Task 2
- **K-fold CV**: Standardization is fit per fold on the fold's training portion only (prevents data leakage); each fold uses a fresh model with the same seed
- **Task 2 HP tuning**: 100 trials total (50 SGD + 50 Adam) with 10-fold CV each. `tuning_method=random` uses numpy uniform/log-uniform sampling; `tuning_method=optuna` uses TPE (Tree-structured Parzen Estimator) with `n_startup_trials = 0.25 * N_TRIALS_EACH` pure random exploration before model-based search, and `multivariate=True` to model parameter correlations. Both strategies produce identical output schemas and the HTML filename reflects the method used.
- **Task 3**: Lambda sweep uses fixed lr=0.1, batch_size=16, 200 epochs with early stopping (patience=10). Run at two training set sizes to illustrate bias-variance trade-off across data regimes.
- **Task 4**: sklearn `LogisticRegression` with `penalty="l1"`, `solver="saga"`, `warm_start=True` (faster path fitting). C values span `logspace(-4, 4, 30)`. CV uses per-fold standardization consistent with Tasks 2-3.

## Troubleshooting

**`FileNotFoundError: data/spambase.data`**
Run the script from the project root, not from `src/`:
```bash
# Wrong:
cd src && python main.py

# Correct:
python src/main.py
```

**`ModuleNotFoundError`**
Make sure the virtual environment is activated and `uv sync` / `pip install -r requirements.txt` has been run.

**`uv: command not found`**
Install uv first: `pip install uv`

**Convergence warnings from sklearn**
The `max_iter=10000` in Task 4 is sufficient for most C values. Warnings can be ignored safely.
