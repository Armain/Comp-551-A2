# COMP 551 Assignment 2: Training and Evaluation Practices

Logistic regression with mini-batch SGD on the UCI Spambase dataset. Covers from-scratch SGD implementation, hyperparameter tuning via K-fold cross-validation, bias-variance trade-off analysis, and L1 regularization paths.

## Project Structure

```
├── src/
│   ├── main.py            # Orchestrator: runs all 4 tasks sequentially
│   ├── preprocessing.py   # Data loading, train/test split, standardization, K-fold
│   ├── models.py          # LogisticRegressionSGD (from-scratch implementation)
│   ├── evaluation.py      # Cross-entropy, accuracy, K-fold CV runner
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
```

| Setting | Type | Description |
|---------|------|-------------|
| `random_seed` | int | Seed for all random number generators (reproducibility) |
| `show_inline_plots` | bool | Whether to display plots interactively (set to `false` for headless runs) |
| `verbose` | bool | Whether to print per-task result DataFrames after each loop (set to `false` for cleaner output) |

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

1. **Task 1**: Trains 30 model configurations (3 batch sizes x 5 learning rates x 2 regularization settings) for 200 epochs. Saves 1 combined training curve plot.
2. **Task 2**: Runs 5-fold CV over 125 hyperparameter configurations (5x5x5 grid). Prints best config and saves a heatmap.
3. **Task 3**: Runs K-fold CV over 8 lambda values. Prints best lambda and final test CE/accuracy. Saves 2 plots.
4. **Task 4**: Fits L1 regularization path over 30 C values using sklearn. Saves 3 plots.

Total runtime: approximately 5-10 minutes.

## Output Files

All figures are saved to `figures/` (created automatically):

| File | Description |
|------|-------------|
| `task1/training_curves.png` | Training CE vs epoch, combined 2x3 grid (no reg vs L2) |
| `task2/cv_results_table.png` | Heatmap of mean validation CE over CV grid |
| `task3/lambda_sweep_ce.png` | Train vs val CE as function of lambda |
| `task3/lambda_sweep_acc.png` | Train vs val accuracy as function of lambda |
| `task4/coef_path.png` | Top-10 L1 coefficient values vs C |
| `task4/sparsity.png` | Number of non-zero coefficients vs C |
| `task4/cv_performance.png` | Mean CV accuracy vs C |

## Implementation Notes

- **Random seed**: `np.random.default_rng(2026)` used throughout
- **Data split**: 5% train (~230 samples), 95% test
- **L2 convention**: `lambda * ||w||_2^2` with gradient term `2*lambda*w` (bias excluded)
- **K-fold CV**: Standardization is fit per fold on the fold's training portion only (no leakage)
- **Task 4**: Uses sklearn `LogisticRegression` with `penalty="l1"` and `solver="saga"` (pure L1)

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
