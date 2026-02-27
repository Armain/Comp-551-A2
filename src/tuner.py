import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

from evaluation import kfold_cv
from preprocessing import RANDOM_SEED
from utils import config

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS_EACH = 100
K_FOLDS = 10
BATCH_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128]
SCALE_OPTIONS = [0.0, 0.001, 0.01, 0.1, 1.0]
LR_RANGE = (1e-4, 1.0) # logscale
LAM_RANGE = (1e-6, 1.0) # logscale
BETA1_RANGE = (0.85, 0.999) # linscale
BETA2_RANGE = (0.99, 0.9999) # linscale


class RandomizedGridSearchCV:
    """Randomized hyperparameter search over SGD and Adam configs with K-fold CV."""

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.X_train = X_train
        self.y_train = y_train

    def run(self) -> tuple[pd.DataFrame, pd.Series]:
        print("\nTask 2: Hyperparameter Tuning (Randomized Search CV)")

        rng_hp = np.random.default_rng(RANDOM_SEED)

        sgd_results = pd.DataFrame({
            "lr":          10 ** rng_hp.uniform(np.log10(LR_RANGE[0]), np.log10(LR_RANGE[1]), N_TRIALS_EACH),
            "lam":         10 ** rng_hp.uniform(np.log10(LAM_RANGE[0]), np.log10(LAM_RANGE[1]), N_TRIALS_EACH),
            "batch_size":  rng_hp.choice(BATCH_OPTIONS, N_TRIALS_EACH).astype(int),
            "init_scale":  rng_hp.choice(SCALE_OPTIONS, N_TRIALS_EACH),
            "use_adam":    False,
            "mean_val_ce": np.nan, "std_val_ce": np.nan, "mean_val_acc": np.nan,
        })

        adam_results = pd.DataFrame({
            "lr":          10 ** rng_hp.uniform(np.log10(LR_RANGE[0]), np.log10(LR_RANGE[1]), N_TRIALS_EACH),
            "lam":         10 ** rng_hp.uniform(np.log10(LAM_RANGE[0]), np.log10(LAM_RANGE[1]), N_TRIALS_EACH),
            "batch_size":  rng_hp.choice(BATCH_OPTIONS, N_TRIALS_EACH).astype(int),
            "init_scale":  rng_hp.choice(SCALE_OPTIONS, N_TRIALS_EACH),
            "beta1":       rng_hp.uniform(*BETA1_RANGE, N_TRIALS_EACH),
            "beta2":       rng_hp.uniform(*BETA2_RANGE, N_TRIALS_EACH),
            "use_adam":    True,
            "mean_val_ce": np.nan, "std_val_ce": np.nan, "mean_val_acc": np.nan,
        })

        for i, row in tqdm(sgd_results.iterrows(), total=N_TRIALS_EACH, desc="SGD Search"):
            cv_summary = kfold_cv(self.X_train, self.y_train, k=K_FOLDS, lr=row["lr"],
                                  batch_size=int(row["batch_size"]), lam=row["lam"],
                                  init_scale=row["init_scale"])
            sgd_results.loc[i, cv_summary.index] = cv_summary

        for i, row in tqdm(adam_results.iterrows(), total=N_TRIALS_EACH, desc="Adam Search"):
            cv_summary = kfold_cv(self.X_train, self.y_train, k=K_FOLDS, lr=row["lr"],
                                  batch_size=int(row["batch_size"]), lam=row["lam"],
                                  init_scale=row["init_scale"],
                                  use_adam=True, beta1=row["beta1"], beta2=row["beta2"])
            adam_results.loc[i, cv_summary.index] = cv_summary

        hp_results = pd.concat([sgd_results, adam_results], ignore_index=True)
        stat_cols = [c for c in hp_results.columns if c.startswith(("mean_", "std_", "p10_", "p90_"))]
        config_cols = [c for c in hp_results.columns if c not in stat_cols]
        hp_results = hp_results[config_cols + stat_cols].sort_values("mean_val_ce").reset_index(drop=True)

        if config.verbose:
            print_cols = ["lr", "batch_size", "init_scale", "lam", "use_adam", "beta1", "beta2",
                          "mean_val_ce", "std_val_ce", "mean_val_acc", "std_val_acc"]
            print(hp_results[print_cols].head(20).to_string(index=False))

        best = hp_results.iloc[0]
        best_params = best[config_cols]
        print(f"\nBest config:\n{best_params.to_string()}")
        return hp_results, best_params


class OptunaTunerCV:
    """Optuna TPE-based hyperparameter search over SGD and Adam configs with K-fold CV."""

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.X_train = X_train
        self.y_train = y_train

    def _objective_sgd(self, trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", *LR_RANGE, log=True)
        lam = trial.suggest_float("lam", *LAM_RANGE, log=True)
        batch_size = trial.suggest_categorical("batch_size", BATCH_OPTIONS)
        init_scale = trial.suggest_categorical("init_scale", SCALE_OPTIONS)

        cv_summary = kfold_cv(self.X_train, self.y_train, k=K_FOLDS, lr=lr,
                              batch_size=int(batch_size), lam=lam, init_scale=init_scale)
        for key, val in cv_summary.items():
            trial.set_user_attr(key, float(val))
        return float(cv_summary["mean_val_ce"])

    def _objective_adam(self, trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", *LR_RANGE, log=True)
        lam = trial.suggest_float("lam", *LAM_RANGE, log=True)
        batch_size = trial.suggest_categorical("batch_size", BATCH_OPTIONS)
        init_scale = trial.suggest_categorical("init_scale", SCALE_OPTIONS)
        beta1 = trial.suggest_float("beta1", *BETA1_RANGE)
        beta2 = trial.suggest_float("beta2", *BETA2_RANGE)

        cv_summary = kfold_cv(self.X_train, self.y_train, k=K_FOLDS, lr=lr,
                              batch_size=int(batch_size), lam=lam, init_scale=init_scale,
                              use_adam=True, beta1=beta1, beta2=beta2)
        for key, val in cv_summary.items():
            trial.set_user_attr(key, float(val))
        return float(cv_summary["mean_val_ce"])

    @staticmethod
    def _study_to_df(study: optuna.Study, use_adam: bool) -> pd.DataFrame:
        rows = []
        for trial in study.trials:
            row = {**trial.params, **trial.user_attrs, "use_adam": use_adam}
            rows.append(row)
        return pd.DataFrame(rows)

    def run(self) -> tuple[pd.DataFrame, pd.Series]:
        print("\nTask 2: Hyperparameter Tuning (Optuna TPE)")

        n_startup = int(0.25 * N_TRIALS_EACH)
        sgd_sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED, n_startup_trials=n_startup, multivariate=True)
        sgd_study = optuna.create_study(direction="minimize", sampler=sgd_sampler)
        with tqdm(total=N_TRIALS_EACH, desc="SGD Search") as pbar:
            sgd_study.optimize(self._objective_sgd, n_trials=N_TRIALS_EACH,
                               callbacks=[lambda study, trial: pbar.update(1)])

        adam_sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED, n_startup_trials=n_startup, multivariate=True)
        adam_study = optuna.create_study(direction="minimize", sampler=adam_sampler)
        with tqdm(total=N_TRIALS_EACH, desc="Adam Search") as pbar:
            adam_study.optimize(self._objective_adam, n_trials=N_TRIALS_EACH,
                                callbacks=[lambda study, trial: pbar.update(1)])

        sgd_results = self._study_to_df(sgd_study, use_adam=False)
        adam_results = self._study_to_df(adam_study, use_adam=True)
        
        hp_results = pd.concat([sgd_results, adam_results], ignore_index=True)
        stat_cols = [c for c in hp_results.columns if c.startswith(("mean_", "std_", "p10_", "p90_"))]
        config_cols = [c for c in hp_results.columns if c not in stat_cols]
        hp_results = hp_results[config_cols + stat_cols].sort_values("mean_val_ce").reset_index(drop=True)

        if config.verbose:
            print_cols = ["lr", "batch_size", "init_scale", "lam", "use_adam", "beta1", "beta2",
                          "mean_val_ce", "std_val_ce", "mean_val_acc", "std_val_acc"]
            print(hp_results[print_cols].head(20).to_string(index=False))

        best = hp_results.iloc[0]
        best_params = best[config_cols]
        print(f"\nBest config:\n{best_params.to_string()}")
        return hp_results, best_params
