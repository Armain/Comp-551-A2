import numpy as np


class LogisticRegressionSGD:
    """Logistic regression trained with mini-batch SGD and L2 regularization.

    Regularization convention: lambda * ||w||_2^2 (gradient term: 2*lambda*w).
    Bias is excluded from the penalty.
    """

    def __init__(
        self,
        n_features: int,
        lr: float = 0.01,
        lam: float = 1e-3,
        batch_size: int = 16,
        init_scale: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialise weights, bias, and hyperparameters.

        Args:
            n_features: Dimensionality of the input.
            lr: Learning rate.
            lam: L2 regularization strength.
            batch_size: Mini-batch size.
            init_scale: Std of Gaussian weight init; 0.0 uses zero init.
            rng: Optional random generator for weight initialisation.
        """
        if init_scale == 0.0:
            self.w = np.zeros(n_features)
        else:
            rng = rng if rng is not None else np.random.default_rng()
            self.w = rng.normal(0, init_scale, n_features)
        self.b = 0.0
        self.lr = lr
        self.lam = lam
        self.batch_size = batch_size

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid computed in two branches.

        Args:
            z: Input array of pre-activations.

        Returns:
            Element-wise sigmoid values in (0, 1).
        """
        result = np.empty_like(z, dtype=float)
        pos = z >= 0
        result[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        result[~pos] = exp_z / (1.0 + exp_z)
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute P(y=1|X).

        Args:
            X: Feature matrix of shape (n, d).

        Returns:
            Predicted probabilities of shape (n,).
        """
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels by thresholding predicted probabilities.

        Args:
            X: Feature matrix of shape (n, d).
            threshold: Decision boundary.

        Returns:
            Integer predictions of shape (n,).
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    @staticmethod
    def cross_entropy(y: np.ndarray, p: np.ndarray) -> float:
        """Mean binary cross-entropy with probability clipping.

        Args:
            y: Ground-truth binary labels.
            p: Predicted probabilities.

        Returns:
            Scalar mean cross-entropy loss.
        """
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Cross-entropy loss on a given dataset (no regularization term).

        Args:
            X: Feature matrix of shape (n, d).
            y: Ground-truth binary labels.

        Returns:
            Scalar cross-entropy loss.
        """
        return self.cross_entropy(y, self.predict_proba(X))

    def fit_epoch(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> None:
        """Run one epoch of shuffled mini-batch SGD in-place.

        Args:
            X: Feature matrix of shape (n, d).
            y: Binary labels of shape (n,).
            rng: Random generator used for shuffling.
        """
        n = len(y)
        idx = rng.permutation(n)
        X, y = X[idx], y[idx]

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            X_b, y_b = X[start:end], y[start:end]
            B = end - start

            p_b = self.predict_proba(X_b)
            error = p_b - y_b

            grad_w = (X_b.T @ error) / B + 2 * self.lam * self.w
            grad_b = error.sum() / B
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_epochs: int = 200,
        rng: np.random.Generator | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        patience: int = 10,
    ) -> dict:
        """Train with optional early stopping on validation loss.

        Stops early when val loss does not improve for `patience` epochs and
        restores the best weights.

        Args:
            X: Training feature matrix.
            y: Training labels.
            max_epochs: Maximum number of training epochs.
            rng: Random generator; a fresh one is created if None.
            X_val: Validation features for early stopping.
            y_val: Validation labels for early stopping.
            patience: Epochs without improvement before stopping.

        Returns:
            History dict with per-epoch train_loss, train_acc, and optionally
            val_loss and val_acc.
        """
        rng = rng if rng is not None else np.random.default_rng()
        history: dict[str, list] = {"train_loss": [], "train_acc": []}
        use_early_stopping = X_val is not None
        if use_early_stopping:
            history["val_loss"] = []
            history["val_acc"] = []
            best_val_loss = np.inf
            best_w = self.w.copy()
            best_b = self.b
            epochs_no_improve = 0

        for _ in range(max_epochs):
            self.fit_epoch(X, y, rng)

            p_train = self.predict_proba(X)
            history["train_loss"].append(self.cross_entropy(y, p_train))
            history["train_acc"].append(np.mean((p_train >= 0.5) == y))

            if use_early_stopping:
                val_loss = self.compute_loss(X_val, y_val)
                p_val = self.predict_proba(X_val)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(np.mean((p_val >= 0.5) == y_val))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_w = self.w.copy()
                    best_b = self.b
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        self.w = best_w
                        self.b = best_b
                        break

        return history
