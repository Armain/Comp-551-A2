import numpy as np


class LogisticRegressionSGD:
    """Logistic regression trained with mini-batch SGD and L2 regularization.

    Regularization convention: lambda * ||w||_2^2 (gradient term: 2*lambda*w).
    Bias is excluded from the penalty.
    """

    def __init__(self, n_features, lr=0.01, lam=1e-3, batch_size=16, init_scale=0.0, rng=None):
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
    def sigmoid(z):
        """Numerically stable sigmoid."""
        result = np.empty_like(z, dtype=float)
        pos = z >= 0
        result[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        result[~pos] = exp_z / (1.0 + exp_z)
        return result

    def predict_proba(self, X):
        """Return P(y=1|X) as array of shape (n,)."""
        z = X @ self.w + self.b
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    @staticmethod
    def cross_entropy(y, p):
        """Mean binary cross-entropy. Clips probabilities for stability."""
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def compute_loss(self, X, y):
        """Cross-entropy loss (no regularization term)."""
        return self.cross_entropy(y, self.predict_proba(X))

    def fit_epoch(self, X, y, rng):
        """One epoch of mini-batch SGD. Shuffles data and iterates over batches."""
        n = len(y)
        idx = rng.permutation(n)
        X, y = X[idx], y[idx]

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            X_b = X[start:end]
            y_b = y[start:end]
            B = end - start

            p_b = self.predict_proba(X_b)
            error = p_b - y_b

            grad_w = (X_b.T @ error) / B + 2 * self.lam * self.w
            grad_b = error.sum() / B  # bias excluded from regularization penalty

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def fit(self, X, y, max_epochs=200, rng=None, X_val=None, y_val=None, patience=10):
        """Train up to max_epochs with optional early stopping on val loss.

        If X_val is provided, stops early when val loss does not improve for
        `patience` epochs and restores the best weights.
        Returns history dict with per-epoch metrics.
        """
        rng = rng if rng is not None else np.random.default_rng()
        history = {"train_loss": [], "train_acc": []}
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
