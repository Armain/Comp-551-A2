import numpy as np


class LogisticRegressionSGD:
    """Logistic regression trained with mini-batch SGD and L2 regularization.

    Regularization convention: lambda * ||w||_2^2 (gradient term: 2*lambda*w).
    Bias is excluded from the penalty.
    """

    def __init__(self, n_features, lr=0.01, lam=1e-3, batch_size=16):
        self.w = np.zeros(n_features)
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
            grad_b = error.sum() / B

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def fit(self, X, y, epochs, rng, X_val=None, y_val=None):
        """Train for given epochs. Returns history dict with per-epoch metrics."""
        history = {"train_loss": [], "train_acc": []}
        if X_val is not None:
            history["val_loss"] = []
            history["val_acc"] = []

        for _ in range(epochs):
            self.fit_epoch(X, y, rng)

            p_train = self.predict_proba(X)
            history["train_loss"].append(self.cross_entropy(y, p_train))
            history["train_acc"].append(np.mean((p_train >= 0.5) == y))

            if X_val is not None:
                p_val = self.predict_proba(X_val)
                history["val_loss"].append(self.cross_entropy(y_val, p_val))
                history["val_acc"].append(np.mean((p_val >= 0.5) == y_val))

        return history
