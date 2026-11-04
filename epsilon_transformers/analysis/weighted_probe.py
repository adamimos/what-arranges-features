"""Weighted affine probe with K-fold CV (the R^2 used by the 5-seed scorers)."""

import numpy as np


def _weighted_r2(X, Y, weights, n_folds=5, rcond=1e-10):
    """Cross-validated weighted R².

    Fit on train folds, evaluate on held-out test fold, average across folds.

    Parameters
    ----------
    X : ndarray, shape (N, d_features)
    Y : ndarray, shape (N, d_targets)
    weights : ndarray, shape (N,)
    n_folds : int
    rcond : float — regularization for lstsq

    Returns
    -------
    r2 : float — average held-out weighted R²
    """
    N = X.shape[0]
    w = weights / weights.sum()

    indices = np.arange(N)
    fold_size = N // n_folds
    fold_r2s = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else N
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]
        w_train = w[train_idx]
        w_test = w[test_idx]

        # Standardize using train statistics
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-12] = 1.0

        X_train_bias = np.hstack([np.ones((len(train_idx), 1)),
                                   (X_train - mean) / std])
        X_test_bias = np.hstack([np.ones((len(test_idx), 1)),
                                  (X_test - mean) / std])

        # Weighted train
        sqrt_w_train = np.sqrt(w_train / w_train.sum())[:, None]
        beta, _, _, _ = np.linalg.lstsq(
            X_train_bias * sqrt_w_train, Y_train * sqrt_w_train, rcond=rcond
        )

        # Evaluate on test fold
        Y_pred = X_test_bias @ beta
        w_test_norm = w_test / w_test.sum()
        Y_mean = (Y_test * w_test_norm[:, None]).sum(axis=0)
        ss_res = (w_test_norm[:, None] * (Y_test - Y_pred) ** 2).sum()
        ss_tot = (w_test_norm[:, None] * (Y_test - Y_mean) ** 2).sum()
        fold_r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)

    return float(np.mean(fold_r2s))
