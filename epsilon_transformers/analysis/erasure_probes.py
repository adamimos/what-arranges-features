"""Weighted ridge / linear / MLP probes with grouped CV, used by the erasure experiments."""

import numpy as np
import torch
import torch.nn as nn

RIDGE = 1e-4                              # original surgery-fit ridge
RIDGES = (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def wfit(X, Y, w, ridge=RIDGE):
    """Weighted ridge affine fit Y ~ [1,X] (identical to the original script)."""
    w = w / w.sum()
    Xb = np.hstack([np.ones((len(X), 1)), X])
    sw = np.sqrt(w)[:, None]
    A = (Xb * sw).T @ (Xb * sw) + ridge * np.eye(Xb.shape[1])
    B = (Xb * sw).T @ (Y * sw)
    return np.linalg.solve(A, B)


def ridge_cv_best(X, Y, w, seed=0):
    """Ridge CV over unique-prefix rows; pooled weighted R^2 vs the global
    weighted mean baseline. Returns (best_r2, best_ridge) over the grid."""
    n = len(X)
    if n < 8:
        return float("nan"), float("nan")
    n_folds = 5 if n >= 20 else 4
    w = w / w.sum()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    ym_all = (Y * w[:, None]).sum(0)
    sst = float((w[:, None] * (Y - ym_all) ** 2).sum())
    if sst <= 0:
        return float("nan"), float("nan")
    best, best_ridge = -np.inf, RIDGES[0]
    for ridge in RIDGES:
        ssr = 0.0
        for te in folds:
            mask = np.ones(n, bool); mask[te] = False
            tr = np.where(mask)[0]
            wt_ = w[tr] / w[tr].sum()
            mx = (X[tr] * wt_[:, None]).sum(0)
            sx = np.sqrt((wt_[:, None] * (X[tr] - mx) ** 2).sum(0)).clip(1e-8)
            my = (Y[tr] * wt_[:, None]).sum(0)
            Xs = (X[tr] - mx) / sx
            Yc = Y[tr] - my
            sw = np.sqrt(wt_)[:, None]
            A = (Xs * sw).T @ (Xs * sw) + ridge * np.eye(X.shape[1])
            B = (Xs * sw).T @ (Yc * sw)
            beta = np.linalg.solve(A, B)
            Yp = ((X[te] - mx) / sx) @ beta + my
            ssr += float((w[te][:, None] * (Y[te] - Yp) ** 2).sum())
        r2 = 1 - ssr / sst
        if r2 > best:
            best, best_ridge = r2, ridge
    return float(best), float(best_ridge)


def probe_linear_cv(X, Y, w, n_folds=None, seed=0):
    return ridge_cv_best(X, Y, w, seed=seed)[0]


def fit_ridge_raw(X, Y, w, ridge):
    """Weighted ridge fit on standardized X, returned as a raw-space affine
    beta ((D+1) x k): Y_hat = [1, X] @ beta."""
    w = w / w.sum()
    mx = (X * w[:, None]).sum(0)
    sx = np.sqrt((w[:, None] * (X - mx) ** 2).sum(0)).clip(1e-8)
    my = (Y * w[:, None]).sum(0)
    Xs = (X - mx) / sx
    Yc = Y - my
    sw = np.sqrt(w)[:, None]
    A = (Xs * sw).T @ (Xs * sw) + ridge * np.eye(X.shape[1])
    B = (Xs * sw).T @ (Yc * sw)
    Bstd = np.linalg.solve(A, B)
    slope = Bstd / sx[:, None]
    return np.vstack([(my - mx @ slope)[None, :], slope])


def probe_mlp_cv(X, Y, w, device, n_folds=4, seed=0, hidden=128, steps=2500,
                 lr=3e-3, wd=1e-4):
    """1-hidden-layer MLP probe, weighted MSE, pooled CV R^2 (same baseline
    convention as probe_linear_cv)."""
    n = len(X)
    if n < 24:
        return float("nan")
    w = w / w.sum()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    ym_all = (Y * w[:, None]).sum(0)
    sst = float((w[:, None] * (Y - ym_all) ** 2).sum())
    if sst <= 0:
        return float("nan")
    ssr = 0.0
    for fi, te in enumerate(folds):
        mask = np.ones(n, bool); mask[te] = False
        tr = np.where(mask)[0]
        wt_ = w[tr] / w[tr].sum()
        mx = (X[tr] * wt_[:, None]).sum(0)
        sx = np.sqrt((wt_[:, None] * (X[tr] - mx) ** 2).sum(0)).clip(1e-8)
        my = (Y[tr] * wt_[:, None]).sum(0)
        sy = np.sqrt((wt_[:, None] * (Y[tr] - my) ** 2).sum(0)).clip(1e-8)
        Xt = torch.tensor((X[tr] - mx) / sx, dtype=torch.float32, device=device)
        Yt = torch.tensor((Y[tr] - my) / sy, dtype=torch.float32, device=device)
        Wt = torch.tensor(wt_, dtype=torch.float32, device=device)
        torch.manual_seed(seed * 1000 + fi)
        net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(),
                            nn.Linear(hidden, Y.shape[1])).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        for _ in range(steps):
            opt.zero_grad()
            loss = (Wt[:, None] * (net(Xt) - Yt) ** 2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            Xe = torch.tensor((X[te] - mx) / sx, dtype=torch.float32, device=device)
            Yp = net(Xe).cpu().numpy() * sy + my
        ssr += float((w[te][:, None] * (Y[te] - Yp) ** 2).sum())
    return float(1 - ssr / sst)
