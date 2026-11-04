"""Exact next-token loss of order-k predictors vs the optimal predictor (App. on higher-order classical baselines).

For a stationary process with GHMM matrices T[x] (shape V,S,S), word probabilities are
p(w) = b0 T[w1]...T[wL] r. Block entropy H(L) = -sum_w p log p. The optimal predictor at
position t (t-1 tokens of context) has loss H(t)-H(t-1); an order-k predictor uses the last
min(k, t-1) tokens, loss H(m+1)-H(m). Averaged over positions 1..n_ctx (as in training).

Usage (repo root): PYTHONPATH=. .venv/bin/python scripts/paper_metrics/orderk_loss.py
"""
import csv
from pathlib import Path
import glob

import numpy as np

from scripts.activation_analysis.run_shuffle_analysis import load_run_config
from epsilon_transformers.process.transition_matrices import get_matrix_from_args

PROCS = {"Bloch": "results/multiseed_bloch_transformer", "Mess3": "results/multiseed_mess3_transformer",
         "Moon": "results/multiseed_moon_transformer"}
OUT = "results/paper_metrics/orderk_loss.csv"


def stationary(T):
    P = T.sum(0)
    ev, R = np.linalg.eig(P)
    r = np.real(R[:, np.argmin(abs(ev - 1))])
    ev, L = np.linalg.eig(P.T)
    b = np.real(L[:, np.argmin(abs(ev - 1))])
    return b / (b @ r), r


def block_entropies(T, Lmax):
    b0, r = stationary(T)
    V = T.shape[0]
    H = [0.0]
    vecs = b0[None, :]
    for L in range(1, Lmax + 1):
        vecs = np.einsum("ns,xst->nxt", vecs, T).reshape(-1, T.shape[1])
        p = vecs @ r
        assert abs(p.sum() - 1) < 1e-8 and p.min() > -1e-12, (L, p.sum(), p.min())
        p = p[p > 0]
        H.append(float(-(p * np.log(p)).sum()))
    return H


rows = []
for name, d in PROCS.items():
    cfg = load_run_config(sorted(glob.glob(f"{d}/run_*"))[0])
    T = np.asarray(get_matrix_from_args(**cfg["process_config"]), dtype=np.float64)
    n = cfg["model_config"]["n_ctx"]
    H = block_entropies(T, n)
    cond = [H[m + 1] - H[m] for m in range(n)]          # loss with m tokens of context
    opt = np.mean([cond[t - 1] for t in range(1, n + 1)])
    for k in range(0, n):
        lk = np.mean([cond[min(k, t - 1)] for t in range(1, n + 1)])
        # training normalization: per-position loss / per-position optimum, then mean
        rpos = np.mean([cond[min(k, t - 1)] / cond[t - 1] for t in range(1, n + 1)])
        rows.append([name, n, k, lk, opt, lk / opt, lk - opt, rpos])
        print(f"{name:5s} k={k} loss={lk:.5f}  optimal={opt:.5f}  ratio={lk/opt:.6f}  per-position ratio={rpos:.6f}")
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["process", "n_ctx", "order_k", "orderk_mean_loss", "optimal_mean_loss", "ratio", "excess_nats", "per_position_ratio_mean"])
    w.writerows(rows)
print("wrote", OUT)
