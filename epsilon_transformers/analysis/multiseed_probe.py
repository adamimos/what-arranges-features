"""Shared pieces of the 5-seed grid scorers: belief and Markov-order targets for a process,
concatenated activations from transformer and recurrent checkpoints, and the convergence check."""

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from epsilon_transformers.analysis.checkpoints import load_run_config, load_model
from epsilon_transformers.analysis.process_targets import enumerate_process_data
from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM, markov_approximation
from epsilon_transformers.training.networks import RNNWrapper

N_FOLDS = 5
RCOND = 1e-10
CONV_THRESH = 1.005      # last-row loss.csv val_loss_mean < 1.005 => converged
MARKOV_ORDER = 3
SHUFFLE_SEED = 0         # fixed row permutation for CV folds


def _stationary_belief(T, rev):
    P = T.sum(axis=0)
    ev, evec = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(ev - 1.0))
    b0 = np.real(evec[:, idx])
    return b0 / (b0 @ rev)


def _walk_beliefs(T, rev, b0, contexts):
    """Belief at the last position for each context, walking the GHMM."""
    n_states = T.shape[1]
    out = np.zeros((len(contexts), n_states), dtype=np.float64)
    for i, toks in enumerate(contexts):
        b = b0.copy()
        for x in toks:
            bn = b @ T[int(x)]
            nrm = float(bn @ rev)
            if nrm > 0:
                b = bn / nrm
        out[i] = b
    return out


def build_process_targets(cfg):
    """Return dict with contexts, belief target, markov3 target, weights."""
    proc = enumerate_process_data(cfg)
    n_ctx = cfg["model_config"]["n_ctx"]
    contexts = proc["seqs"].numpy()[:, :n_ctx]     # tokens the model sees
    belief = proc["beliefs"]                        # minimal/GHMM belief
    weights = proc["weights"]

    T = get_matrix_from_args(**cfg["process_config"])
    vocab = int(T.shape[0])

    ghmm = TransitionMatrixGHMM(T)
    m3 = markov_approximation(ghmm, order=MARKOV_ORDER)
    T3 = m3.transition_matrices
    rev3 = np.asarray(m3.right_eigenvector).squeeze()
    b0_3 = _stationary_belief(T3, rev3)
    markov3 = _walk_beliefs(T3, rev3, b0_3, contexts)

    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(len(weights))
    return {
        "contexts": contexts, "n_ctx": n_ctx, "vocab": vocab,
        "belief": belief, "markov3": markov3, "weights": weights,
        "perm": perm,
        "belief_dim": int(belief.shape[1]), "markov3_dim": int(markov3.shape[1]),
        "N": int(len(weights)),
    }


def extract_transformer_concat(run_dir, ckpt, contexts, device, bs=8192):
    """Concat hook_resid_pre across all layers, last position."""
    m = load_model(str(run_dir), ckpt, device)
    n_layers = m.cfg.n_layers
    names = [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)]
    chunks = {k: [] for k in names}
    inp = torch.from_numpy(contexts).long()
    with torch.no_grad():
        for s in range(0, inp.shape[0], bs):
            _, cache = m.run_with_cache(inp[s:s + bs].to(device), names_filter=names)
            for k in names:
                chunks[k].append(cache[k][:, -1, :].cpu().numpy())
    del m
    if "cuda" in str(device):
        torch.cuda.empty_cache()
    return np.concatenate([np.concatenate(chunks[k], 0) for k in names], axis=1)


def load_rnn(run_dir, ckpt, device, vocab):
    cfg = load_run_config(str(run_dir))
    mc = cfg["model_config"]
    cls = getattr(nn, mc["rnn_type"].upper())
    rnn = cls(input_size=vocab, hidden_size=mc["hidden_size"],
              num_layers=mc["num_layers"], dropout=mc.get("dropout", 0.0),
              batch_first=True, bidirectional=mc.get("bidirectional", False)).to(device)
    out = nn.Linear(mc["hidden_size"], vocab).to(device)
    model = RNNWrapper(rnn, out).to(device)
    state = torch.load(f"{run_dir}/{ckpt}", map_location=device, weights_only=True)
    model.load_state_dict(state)      # asserts vocab correctness via shape match
    model.eval()
    return model, cfg


def extract_rnn_concat(model, contexts, vocab, device, bs=8192):
    """Concat h_n across all layers, last time step. Handles LSTM (h_n,c_n)
    tuple and GRU/vanilla-RNN (h_n tensor)."""
    outs = []
    with torch.no_grad():
        for s in range(0, len(contexts), bs):
            x = torch.from_numpy(contexts[s:s + bs]).long().to(device)
            oh = torch.zeros(*x.shape, vocab, device=device)
            oh.scatter_(2, x.unsqueeze(-1), 1.0)
            _, hidden = model.rnn(oh)
            h_n = hidden[0] if isinstance(hidden, tuple) else hidden  # (num_layers,N,H)
            outs.append(torch.cat([h_n[i] for i in range(h_n.shape[0])], dim=1).cpu().numpy())
    return np.concatenate(outs, 0)


def final_val_lb(run_dir: Path) -> float:
    f = run_dir / "loss.csv"
    if not f.exists():
        return float("inf")
    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    return float(rows[-1]["val_loss_mean"]) if rows else float("inf")


def _design(X, mean, std):
    return np.hstack([np.ones((len(X), 1)), (X - mean) / std])


def _fit(Xb, Y, w):
    s = np.sqrt(w / w.sum())[:, None]
    beta, *_ = np.linalg.lstsq(Xb * s, Y * s, rcond=RCOND)
    return beta


def scores(X, Y, weights):
    """Same folds/standardization/weighting as _weighted_r2, plus RMSEs."""
    N = X.shape[0]
    w = weights / weights.sum()
    idx = np.arange(N)
    fs = N // N_FOLDS
    r2s, pred = [], np.zeros_like(Y)
    for f in range(N_FOLDS):
        a = f * fs
        b = a + fs if f < N_FOLDS - 1 else N
        te, tr = idx[a:b], np.concatenate([idx[:a], idx[b:]])
        mean, std = X[tr].mean(0), X[tr].std(0)
        std[std < 1e-12] = 1.0
        beta = _fit(_design(X[tr], mean, std), Y[tr], w[tr])
        yp = _design(X[te], mean, std) @ beta
        pred[te] = yp
        wt = w[te] / w[te].sum()
        ym = (Y[te] * wt[:, None]).sum(0)
        ss_res = (wt[:, None] * (Y[te] - yp) ** 2).sum()
        ss_tot = (wt[:, None] * (Y[te] - ym) ** 2).sum()
        r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    rmse_cv = float(np.sqrt((w[:, None] * (Y - pred) ** 2).sum()))
    mean, std = X.mean(0), X.std(0)
    std[std < 1e-12] = 1.0
    Xb = _design(X, mean, std)
    rmse_in = float(np.sqrt((w[:, None] * (Y - Xb @ _fit(Xb, Y, w)) ** 2).sum()))
    return float(np.mean(r2s)), rmse_cv, rmse_in


def acts_for(run_dir, ckpt, cfg, proc, device):
    if "rnn_type" in cfg["model_config"]:
        m, _ = load_rnn(run_dir, ckpt, device, proc["vocab"])
        a = extract_rnn_concat(m, proc["contexts"], proc["vocab"], device)
        del m
    else:
        a = extract_transformer_concat(run_dir, ckpt, proc["contexts"], device)
    if "cuda" in str(device):
        torch.cuda.empty_cache()
    return a
