"""Helpers for the independent second implementation of the qRRXOR interventions (the
verify_* experiments): process setup, prefix enumeration, branch-tree theory, weighted ridge
probes, model loading, matched pairs, recurrent rollouts."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformer_lens import HookedTransformer, HookedTransformerConfig

from epsilon_transformers.process.transition_matrices import get_matrix_from_args

XF_DIR = Path("results/qrrxor_robust_transformer")
OUT_DIR = Path("results/causal_transplant/verify")
N_CTX = 10
P = 7                       # patch position: prefix length 7, position index 6
ROLL = N_CTX - P
EPS = 1e-12
FAMS = [("results/qrrxor_robust_gru", "GRU"),
        ("results/qrrxor_robust_lstm", "LSTM"),
        ("results/qrrxor_robust_rnn", "RNN")]


def process_setup(proc_cfg):
    pc = dict(proc_cfg); name = pc.pop("name")
    T = get_matrix_from_args(name=name, **pc)          # (vocab, d, d)
    vocab, d = T.shape[0], T.shape[1]
    Pm = T.sum(0)
    ev, evec = np.linalg.eig(Pm)                       # right eigvec: P rev = rev
    rev = np.real(evec[:, np.argmin(np.abs(ev - 1.0))]); rev /= rev.sum()
    ev2, evec2 = np.linalg.eig(Pm.T)                   # left eigvec: b0 P = b0
    b0 = np.real(evec2[:, np.argmin(np.abs(ev2 - 1.0))]); b0 /= b0 @ rev
    E = np.stack([T[x] @ rev for x in range(vocab)], 1)     # (d, vocab)
    Pcol = E @ np.linalg.pinv(E.T @ E) @ E.T
    Pker = np.eye(d) - Pcol
    return T, rev, b0, E, Pcol, Pker, vocab, d


def enum_prefixes(T, rev, b0, plen):
    """All vocab^plen prefixes in big-endian lex order: tokens, beliefs, weights, p0."""
    vocab, d = T.shape[0], T.shape[1]
    toks = np.array([[(i >> (plen - 1 - t)) & 1 for t in range(plen)]
                     for i in range(2 ** plen)], dtype=np.int64)
    bel = np.zeros((len(toks), d)); wt = np.zeros(len(toks))
    for i, tk in enumerate(toks):
        b = b0.copy(); pr = 1.0
        for x in tk:
            px = max(float(b @ T[x] @ rev), 0.0)
            pr *= px
            bn = b @ T[x]
            nz = float(bn @ rev)
            if nz > 0:
                b = bn / nz
        bel[i] = b; wt[i] = pr
    wt /= wt.sum()
    p0 = bel @ T[0] @ rev
    return toks, bel, wt, p0


def branch_mats(T, k):
    """Cumulative transition products for the 2^k branches (big-endian)."""
    branches = np.array([[(u >> (k - 1 - t)) & 1 for t in range(k)]
                         for u in range(2 ** k)], dtype=np.int64)
    mats = []
    for u in branches:
        M = np.eye(T.shape[1])
        for x in u:
            M = M @ T[x]
        mats.append(M)
    return branches, np.stack(mats)


def theory_tree(T, rev, bels, bmats):
    out = np.stack([bels @ M @ rev for M in bmats], 1)
    out = np.clip(out, 0, None)
    return out / out.sum(1, keepdims=True)


def tree_kl(Pt, Qt):
    Pt = np.clip(Pt, EPS, None); Pt = Pt / Pt.sum(-1, keepdims=True)
    Qt = np.clip(Qt, EPS, None); Qt = Qt / Qt.sum(-1, keepdims=True)
    return (Pt * (np.log(Pt) - np.log(Qt))).sum(-1)


def wridge(X, Y, w, lam=1e-6):
    """Weighted ridge affine fit; returns beta ((D+1) x k)."""
    w = w / w.sum()
    Xb = np.hstack([np.ones((len(X), 1)), X])
    sw = np.sqrt(w)[:, None]
    A = (Xb * sw).T @ (Xb * sw) + lam * np.eye(Xb.shape[1])
    A[0, 0] -= lam                                    # don't penalize intercept
    return np.linalg.solve(A, (Xb * sw).T @ (Y * sw))


def wr2(X, Y, w, beta):
    w = w / w.sum()
    Yp = np.hstack([np.ones((len(X), 1)), X]) @ beta
    ym = (Y * w[:, None]).sum(0)
    ssr = float((w[:, None] * (Y - Yp) ** 2).sum())
    sst = float((w[:, None] * (Y - ym) ** 2).sum())
    return 1 - ssr / sst if sst > 0 else 0.0


def cv_ridge_r2(X, Y, w, seed=0, lams=(1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """5-fold CV pooled weighted R2, best over ridge grid; rows must be unique units.
    Returns (best_r2, best_lam)."""
    n = len(X)
    if n < 8:
        return float("nan"), float("nan")
    nf = 5 if n >= 20 else 4
    w = w / w.sum()
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), nf)
    ym = (Y * w[:, None]).sum(0)
    sst = float((w[:, None] * (Y - ym) ** 2).sum())
    if sst <= 0:
        return float("nan"), float("nan")
    best, blam = -np.inf, lams[0]
    for lam in lams:
        ssr = 0.0
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            beta = wridge(X[tr], Y[tr], w[tr], lam)
            Yp = np.hstack([np.ones((len(te), 1)), X[te]]) @ beta
            ssr += float((w[te][:, None] * (Y[te] - Yp) ** 2).sum())
        r2 = 1 - ssr / sst
        if r2 > best:
            best, blam = r2, lam
    return float(best), float(blam)


def load_xf(run_dir, device):
    with open(run_dir / "hooked_model_config.json") as f:
        mc = json.load(f)
    mc["dtype"] = getattr(torch, mc["dtype"].split(".")[-1])
    mc["device"] = device
    model = HookedTransformer(HookedTransformerConfig(**mc)).to(device)
    pts = sorted(run_dir.glob("*.pt"), key=lambda p: int(p.stem))
    state = torch.load(pts[-1], map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def run_cfg(run_dir):
    with open(run_dir / "run_config.yaml") as f:
        return yaml.safe_load(f)


def matched_pairs(bel, wt, p0, Pker, n_pairs=200, tol=1e-3, all_matched=False):
    N = len(bel)
    ii, jj = np.triu_indices(N, k=1)
    dp0 = np.abs(p0[ii] - p0[jj])
    kern = np.linalg.norm((bel[ii] - bel[jj]) @ Pker, axis=1)
    ok = (dp0 < tol) & (np.minimum(wt[ii], wt[jj]) > np.quantile(wt, 0.10))
    if all_matched:
        keep = np.where(ok)[0]
    else:
        order = np.argsort(-kern)
        keep = order[ok[order]][:n_pairs]
    return np.concatenate([ii[keep], jj[keep]]), np.concatenate([jj[keep], ii[keep]])


def softmax_at(model, x, hooks=None, bs=4096):
    outs = []
    for s in range(0, len(x), bs):
        xb = x[s:s + bs]
        with torch.no_grad():
            if hooks:
                lg = model.run_with_hooks(xb, fwd_hooks=hooks(s, s + len(xb)))
            else:
                lg = model(xb)
        outs.append(F.softmax(lg[:, P - 1:P - 1 + ROLL, :].float(), -1).cpu().numpy())
    return np.concatenate(outs)


class Wrap(nn.Module):
    def __init__(self, rnn, out):
        super().__init__()
        self.rnn = rnn
        self.output_layer = out


def build(mc, vocab, device, state=None, init_seed=None):
    if init_seed is not None:
        torch.manual_seed(init_seed)
    cls = getattr(nn, mc["rnn_type"].upper())
    rnn = cls(input_size=vocab, hidden_size=mc["hidden_size"],
              num_layers=mc["num_layers"], batch_first=True)
    m = Wrap(rnn, nn.Linear(mc["hidden_size"], vocab)).to(device)
    if state is not None:
        m.load_state_dict(state)
    m.eval()
    return m


def states_last(model, toks, vocab, device, is_lstm):
    """Carried state after consuming toks: (N, D), h-layers then c-layers."""
    x = torch.from_numpy(toks).long().to(device)
    oh = F.one_hot(x, num_classes=vocab).float()
    with torch.no_grad():
        _, h = model.rnn(oh)
    parts = h if is_lstm else (h,)
    return torch.cat([p[i] for p in parts for i in range(p.shape[0])], 1)


def rollout(model, states, branches, L, H, is_lstm, device, vocab=2):
    """Step probs of branch tokens from given carried states: (n, nb, ROLL)."""
    n, nb = len(states), len(branches)
    st = states if torch.is_tensor(states) else torch.tensor(
        states, dtype=torch.float32, device=device)
    with torch.no_grad():
        p1 = F.softmax(model.output_layer(st[:, (L - 1) * H:L * H]), -1).cpu().numpy()
    out = np.zeros((n, nb, ROLL))
    out[:, :, 0] = p1[:, branches[:, 0]]
    rep = st.repeat_interleave(nb, 0)                       # (n*nb, D)
    n_parts = 2 if is_lstm else 1
    blocks = rep.view(-1, n_parts * L, H).permute(1, 0, 2).contiguous()
    h0 = (blocks[:L], blocks[L:]) if is_lstm else blocks
    bt = torch.from_numpy(np.tile(branches[:, :ROLL - 1], (n, 1))).long().to(device)
    oh = F.one_hot(bt, num_classes=vocab).float()
    with torch.no_grad():
        o, _ = model.rnn(oh, h0)
        pr = F.softmax(model.output_layer(o), -1).cpu().numpy()  # (n*nb, ROLL-1, vocab)
    pr = pr.reshape(n, nb, ROLL - 1, vocab)
    for s in range(1, ROLL):
        out[:, :, s] = pr[:, np.arange(nb), s - 1, branches[:, s]]
    return out


def hybrid_tree(T, rev, bel_w, bel_v, branches):
    """P(u1|eta_w) * P(u2..|u1, eta_v): donor conditional with w's first step."""
    p1w = np.stack([np.clip(bel_w @ T[x] @ rev, EPS, None) for x in range(T.shape[0])], 1)
    p1w /= p1w.sum(1, keepdims=True)
    out = np.zeros((len(bel_w), len(branches)))
    for u, tk in enumerate(branches):
        M = np.eye(T.shape[1])
        for x in tk:
            M = M @ T[x]
        num = np.clip(bel_v @ M @ rev, EPS, None)             # P(u1..uk | eta_v)
        den = np.clip(bel_v @ T[tk[0]] @ rev, EPS, None)      # P(u1 | eta_v)
        out[:, u] = p1w[:, tk[0]] * num / den
    return out / out.sum(1, keepdims=True)


def pooled_probe(model, T, rev, b0, vocab, device, is_lstm):
    Xs, Ys, Ws = [], [], []
    for plen in range(4, N_CTX):
        tk, bl, w, _ = enum_prefixes(T, rev, b0, plen)
        Xs.append(states_last(model, tk, vocab, device, is_lstm).cpu().numpy())
        Ys.append(bl); Ws.append(w)
    X, Y, W = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ws)
    beta = wridge(X, Y, W, 1e-6)
    return X, Y, W, beta
