"""Intervention machinery for the qRRXOR patching experiments: exact prefix beliefs, weighted
affine probes, recurrent state collection, branch-tree theory targets and KLs, matched-pair
selection, and the recurrent and transformer patching backends."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from epsilon_transformers.analysis.checkpoints import (
    load_run_config, list_checkpoints, load_model,
)
from epsilon_transformers.analysis.qrrxor_models import build_rnn

PATCH_P = 7
N_CTX = 10
ROLL = N_CTX - PATCH_P            # 3 predicted tokens
PAIR_TOL = 1e-3
N_PAIRS = 200
XF_SITES = ("blocks.1.hook_resid_pre", "blocks.2.hook_resid_pre", "blocks.3.hook_resid_pre")
EPS = 1e-12


def prefix_beliefs(T, rev, b0, P):
    """All vocab^P prefixes (lex order, big-endian over time): tokens, belief,
    probability, p(x=0|belief)."""
    vocab, dim = T.shape[0], T.shape[1]
    n = vocab ** P
    toks = np.zeros((n, P), dtype=np.int64)
    bel = np.zeros((n, dim))
    wt = np.zeros(n)
    for i in range(n):
        s = i
        tk = np.zeros(P, dtype=np.int64)
        for t in range(P):
            tk[P - 1 - t] = s % vocab
            s //= vocab
        b = b0.copy()
        prob = 1.0
        for x in tk:
            p_x = max(float(b @ T[x] @ rev), 0.0)
            prob *= p_x
            b_new = b @ T[x]
            nrm = float(b_new @ rev)
            if nrm > 0:
                b = b_new / nrm
        toks[i] = tk
        bel[i] = b
        wt[i] = prob
    wt = wt / wt.sum()
    p0 = bel @ T[0] @ rev
    return toks, bel, wt, p0


def fit_affine(X, Y, w, rcond=1e-10):
    """Weighted affine fit Y ~ [1, X] beta. Returns beta ((D+1) x dy)."""
    w = w / w.sum()
    sw = np.sqrt(w)[:, None]
    Xb = np.hstack([np.ones((len(X), 1)), X])
    beta, *_ = np.linalg.lstsq(Xb * sw, Y * sw, rcond=rcond)
    return beta


def weighted_r2_eval(beta, X, Y, w):
    Yp = np.hstack([np.ones((len(X), 1)), X]) @ beta
    w = w / w.sum()
    ym = (Y * w[:, None]).sum(0)
    ssr = (w[:, None] * (Y - Yp) ** 2).sum()
    sst = (w[:, None] * (Y - ym) ** 2).sum()
    return float(1 - ssr / sst) if sst > 0 else 0.0


def rnn_states_all_steps(model, seqs, vocab, device, bs=2048):
    """Full carried state at every step. Returns (N, T, D) float32 where
    D = L*H for RNN/GRU and 2*L*H for LSTM (all h layers, then all c layers)."""
    outs = []
    with torch.no_grad():
        for s in range(0, len(seqs), bs):
            x = torch.from_numpy(seqs[s:s + bs]).long().to(device)
            oh = torch.nn.functional.one_hot(x, num_classes=vocab).float()
            h = None
            per_step = []
            for t in range(x.shape[1]):
                _, h = model.rnn(oh[:, t:t + 1], h)
                parts = h if isinstance(h, tuple) else (h,)   # LSTM: (h, c)
                per_step.append(torch.cat(
                    [p[i] for p in parts for i in range(p.shape[0])], dim=1))
            outs.append(torch.stack(per_step, dim=1).cpu().numpy())
    return np.concatenate(outs, 0)


def branch_tokens(k, vocab=2):
    """All vocab^k continuations, lex big-endian: (vocab^k, k)."""
    n = vocab ** k
    out = np.zeros((n, k), dtype=np.int64)
    for u in range(n):
        s = u
        for t in range(k):
            out[u, k - 1 - t] = s % vocab
            s //= vocab
    return out


def theory_tree(T, rev, bels, branches):
    """P(u|η) for each belief row and branch: (n_bel, n_branch). Clipped >= 0."""
    out = np.zeros((len(bels), len(branches)))
    for u, tk in enumerate(branches):
        M = np.eye(T.shape[1])
        for x in tk:
            M = M @ T[x]
        out[:, u] = bels @ M @ rev
    out = np.clip(out, 0.0, None)
    out = out / out.sum(axis=1, keepdims=True)
    return out


def tree_kl(P, Q):
    """KL(P||Q) rowwise over branch trees."""
    P = np.clip(P, EPS, None); P = P / P.sum(-1, keepdims=True)
    Q = np.clip(Q, EPS, None); Q = Q / Q.sum(-1, keepdims=True)
    return (P * (np.log(P) - np.log(Q))).sum(-1)


def stepwise_kls_from_probs(step_p, step_q, branches):
    """Per-step KL between two models' stepwise conditionals along the tree.

    step_p/step_q: (n, n_branch, ROLL) prob of the branch token at each step.
    Step s KL averaged over distinct length-(s-1) branch prefixes weighted by P.
    Returns (n, ROLL).
    """
    n, nb, K = step_p.shape
    out = np.zeros((n, K))
    for s in range(K):
        # group branches by their first s tokens; token at step s varies
        seen = {}
        for u in range(nb):
            key = tuple(branches[u, :s])
            seen.setdefault(key, []).append(u)
        tot = np.zeros(n)
        for key, us in seen.items():
            # weight of prefix under P = prod of first s step-probs (any branch in group)
            u0 = us[0]
            wpre = np.ones(n) if s == 0 else np.prod(step_p[:, u0, :s], axis=1)
            # distribution over next token at step s: collect distinct tokens
            toks = {}
            for u in us:
                toks.setdefault(branches[u, s], u)
            p = np.stack([step_p[:, u, s] for u in toks.values()], axis=1)
            q = np.stack([step_q[:, u, s] for u in toks.values()], axis=1)
            p = np.clip(p, EPS, None); p /= p.sum(1, keepdims=True)
            q = np.clip(q, EPS, None); q /= q.sum(1, keepdims=True)
            tot += wpre * (p * (np.log(p) - np.log(q))).sum(1)
        out[:, s] = tot
    return out


class RNNBackend:
    """Vanilla RNN: patch carried h (all layers) after 7 tokens, continue."""

    def __init__(self, run_dir, device, T, rev, b0, vocab):
        cfg = load_run_config(str(run_dir))
        self.seed = cfg["model_config"]["seed"]
        state = torch.load(Path(run_dir) / list_checkpoints(str(run_dir))[-1],
                           map_location=device, weights_only=True)
        self.model = build_rnn(cfg, vocab, device, state=state)
        self.device = device
        self.vocab = vocab
        self.L = self.model.rnn.num_layers
        self.H = self.model.rnn.hidden_size
        self.is_lstm = cfg["model_config"]["rnn_type"].upper() == "LSTM"
        # write probe: pooled positions 4..9 (validated in phase 0)
        Xs, Ys, Ws = [], [], []
        for p in range(4, N_CTX):
            tk_p, bel_p, wt_p, _ = prefix_beliefs(T, rev, b0, p)
            S = rnn_states_all_steps(self.model, tk_p, vocab, device)[:, -1, :]
            Xs.append(S); Ys.append(bel_p); Ws.append(wt_p)
        X = np.concatenate(Xs); Y = np.concatenate(Ys); W = np.concatenate(Ws)
        self.beta = fit_affine(X, Y, W / W.sum())
        self.Wmat = self.beta[1:]                      # (L*H, 4)
        self.pinvW = np.linalg.pinv(self.Wmat)         # (4, L*H)
        self.G = fit_affine(Y, X, W / W.sum())[1:]     # belief->state decoder (4, L*H)

    def states_at_patch(self, toks7):
        S = rnn_states_all_steps(self.model, toks7, self.vocab, self.device)[:, -1, :]
        return S                                       # (n, L*H)

    def rollout(self, states, branches, bs=65536):
        """states (n, L*H) numpy -> stepwise probs of branch tokens.

        Returns step_probs (n, n_branch, ROLL) where step 0 prob comes from the
        (patched) state's top-layer readout and steps 1.. from continuation.
        """
        n = len(states)
        nb = len(branches)
        dev = self.device
        st = torch.from_numpy(states).float().to(dev)
        # step-1 logits directly from top-layer h (first L*H block is h for LSTM)
        top = st[:, (self.L - 1) * self.H:self.L * self.H]
        with torch.no_grad():
            l1 = self.model.output_layer(top)          # (n, vocab)
        p1 = F.softmax(l1, dim=-1).cpu().numpy()
        # continuation: feed branch tokens 0..ROLL-2 from patched state
        bt = torch.from_numpy(branches).long()
        step_probs = np.zeros((n, nb, ROLL))
        step_probs[:, :, 0] = p1[:, branches[:, 0]]
        idx_pairs = [(i, u) for i in range(n) for u in range(nb)]
        n_parts = 2 if self.is_lstm else 1
        h0 = st.view(n, n_parts * self.L, self.H)
        for s0 in range(0, len(idx_pairs), bs):
            chunk = idx_pairs[s0:s0 + bs]
            ii = torch.tensor([c[0] for c in chunk])
            uu = torch.tensor([c[1] for c in chunk])
            x = bt[uu][:, :ROLL - 1].to(dev)           # feed first ROLL-1 branch tokens
            oh = F.one_hot(x, num_classes=self.vocab).float()
            hh = h0[ii].permute(1, 0, 2).contiguous()
            h = (hh[:self.L], hh[self.L:]) if self.is_lstm else hh
            with torch.no_grad():
                o, _ = self.model.rnn(oh, h)
                lg = self.model.output_layer(o)        # (chunk, ROLL-1, vocab)
            pr = F.softmax(lg, dim=-1).cpu().numpy()
            for k, (i, u) in enumerate(chunk):
                for s in range(1, ROLL):
                    step_probs[i, u, s] = pr[k, s - 1, branches[u, s]]
        return step_probs


class XFBackend:
    """Transformer: patch resid at `site`, position 6, run full sequences."""

    def __init__(self, run_dir, device, T, rev, b0, vocab, site):
        cfg = load_run_config(str(run_dir))
        self.seed = cfg["model_config"]["seed"]
        self.model = load_model(str(run_dir), list_checkpoints(str(run_dir))[-1], device)
        self.device = device
        self.vocab = vocab
        self.site = site
        toks7, bel7, wt7, _ = prefix_beliefs(T, rev, b0, PATCH_P)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                torch.from_numpy(toks7).long().to(device), names_filter=[site])
        A_full = cache[site].float().cpu().numpy()          # (128, 7, d)
        A = A_full[:, -1, :]
        self.beta = fit_affine(A, bel7, wt7 / wt7.sum())
        self.Wmat = self.beta[1:]
        self.pinvW = np.linalg.pinv(self.Wmat)
        self.G = fit_affine(bel7, A, wt7 / wt7.sum())[1:]   # belief->state decoder
        self._A7 = A
        self._A_full = A_full
        # per-position beliefs (prefix lengths 4..7) and decoders for prefix_bel
        self.pos_bel, self.pos_G = {}, {}
        for p_idx in range(3, PATCH_P):                     # position index = prefix len - 1
            plen = p_idx + 1
            _, bel_p, _, _ = prefix_beliefs(T, rev, b0, plen)
            # map each of the 128 prefixes to its length-plen prefix belief
            idx = np.array([int("".join(map(str, tk[:plen])), 2) for tk in toks7])
            self.pos_bel[p_idx] = bel_p[idx]                # (128, 4)
            self.pos_G[p_idx] = fit_affine(bel_p[idx], A_full[:, p_idx, :],
                                           wt7 / wt7.sum())[1:]

    def states_at_patch(self, toks7):
        return self._A7

    def rollout(self, deltas, seqs, bs=16384):
        """deltas added at `site`; seqs (n, 10) tokens.

        deltas (n, d_model): added at position 6 only.
        deltas (n, PATCH_P, d_model): added at positions 0..6 (prefix patch).
        Returns softmax over vocab at the 3 predicting positions: (n, ROLL, vocab).
        """
        n = len(seqs)
        out = np.zeros((n, ROLL, self.vocab))
        dev = self.device
        for s0 in range(0, n, bs):
            x = torch.from_numpy(seqs[s0:s0 + bs]).long().to(dev)
            dl = torch.from_numpy(deltas[s0:s0 + bs]).to(dev, dtype=self.model.cfg.dtype)

            def hook(t, hook):
                t = t.clone()
                if dl.ndim == 3:
                    t[:, :PATCH_P, :] += dl
                else:
                    t[:, PATCH_P - 1, :] += dl
                return t

            with torch.no_grad():
                lg = self.model.run_with_hooks(x, fwd_hooks=[(self.site, hook)])
            pr = F.softmax(lg[:, PATCH_P - 1:PATCH_P - 1 + ROLL, :].float(), dim=-1)
            out[s0:s0 + bs] = pr.cpu().numpy()
        return out


def select_pairs(bel, wt, p0, P_E, P_E_perp):
    N = len(bel)
    ii, jj = np.triu_indices(N, k=1)
    dp0 = np.abs(p0[ii] - p0[jj])
    kern = np.linalg.norm((bel[ii] - bel[jj]) @ P_E_perp, axis=1)
    wmin = np.minimum(wt[ii], wt[jj])
    m = (dp0 < PAIR_TOL) & (wmin > np.quantile(wt, 0.10))
    order = np.argsort(-(kern * m))
    keep = order[m[order]][:N_PAIRS]
    return ii[keep], jj[keep], kern[keep], dp0[keep]
