"""
Causal belief-transplant experiment (qRRXOR).

NOTE: this module is shipped as a LIBRARY (RNNBackend, transformer backend, probe helpers)
for causal_ntp_neutral.py / causal_xf_channels.py / causal_local_readout.py / the erasure
audits. Its own main() is an earlier transplant design whose numbers the paper does NOT
quote; the paper's patching numbers come from causal_ntp_neutral.py.

qRRXOR, patch at prefix length 7 (position index 6), track predictions for
tokens 8,9,10. Transplant the next-token-INVISIBLE ker(E^T) component of the
belief difference between matched history pairs into the network state, via the
pseudo-inverse of an affine read probe.

Per (arch, seed[, site]) this script reports, over ~198 matched pairs x 2
directions:

  patch types
    ker      Δa = α · Δη_ker @ G, G the belief->state DECODER (primary;
             α ∈ {0,.25,.5,.75,1}; variance-bearing representation directions)
    chord    Δa = α · proj_span(G_ker)(a_v − a_w)  (on-manifold variant)
    pinv     Δa = Δη_ker @ pinv(W_encoder)         (min-norm probe-preimage write;
             secondary — expected inert: ~0.4% of state norm, high-gain read dirs)
    resid_swap Δa = (a_v − a_w) − Δη_full @ G      (swap everything EXCEPT the
             belief component; belief-causality predicts ~no targeted movement)
    random   matched-norm uniform directions       (10 draws, α=1)
    null     matched-norm directions with Δa@W=0   (10 draws, α=1; probe-invisible)
    fullswap replace state/residual entirely with v's (positive control / routing ceiling)

  metrics per patch
    step-wise KL(patched || unpatched) at steps +1,+2,+3   (profile: flat then rising)
    3-step tree KL(Q_patched || theory(η_α)), || theory(η_w)), || Q_unpatched)
    KL(Q_patched || Q_model(v)) — behavioral target (model's own dist given v)
    Δlog-conditional regression: observed log Q_p(u) − log Q_u(u) vs theory
    ‖Δa‖ / ‖a‖

Theory targets are exact (linear belief dynamics): η_α = η_w + α·(η_v−η_w)@P_E_perp,
P(u_{1:k}|η) = η T_{u1}..T_{uk} rev.

Usage (one run):
  PYTHONPATH=. uv run python scripts/activation_analysis/causal_belief_transplant.py \
      --arch rnn --run_dir results/qrrxor_robust_rnn/run_0_... --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from scripts.activation_analysis.run_shuffle_analysis import (
    load_run_config, list_checkpoints, load_model,
)
from scripts.activation_analysis.enumerate_process import _stationary_belief
from scripts.activation_analysis.shuffle_control import (
    compute_emission_matrix, compute_projections,
)
from scripts.activation_analysis.qrrxor_shuffle_null_analysis import build_rnn
from scripts.activation_analysis.causal_phase0 import (
    prefix_beliefs, fit_affine, weighted_r2_eval, rnn_states_all_steps,
)
from epsilon_transformers.process.transition_matrices import get_matrix_from_args

PATCH_P = 7
N_CTX = 10
ROLL = N_CTX - PATCH_P            # 3 predicted tokens
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_CONTROL_DRAWS = 10
PAIR_TOL = 1e-3
N_PAIRS = 200
XF_SITES = ("blocks.1.hook_resid_pre", "blocks.2.hook_resid_pre", "blocks.3.hook_resid_pre")
EPS = 1e-12


# ---------------------------------------------------------------------------
# process helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# model rollout backends
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

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


def xf_step_probs_from_softmax(soft, branches):
    """(n_states_runs, ROLL, vocab) softmax -> step probs of branch tokens.
    Caller arranges one run per (state, branch)."""
    n = len(soft)
    out = np.zeros((n, ROLL))
    for s in range(ROLL):
        out[:, s] = soft[np.arange(n), s, branches[:, s]]
    return out


def run_experiment(arch, run_dir, device, site=None, out_dir="results/causal_transplant"):
    t0 = time.time()
    cfg = load_run_config(str(run_dir))
    pc = cfg["process_config"].copy(); name = pc.pop("name")
    T = get_matrix_from_args(name=name, **pc)
    P_mat = T.sum(axis=0)
    eigvals, eigvecs = np.linalg.eig(P_mat)
    rev = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))]); rev /= rev.sum()
    b0 = _stationary_belief(T, rev)
    E = compute_emission_matrix(T, rev)
    P_E, P_E_perp = compute_projections(E)
    vocab = T.shape[0]

    toks7, bel7, wt7, p07 = prefix_beliefs(T, rev, b0, PATCH_P)
    branches = branch_tokens(ROLL, vocab)
    pi, pj, kern, dp0 = select_pairs(bel7, wt7, p07, P_E, P_E_perp)
    # both directions: transplant j's ker into i, and i's ker into j
    src = np.concatenate([pi, pj])     # history whose state we patch  (w)
    dst = np.concatenate([pj, pi])     # history donating the ker component (v)
    npairs = len(src)

    if arch == "rnn":
        be = RNNBackend(run_dir, device, T, rev, b0, vocab)
        arch_label = cfg["model_config"]["rnn_type"].upper()   # RNN / LSTM / GRU
    else:
        be = XFBackend(run_dir, device, T, rev, b0, vocab, site)
        arch_label = "transformer"
    A = be.states_at_patch(toks7)                     # (128, d)
    d_state = A.shape[1]
    probe_pred = np.hstack([np.ones((len(A), 1)), A]) @ be.beta
    probe_r2 = weighted_r2_eval(be.beta, A, bel7, wt7)

    # belief-space quantities
    d_eta_full = bel7[dst] - bel7[src]
    d_eta_ker = d_eta_full @ P_E_perp                 # (npairs, 4)
    eta_alpha = {a: bel7[src] + a * d_eta_ker for a in ALPHAS}

    # activation-space writes — PRIMARY: decoder (belief->state) directions
    d_ker_act = d_eta_ker @ be.G                      # (npairs, d)
    d_pinv_act = d_eta_ker @ be.pinvW                 # secondary: min-norm probe preimage
    d_resid_swap = (A[dst] - A[src]) - d_eta_full @ be.G   # swap minus belief component
    # chord: project actual activation difference onto decoder ker-image subspace
    Bk = np.linalg.svd(P_E_perp)[0][:, :2]            # basis of ker(E^T)
    D = (Bk.T @ be.G)                                 # (2, d) ker-image directions
    P_act = np.linalg.pinv(D) @ D                     # (d, d) projector onto span
    d_chord_act = (A[dst] - A[src]) @ P_act.T

    # readback: encoder applied to the decoder write should return Δη_ker
    rb = d_ker_act @ be.Wmat
    readback_err = float(np.median(np.linalg.norm(rb - d_eta_ker, axis=1) /
                                   np.maximum(np.linalg.norm(d_eta_ker, axis=1), EPS)))

    norms_a = np.linalg.norm(A[src], axis=1)
    norms_d = np.linalg.norm(d_ker_act, axis=1)

    rng = np.random.default_rng(0)
    # matched-norm random directions
    rand_dirs = rng.standard_normal((N_CONTROL_DRAWS, npairs, d_state))
    rand_dirs /= np.linalg.norm(rand_dirs, axis=2, keepdims=True)
    rand_dirs *= norms_d[None, :, None]
    # null(W) directions: project random out of row-space of W^T (Δa @ W = 0)
    Uw = np.linalg.svd(be.Wmat, full_matrices=False)[0]   # (d, 4)
    null_dirs = rng.standard_normal((N_CONTROL_DRAWS, npairs, d_state))
    null_dirs -= np.einsum("kpd,df->kpf", np.einsum("kpd,df->kpf", null_dirs, Uw), Uw.T)
    null_dirs /= np.maximum(np.linalg.norm(null_dirs, axis=2, keepdims=True), EPS)
    null_dirs *= norms_d[None, :, None]

    # ---- build the patch batch ----
    jobs = []   # (kind, alpha, draw, delta (npairs, d))
    for a in ALPHAS:
        jobs.append(("ker", a, 0, a * d_ker_act))
        if a > 0:
            jobs.append(("chord", a, 0, a * d_chord_act))
    jobs.append(("pinv", 1.0, 0, d_pinv_act))
    jobs.append(("resid_swap", 1.0, 0, d_resid_swap))
    for k in range(N_CONTROL_DRAWS):
        jobs.append(("random", 1.0, k, rand_dirs[k]))
        jobs.append(("null", 1.0, k, null_dirs[k]))
    jobs.append(("fullswap", 1.0, 0, A[dst] - A[src]))
    if arch == "transformer":
        # routing controls: patch the whole prefix (positions 0..6), not one position
        jobs.append(("prefix_swap", 1.0, 0, be._A_full[dst] - be._A_full[src]))
        d_pb = np.zeros_like(be._A_full[src])
        for p_idx, Gp in be.pos_G.items():
            d_eta_p = be.pos_bel[p_idx][dst] - be.pos_bel[p_idx][src]
            d_pb[:, p_idx, :] = d_eta_p @ Gp
        jobs.append(("prefix_bel", 1.0, 0, d_pb))

    # ---- model rollouts ----
    nb = len(branches)
    if arch == "rnn":
        def get_step_probs(delta):
            return be.rollout(A[src] + delta, branches)          # (npairs, nb, ROLL)
        base_probs = be.rollout(A, branches)                     # all 128 prefixes
        Qu_steps = base_probs[src]                               # unpatched w
        Qv_steps = base_probs[dst]                               # model given v
    else:
        # one run per (pair, branch): seq = prefix(src) + branch
        seqs = np.concatenate(
            [np.repeat(toks7[src], nb, axis=0),
             np.tile(branches, (npairs, 1))], axis=1)
        def get_step_probs(delta):
            soft = be.rollout(np.repeat(delta, nb, axis=0), seqs)
            sp = xf_step_probs_from_softmax(soft, np.tile(branches, (npairs, 1)))
            return sp.reshape(npairs, nb, ROLL)
        z = np.zeros((npairs, d_state))
        Qu_steps = get_step_probs(z)
        seqs_v = np.concatenate(
            [np.repeat(toks7[dst], nb, axis=0),
             np.tile(branches, (npairs, 1))], axis=1)
        soft_v = be.rollout(np.zeros((npairs * nb, d_state)), seqs_v)
        Qv_steps = xf_step_probs_from_softmax(
            soft_v, np.tile(branches, (npairs, 1))).reshape(npairs, nb, ROLL)

    Qu_tree = np.prod(Qu_steps, axis=2)
    Qv_tree = np.prod(Qv_steps, axis=2)
    th_w = theory_tree(T, rev, bel7[src], branches)
    th_v = theory_tree(T, rev, bel7[dst], branches)

    rows = []
    detail = {"pairs": {"src": src, "dst": dst, "kern": kern.tolist() * 2,
                        "dp0": dp0.tolist() * 2},
              "probe_r2": probe_r2, "site": site, "arch": arch_label, "seed": be.seed}
    for kind, a, draw, delta in jobs:
        sp = get_step_probs(delta)                    # (npairs, nb, ROLL)
        Qp_tree = np.prod(sp, axis=2)
        # theory target per kind: ker/chord/pinv aim at η_α; fullswap at η_v;
        # random/null/resid_swap predict NO belief change (target η_w)
        if kind in ("ker", "chord"):
            th_a = theory_tree(T, rev, eta_alpha[a], branches)
        elif kind == "pinv":
            th_a = theory_tree(T, rev, eta_alpha[1.0], branches)
        elif kind in ("fullswap", "prefix_swap", "prefix_bel"):
            th_a = th_v
        else:
            th_a = th_w
        klsteps = stepwise_kls_from_probs(sp, Qu_steps, branches)
        # Δlog regression vs theory (ker/chord only)
        obs = np.log(np.clip(Qp_tree, EPS, None)) - np.log(np.clip(Qu_tree, EPS, None))
        pred = np.log(np.clip(th_a, EPS, None)) - np.log(np.clip(th_w, EPS, None))
        ow, pw = obs.ravel(), pred.ravel()
        if np.var(pw) > 0:
            slope = float(np.cov(ow, pw)[0, 1] / np.var(pw))
            r2 = float(np.corrcoef(ow, pw)[0, 1] ** 2)
        else:
            slope, r2 = float("nan"), float("nan")
        rows.append({
            "arch": arch_label, "seed": be.seed, "site": site or "", "kind": kind,
            "alpha": a, "draw": draw,
            "kl_step1": float(np.median(klsteps[:, 0])),
            "kl_step2": float(np.median(klsteps[:, 1])),
            "kl_step3": float(np.median(klsteps[:, 2])),
            "kl_tree_to_target": float(np.median(tree_kl(Qp_tree, th_a))),
            "kl_tree_to_orig_theory": float(np.median(tree_kl(Qp_tree, th_w))),
            "kl_unpatched_to_target": float(np.median(tree_kl(Qu_tree, th_a))),
            "kl_to_model_v": float(np.median(tree_kl(Qp_tree, Qv_tree))),
            "kl_unpatched_to_model_v": float(np.median(tree_kl(Qu_tree, Qv_tree))),
            "dlog_slope": slope, "dlog_r2": r2,
            "norm_ratio_median": float(np.median(
                np.linalg.norm(delta.reshape(len(delta), -1), axis=1) /
                np.maximum(norms_a, EPS))),
            "probe_r2": probe_r2, "readback_err": readback_err, "n_pairs": npairs,
        })
        print(f"  [{arch_label}{'/' + site if site else ''} s{be.seed}] {kind:8s} a={a:<4} d={draw} "
              f"kl1={rows[-1]['kl_step1']:.4f} kl2={rows[-1]['kl_step2']:.4f} "
              f"kl3={rows[-1]['kl_step3']:.4f} "
              f"to_tgt={rows[-1]['kl_tree_to_target']:.4f} "
              f"unp_to_tgt={rows[-1]['kl_unpatched_to_target']:.4f} "
              f"R2={rows[-1]['dlog_r2']:.3f} slope={rows[-1]['dlog_slope']:.3f}", flush=True)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f"{arch_label.lower()}_s{be.seed}" + (f"_{site.replace('.', '_')}" if site else "")
    import csv as _csv
    with open(out / f"{tag}.csv", "w", newline="") as fh:
        wcsv = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows)
    joblib.dump({"rows": rows, "detail": detail}, out / f"{tag}.joblib")
    print(f"  saved {out}/{tag}.csv   ({time.time()-t0:.1f}s)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["rnn", "transformer"], required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sites", default=",".join(XF_SITES))
    ap.add_argument("--out_dir", default="results/causal_transplant")
    args = ap.parse_args()
    if args.arch == "rnn":
        run_experiment("rnn", args.run_dir, args.device, out_dir=args.out_dir)
    else:
        for site in args.sites.split(","):
            run_experiment("transformer", args.run_dir, args.device, site=site,
                           out_dir=args.out_dir)


if __name__ == "__main__":
    main()
