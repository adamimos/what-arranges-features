"""
Phase 0 feasibility gates for the causal belief-transplant experiment
(qRRXOR, transformer + vanilla RNN).

NOTE: shipped as a LIBRARY for prefix_beliefs(). Its weighted_r2_cv() uses unregularized
lstsq with row-level folds and is NOT used for any number in the paper.

Read-only on existing checkpoints. Gates, in order:

  A. Process machinery — per-prefix beliefs at P in {5,6,7,8}; distinct-belief
     counts; pair mining at the patch position (|Δp0| col(E) match vs ‖δ_ker‖);
     THEORY EFFECT-SIZE CEILING: analytic per-step KL profiles for candidate
     matched pairs (step1 must be ~0, steps 2-3 must be detectably > 0).
  B. Vanilla RNN probes (5 seeds) — pooled-position probe (positions 4..9,
     dedup by prefix, 256-dim h-concat), held-out-position-7 transfer R²,
     single-position-7 probe R².
  C. Transformer probes (5 seeds) — per-site probes at position 6 of the
     7-token prefix (resid_pre L0..3 + resid_post L3, 64-dim each), CV weighted R².
  D. Harness identity checks (seed-42 runs) — RNN stepwise==one-shot,
     capture->continue==full run, swap mechanics; transformer no-op hook == clean,
     position-6 patch leaves positions <6 unchanged.

Usage:
    PYTHONPATH=. uv run python scripts/activation_analysis/causal_phase0.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.activation_analysis.run_shuffle_analysis import (
    load_run_config, list_checkpoints, load_model,
)
from scripts.activation_analysis.enumerate_process import _stationary_belief
from scripts.activation_analysis.shuffle_control import (
    compute_emission_matrix, compute_projections,
)
from scripts.activation_analysis.qrrxor_shuffle_null_analysis import build_rnn
from epsilon_transformers.process.transition_matrices import get_matrix_from_args

RNN_DIR = Path("results/qrrxor_robust_rnn")
XF_DIR = Path("results/qrrxor_robust_transformer")
PATCH_P = 7            # prefix length at the patch position (tokens 0..6)
N_CTX = 10
ROLL_STEPS = N_CTX - PATCH_P   # 3 future predictions (tokens 8,9,10 -> indices 7,8,9)


# ---------------------------------------------------------------------------
# A. Process machinery
# ---------------------------------------------------------------------------

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


def string_probs(T, rev, bel, k):
    """P(u_{1:k} | belief) for all vocab^k strings u, linear in belief.
    Returns (n_beliefs, vocab^k), lex order big-endian over time."""
    vocab = T.shape[0]
    out = np.zeros((bel.shape[0], vocab ** k))
    for u in range(vocab ** k):
        s = u
        tk = np.zeros(k, dtype=np.int64)
        for t in range(k):
            tk[k - 1 - t] = s % vocab
            s //= vocab
        M = np.eye(T.shape[1])
        for x in tk:
            M = M @ T[x]
        out[:, u] = bel @ M @ rev
    return out


def kl(p, q, axis=-1, eps=1e-12):
    p = np.clip(p, eps, None); p = p / p.sum(axis=axis, keepdims=True)
    q = np.clip(q, eps, None); q = q / q.sum(axis=axis, keepdims=True)
    return (p * (np.log(p) - np.log(q))).sum(axis=axis)


def stepwise_theory_kls(T, rev, b_from, b_to, steps):
    """Per-step KL between the belief-conditioned prediction processes started
    at b_from vs b_to, along the b_from-generated tree (exact, enumerated).

    Step s KL = E_{u_{1:s-1} ~ P(.|b_from)} KL( P(x_s | u, b_from) || P(x_s | u, b_to) ).
    """
    vocab = T.shape[0]
    out = []
    # enumerate prefixes of length s-1
    for s in range(1, steps + 1):
        tot = 0.0
        for u in range(vocab ** (s - 1)):
            uu = u
            tk = np.zeros(s - 1, dtype=np.int64)
            for t in range(s - 1):
                tk[s - 2 - t] = uu % vocab
                uu //= vocab
            M = np.eye(T.shape[1])
            for x in tk:
                M = M @ T[x]
            bf = b_from @ M
            bt = b_to @ M
            pu = max(float(bf @ rev), 0.0)          # P(u | b_from)
            if pu <= 1e-14:
                continue
            bf = bf / float(bf @ rev)
            nt = float(bt @ rev)
            if nt <= 1e-14:
                continue
            bt = bt / nt
            pf = np.array([max(float(bf @ T[x] @ rev), 1e-12) for x in range(vocab)])
            pt = np.array([max(float(bt @ T[x] @ rev), 1e-12) for x in range(vocab)])
            pf = pf / pf.sum(); pt = pt / pt.sum()
            tot += pu * float((pf * (np.log(pf) - np.log(pt))).sum())
        out.append(tot)
    return np.array(out)


# ---------------------------------------------------------------------------
# Weighted probe (matches _weighted_r2 conventions; also returns the map)
# ---------------------------------------------------------------------------

def fit_affine(X, Y, w, rcond=1e-10):
    """Weighted affine fit Y ~ [1, X] beta. Returns beta ((D+1) x dy)."""
    w = w / w.sum()
    sw = np.sqrt(w)[:, None]
    Xb = np.hstack([np.ones((len(X), 1)), X])
    beta, *_ = np.linalg.lstsq(Xb * sw, Y * sw, rcond=rcond)
    return beta


def weighted_r2_cv(X, Y, w, n_folds=5, rcond=1e-10, seed=0):
    """CV weighted R², rows pre-shuffled (rows must be unique units)."""
    N = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    X, Y, w = X[perm], Y[perm], w[perm]
    w = w / w.sum()
    fold = N // n_folds
    r2s = []
    for f in range(n_folds):
        te = np.arange(f * fold, (f + 1) * fold if f < n_folds - 1 else N)
        tr = np.setdiff1d(np.arange(N), te)
        beta = fit_affine(X[tr], Y[tr], w[tr], rcond)
        Yp = np.hstack([np.ones((len(te), 1)), X[te]]) @ beta
        wt = w[te] / w[te].sum()
        ym = (Y[te] * wt[:, None]).sum(0)
        ssr = (wt[:, None] * (Y[te] - Yp) ** 2).sum()
        sst = (wt[:, None] * (Y[te] - ym) ** 2).sum()
        r2s.append(1 - ssr / sst if sst > 0 else 0.0)
    return float(np.mean(r2s))


def weighted_r2_eval(beta, X, Y, w):
    Yp = np.hstack([np.ones((len(X), 1)), X]) @ beta
    w = w / w.sum()
    ym = (Y * w[:, None]).sum(0)
    ssr = (w[:, None] * (Y - Yp) ** 2).sum()
    sst = (w[:, None] * (Y - ym) ** 2).sum()
    return float(1 - ssr / sst) if sst > 0 else 0.0


# ---------------------------------------------------------------------------
# RNN state collection
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/causal_phase0.json")
    args = ap.parse_args()
    t0 = time.time()
    dev = args.device
    report = {}

    cfg = load_run_config(str(sorted(RNN_DIR.glob("run_*"))[0]))
    pc = cfg["process_config"].copy(); name = pc.pop("name")
    T = get_matrix_from_args(name=name, **pc)
    P_mat = T.sum(axis=0)
    eigvals, eigvecs = np.linalg.eig(P_mat)
    rev = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))])
    rev = rev / rev.sum()
    b0 = _stationary_belief(T, rev)
    E = compute_emission_matrix(T, rev)
    P_E, P_E_perp = compute_projections(E)
    vocab = T.shape[0]
    print(f"process={name} vocab={vocab} dim={T.shape[1]} rank(E)={np.linalg.matrix_rank(E)} "
          f"dim_ker={T.shape[1]-np.linalg.matrix_rank(E)}")

    # ---------------- A. prefixes, pair mining, theory ceiling ----------------
    print("\n=== A. Process machinery ===")
    for P in (5, 6, 7, 8):
        _, bel, wt, p0 = prefix_beliefs(T, rev, b0, P)
        n_distinct = len(np.unique(np.round(bel, 10), axis=0))
        print(f"  P={P}: prefixes={len(bel)} distinct_beliefs={n_distinct} "
              f"p0 range [{p0.min():.4f},{p0.max():.4f}]")
        if P == PATCH_P:
            report[f"distinct_beliefs_P{P}"] = n_distinct

    toks, bel, wt, p0 = prefix_beliefs(T, rev, b0, PATCH_P)
    N = len(bel)
    # variance split at patch position
    d = bel - bel.mean(0)
    vpar = float(np.var(d @ P_E)); vperp = float(np.var(d @ P_E_perp))
    print(f"  P={PATCH_P} belief-delta variance split: col(E) {vpar:.3e} vs ker(E^T) {vperp:.3e} "
          f"(beyond-NTP frac {vperp/(vpar+vperp):.3f})")
    report["ker_var_frac_P7"] = vperp / (vpar + vperp)

    ii, jj = np.triu_indices(N, k=1)
    dp0 = np.abs(p0[ii] - p0[jj])
    dbel = bel[ii] - bel[jj]
    kern = np.linalg.norm(dbel @ P_E_perp, axis=1)
    coln = np.linalg.norm(dbel @ P_E, axis=1)
    wmin = np.minimum(wt[ii], wt[jj])
    w_thresh = np.quantile(wt, 0.10)
    print(f"  pairs={len(ii)}  |Δp0| quantiles {np.quantile(dp0,[.1,.5,.9]).round(4)}  "
          f"kernorm quantiles {np.quantile(kern,[.1,.5,.9]).round(4)}")
    for tol in (1e-2, 1e-3, 1e-4):
        m = (dp0 < tol) & (wmin > w_thresh)
        m_big = m & (kern > np.median(kern))
        print(f"  matched pairs |Δp0|<{tol:g} (wt>p10): {m.sum()}   ...and kernorm>median: {m_big.sum()}")
        report[f"pairs_tol{tol:g}"] = int(m.sum())
        report[f"pairs_tol{tol:g}_bigker"] = int(m_big.sum())

    # theory effect-size ceiling on top matched pairs
    tol = 1e-3
    m = (dp0 < tol) & (wmin > w_thresh)
    cand = np.argsort(-kern * m)[:200]
    cand = cand[m[cand]]
    prof = []
    for c in cand[:200]:
        a, b = ii[c], jj[c]
        b_chim = bel[a] + (bel[b] - bel[a]) @ P_E_perp     # transplant v's ker into w
        prof.append(stepwise_theory_kls(T, rev, b_chim, bel[a], ROLL_STEPS))
    prof = np.array(prof)   # KL(patched-belief predictions || original-belief predictions)
    q = lambda x: np.quantile(x, [0.25, 0.5, 0.75]).round(4).tolist()
    print(f"  THEORY CEILING over {len(prof)} matched pairs (chimera vs original), per-step KL (nats):")
    for s in range(ROLL_STEPS):
        print(f"    step +{s+1}: quartiles {q(prof[:, s])}  max {prof[:, s].max():.4f}")
    report["theory_kl_median_by_step"] = np.median(prof, axis=0).round(6).tolist()
    # chimera validity: min branch prob over 3-step tree
    sp = string_probs(T, rev, np.array([bel[ii[c]] + (bel[jj[c]] - bel[ii[c]]) @ P_E_perp
                                        for c in cand[:200]]), ROLL_STEPS)
    print(f"  chimera 3-step branch prob min={sp.min():.3e} (negative => off-hull)  "
          f"rows with any branch < -1e-9: {(sp < -1e-9).any(axis=1).sum()}/{len(sp)}")
    report["chimera_min_branch_prob"] = float(sp.min())

    # ---------------- B. RNN probes ----------------
    print(f"\n=== B. Vanilla RNN probes (pooled positions 4..{N_CTX-1}) ===  [{time.time()-t0:.0f}s]")
    # pooled dedup dataset: for p in 4..9, unique prefixes of length p
    pooled = {}
    for p in range(4, N_CTX):
        tk_p, bel_p, wt_p, _ = prefix_beliefs(T, rev, b0, p)
        pooled[p] = (tk_p, bel_p, wt_p)
    for rd in sorted(RNN_DIR.glob("run_*")):
        rcfg = load_run_config(str(rd))
        seed = rcfg["model_config"]["seed"]
        ckpt = list_checkpoints(str(rd))[-1]
        state = torch.load(rd / ckpt, map_location=dev, weights_only=True)
        model = build_rnn(rcfg, vocab, dev, state=state)
        Xs, Ys, Ws, Ps = [], [], [], []
        for p, (tk_p, bel_p, wt_p) in pooled.items():
            S = rnn_states_all_steps(model, tk_p, vocab, dev)[:, -1, :]  # state after p tokens
            Xs.append(S); Ys.append(bel_p); Ws.append(wt_p)
            Ps.append(np.full(len(S), p))
        X = np.concatenate(Xs); Y = np.concatenate(Ys)
        W = np.concatenate(Ws); Pv = np.concatenate(Ps)
        W = W / W.sum()
        r2_pool = weighted_r2_cv(X, Y, W)
        # held-out position transfer: fit without p=7, eval at p=7
        tr = Pv != PATCH_P; te = Pv == PATCH_P
        beta = fit_affine(X[tr], Y[tr], W[tr])
        r2_ho7 = weighted_r2_eval(beta, X[te], Y[te], W[te])
        r2_p7 = weighted_r2_cv(X[te], Y[te], W[te])
        print(f"  [RNN seed {seed}] pooled CV R2={r2_pool:.4f}  held-out-pos7 R2={r2_ho7:.4f}  "
              f"pos7-only CV R2={r2_p7:.4f}   (n_pool={len(X)})")
        report.setdefault("rnn_probe", []).append(
            {"seed": seed, "r2_pooled_cv": r2_pool, "r2_heldout_pos7": r2_ho7, "r2_pos7_cv": r2_p7})
        del model
        torch.cuda.empty_cache() if "cuda" in dev else None

    # ---------------- C. Transformer per-site probes ----------------
    print(f"\n=== C. Transformer per-site probes at position {PATCH_P-1} (7-token prefixes) ===  "
          f"[{time.time()-t0:.0f}s]")
    tk7 = torch.from_numpy(toks).long()
    for rd in sorted(XF_DIR.glob("run_*")):
        rcfg = load_run_config(str(rd))
        seed = rcfg["model_config"]["seed"]
        ckpt = list_checkpoints(str(rd))[-1]
        model = load_model(str(rd), ckpt, dev)
        names = [f"blocks.{i}.hook_resid_pre" for i in range(model.cfg.n_layers)]
        names.append(f"blocks.{model.cfg.n_layers-1}.hook_resid_post")
        with torch.no_grad():
            _, cache = model.run_with_cache(tk7.to(dev), names_filter=names)
        row = {"seed": seed}
        msg = f"  [XF seed {seed}]"
        for nm in names:
            A = cache[nm][:, -1, :].float().cpu().numpy()
            r2 = weighted_r2_cv(A, bel, wt)
            short = nm.replace("blocks.", "L").replace(".hook_resid_", "_")
            row[short] = r2
            msg += f" {short}={r2:.4f}"
        print(msg)
        report.setdefault("xf_probe", []).append(row)
        del model, cache
        torch.cuda.empty_cache() if "cuda" in dev else None

    # ---------------- D. Harness identity checks (seed 42 runs) ----------------
    print(f"\n=== D. Harness identity checks ===  [{time.time()-t0:.0f}s]")
    # RNN
    rd = sorted(RNN_DIR.glob("run_*"))[0]
    rcfg = load_run_config(str(rd))
    state = torch.load(rd / list_checkpoints(str(rd))[-1], map_location=dev, weights_only=True)
    model = build_rnn(rcfg, vocab, dev, state=state)
    seqs10, _, _, _ = prefix_beliefs(T, rev, b0, N_CTX)
    x = torch.from_numpy(seqs10[:256]).long().to(dev)
    oh = torch.nn.functional.one_hot(x, num_classes=vocab).float()
    with torch.no_grad():
        full_out, _ = model.rnn(oh)
        full_logits = model.output_layer(full_out)
        # stepwise with carried state
        h = None; step_logits = []
        for t in range(N_CTX):
            o, h = model.rnn(oh[:, t:t + 1], h)
            step_logits.append(model.output_layer(o))
        step_logits = torch.cat(step_logits, dim=1)
        d1 = (full_logits - step_logits).abs().max().item()
        # capture at P=7, continue
        h = None
        o, h = model.rnn(oh[:, :PATCH_P], None)
        o2, _ = model.rnn(oh[:, PATCH_P:], h)
        cont_logits = model.output_layer(o2)
        d2 = (full_logits[:, PATCH_P:] - cont_logits).abs().max().item()
    print(f"  RNN stepwise==one-shot: max|Δlogit| = {d1:.2e}")
    print(f"  RNN capture@7->continue == full run: max|Δlogit| = {d2:.2e}")
    report["rnn_stepwise_maxdiff"] = d1
    report["rnn_capture_continue_maxdiff"] = d2

    # Transformer
    rd = sorted(XF_DIR.glob("run_*"))[0]
    model = load_model(str(rd), list_checkpoints(str(rd))[-1], dev)
    x10 = torch.from_numpy(seqs10[:256]).long().to(dev)
    with torch.no_grad():
        clean = model(x10)
        zero_hook = lambda t, hook: t
        noop = model.run_with_hooks(x10, fwd_hooks=[("blocks.1.hook_resid_pre", zero_hook)])
        d3 = (clean - noop).abs().max().item()
        delta = torch.randn(model.cfg.d_model, device=dev) * 0.5
        def patch_hook(t, hook):
            t = t.clone(); t[:, PATCH_P - 1, :] += delta; return t
        patched = model.run_with_hooks(x10, fwd_hooks=[("blocks.1.hook_resid_pre", patch_hook)])
        d4_before = (clean[:, :PATCH_P - 1] - patched[:, :PATCH_P - 1]).abs().max().item()
        d4_after = (clean[:, PATCH_P - 1:] - patched[:, PATCH_P - 1:]).abs().max().item()
    print(f"  XF no-op hook == clean: max|Δlogit| = {d3:.2e}")
    print(f"  XF pos-{PATCH_P-1} patch: positions <{PATCH_P-1} max|Δlogit| = {d4_before:.2e} "
          f"(must be 0);  positions >= {PATCH_P-1}: {d4_after:.3f} (must be > 0)")
    report["xf_noop_maxdiff"] = d3
    report["xf_patch_upstream_leak"] = d4_before
    report["xf_patch_downstream_effect"] = d4_after

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nSaved {args.out}   total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
