"""
"Everything except next-token space" transplant (recurrent architectures).

Generalizes the 2-dim ker(Eᵀ) regression write of an earlier belief-transplant design
(not shipped; its patching backends are in epsilon_transformers/analysis/interventions.py):
instead of writing a probe-derived vector, transplant the network's OWN full
state difference (a_v − a_w) with only the next-token-visible direction removed.

For RNN/GRU/LSTM the local next-token logit is EXACTLY linear in the top-layer h
(output_layer is a Linear on h), so the NTP direction u = W_out[0] − W_out[1]
(embedded in the top-layer h block of the concatenated state) spans exactly the
locally-visible subspace. Projecting it out makes the patch next-token-invisible
BY CONSTRUCTION for EVERY history pair — no NTP-matched-pair selection needed, so
all vocab^7 choose 2 = 8128 pairs (x2 directions) are usable.

Belief-space accounting: with η·rev = 1 fixed, col(E) contributes exactly ONE
free direction (p0) and ker(Eᵀ) contributes TWO, so removing u transplants the
entire beyond-NTP belief content. Theory target is therefore the same chimera as
the regression write: η_chi = η_w + (η_v − η_w) P_perp.

patch kinds
  ntp_neutral     Δa = (a_v − a_w) − proj_u(a_v − a_w)     [PRIMARY, exact-invisible]
  fullswap        Δa = (a_v − a_w)                          [ceiling; step-1 moves]
  ker_write       Δa = Δη_ker @ G                           [the regression method]
  rand_neutral    matched-norm random, projected off u       [control]

Usage:
  PYTHONPATH=. uv run python scripts/activation_analysis/causal_ntp_neutral.py \
      --run_dir results/qrrxor_robust_lstm/run_0_... --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from epsilon_transformers.analysis.checkpoints import load_run_config
from epsilon_transformers.analysis.process_targets import _stationary_belief
from epsilon_transformers.analysis.shuffle_control import (
    compute_emission_matrix, compute_projections,
)
from epsilon_transformers.analysis.interventions import (
    prefix_beliefs, RNNBackend, branch_tokens, theory_tree, tree_kl, stepwise_kls_from_probs,
    PATCH_P, ROLL, EPS,
)
from epsilon_transformers.process.transition_matrices import get_matrix_from_args

N_RAND = 5
MAX_PAIRS = 8128          # all pairs of the 128 length-7 prefixes


def run_one(run_dir, device, out_dir, matched_only=False):
    t0 = time.time()
    cfg = load_run_config(str(run_dir))
    seed = cfg["model_config"]["seed"]
    arch = cfg["model_config"]["rnn_type"].upper()
    pc = cfg["process_config"].copy(); name = pc.pop("name")
    T = get_matrix_from_args(name=name, **pc)
    P_mat = T.sum(axis=0)
    ev, evec = np.linalg.eig(P_mat)
    rev = np.real(evec[:, np.argmin(np.abs(ev - 1.0))]); rev /= rev.sum()
    b0 = _stationary_belief(T, rev)
    E = compute_emission_matrix(T, rev)
    P_E, P_E_perp = compute_projections(E)
    vocab = T.shape[0]
    toks7, bel7, wt7, p07 = prefix_beliefs(T, rev, b0, PATCH_P)
    branches = branch_tokens(ROLL, vocab)

    be = RNNBackend(run_dir, device, T, rev, b0, vocab)
    A = be.states_at_patch(toks7)

    # exact next-token direction: local logit diff is linear in top-layer h
    Wout = be.model.output_layer.weight.detach().float().cpu().numpy()   # (vocab, H)
    u = np.zeros(A.shape[1])
    u[(be.L - 1) * be.H: be.L * be.H] = Wout[0] - Wout[1]
    u = u / np.linalg.norm(u)

    ii, jj = np.triu_indices(len(A), k=1)
    if matched_only:
        keep = np.abs(p07[ii] - p07[jj]) < 1e-3
        ii, jj = ii[keep], jj[keep]
    src = np.concatenate([ii, jj]); dst = np.concatenate([jj, ii])
    npairs = len(src)

    d_full = A[dst] - A[src]
    d_neutral = d_full - np.outer(d_full @ u, u)
    d_ker = ((bel7[dst] - bel7[src]) @ P_E_perp) @ be.G
    rng = np.random.default_rng(0)

    # theory: chimera keeps w's next-token part, takes v's beyond-NTP part
    eta_chi = bel7[src] + (bel7[dst] - bel7[src]) @ P_E_perp
    th_chi = theory_tree(T, rev, eta_chi, branches)
    th_w = theory_tree(T, rev, bel7[src], branches)
    th_v = theory_tree(T, rev, bel7[dst], branches)

    base = be.rollout(A, branches)
    Qu_steps = base[src]
    Qu_tree = np.prod(Qu_steps, 2)
    Qv_tree = np.prod(base[dst], 2)

    jobs = [("ntp_neutral", d_neutral), ("fullswap", d_full), ("ker_write", d_ker)]
    for k in range(N_RAND):
        r = rng.standard_normal(d_neutral.shape)
        r -= np.outer(r @ u, u)
        r *= (np.linalg.norm(d_neutral, axis=1) /
              np.maximum(np.linalg.norm(r, axis=1), EPS))[:, None]
        jobs.append((f"rand_neutral_{k}", r))

    rows = []
    for kind, delta in jobs:
        sp = be.rollout(A[src] + delta, branches)
        Qp = np.prod(sp, 2)
        kls = stepwise_kls_from_probs(sp, Qu_steps, branches)
        target = th_v if kind == "fullswap" else th_chi
        obs = np.log(np.clip(Qp, EPS, None)) - np.log(np.clip(Qu_tree, EPS, None))
        pred = np.log(np.clip(target, EPS, None)) - np.log(np.clip(th_w, EPS, None))
        r2 = float(np.corrcoef(obs.ravel(), pred.ravel())[0, 1] ** 2)
        slope = float(np.cov(obs.ravel(), pred.ravel())[0, 1] / np.var(pred.ravel()))
        tgt_frac = 1 - np.median(tree_kl(Qp, target)) / max(
            np.median(tree_kl(Qu_tree, target)), EPS)
        rows.append({
            "arch": arch, "seed": seed, "kind": kind, "n_pairs": npairs,
            "matched_only": matched_only,
            "kl_step1": float(np.median(kls[:, 0])),
            "kl_step2": float(np.median(kls[:, 1])),
            "kl_step3": float(np.median(kls[:, 2])),
            "kl_step1_max": float(np.max(kls[:, 0])),
            "tgt_frac": float(tgt_frac), "dlog_r2": r2, "dlog_slope": slope,
            "norm_ratio": float(np.median(np.linalg.norm(delta, axis=1) /
                                          np.linalg.norm(A[src], axis=1))),
        })
        if not kind.startswith("rand_neutral_") or kind.endswith("_0"):
            print(f"  [{arch} s{seed}] {kind:14s} kl1={rows[-1]['kl_step1']:.6f} "
                  f"(max {rows[-1]['kl_step1_max']:.2e}) kl2={rows[-1]['kl_step2']:.4f} "
                  f"kl3={rows[-1]['kl_step3']:.4f} tgt%={100*tgt_frac:.1f} "
                  f"R2={r2:.3f} slope={slope:.3f}", flush=True)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    f = out / f"ntp_neutral_{arch.lower()}_s{seed}.csv"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  saved {f}  ({npairs} pairs, {time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default="results/causal_transplant")
    ap.add_argument("--matched_only", action="store_true")
    args = ap.parse_args()
    run_one(args.run_dir, args.device, args.out_dir, args.matched_only)


if __name__ == "__main__":
    main()
