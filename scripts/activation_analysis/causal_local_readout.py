"""
Local-readout causal test (companion to causal_belief_transplant.py).

Question: are the probe's belief coordinates causally read for the LOCAL
next-token prediction at the patched position? Uses ANTI-matched pairs
(large |Δp0|) so the col(E) component of the belief difference is large.

Writes at position 7 (index 6), read POSITION-6 LOGITS ONLY:
    col     Δa = Δη_col @ G   -> local dist should move to NTP(η_v) exactly
                                  (η_w + Δη_col has the same E-readout as η_v)
    full    Δa = Δη_full @ G  -> same local target NTP(η_v)
    ker     Δa = Δη_ker @ G   -> local dist should NOT move (contrast)
    random  matched-norm to col write (10 draws) -> no targeted movement

Metrics: KL(patched||unpatched); targeting fraction toward NTP(η_v); binary
log-odds regression observed Δ vs theory Δ (col/full only).

Usage:
  PYTHONPATH=. uv run python scripts/activation_analysis/causal_local_readout.py --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.activation_analysis.run_shuffle_analysis import load_run_config
from scripts.activation_analysis.enumerate_process import _stationary_belief
from scripts.activation_analysis.shuffle_control import (
    compute_emission_matrix, compute_projections,
)
from scripts.activation_analysis.causal_phase0 import prefix_beliefs
from scripts.activation_analysis.causal_belief_transplant import (
    RNNBackend, XFBackend, PATCH_P, XF_SITES, EPS, branch_tokens,
)
from epsilon_transformers.process.transition_matrices import get_matrix_from_args

N_PAIRS = 200
N_DRAWS = 10

GROUPS = [
    ("rnn", "results/qrrxor_robust_rnn", None),
    ("rnn", "results/qrrxor_robust_lstm", None),
    ("rnn", "results/qrrxor_robust_gru", None),
    ("transformer", "results/qrrxor_robust_transformer", XF_SITES),
]


def kl2(p, q):
    p = np.clip(np.stack([p, 1 - p], -1), EPS, None)
    q = np.clip(np.stack([q, 1 - q], -1), EPS, None)
    p /= p.sum(-1, keepdims=True); q /= q.sum(-1, keepdims=True)
    return (p * (np.log(p) - np.log(q))).sum(-1)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def local_p0(arch, be, states_or_delta, toks7, src, branches):
    """p(token=0) at position 6 under the patch."""
    if arch == "rnn":
        sp = be.rollout(states_or_delta, branches[:1])      # only need step 0
        # step 0 prob of branch token branches[0,0]; recover p0
        p_tok = sp[:, 0, 0]
        return p_tok if branches[0, 0] == 0 else 1 - p_tok
    else:
        nb1 = 1
        seqs = np.concatenate([toks7[src], np.tile(branches[0], (len(src), 1))], axis=1)
        soft = be.rollout(states_or_delta, seqs)
        return soft[:, 0, 0]                                # P(token 0) at position 6


def run_group(arch, dpath, sites, device, out_rows):
    cfg = load_run_config(str(sorted(Path(dpath).glob("run_*"))[0]))
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
    branches = branch_tokens(3, vocab)

    # ANTI-matched pairs: large |Δp0|, weight-filtered
    N = len(bel7)
    ii, jj = np.triu_indices(N, k=1)
    dp0 = np.abs(p07[ii] - p07[jj])
    wmin = np.minimum(wt7[ii], wt7[jj])
    m = wmin > np.quantile(wt7, 0.10)
    order = np.argsort(-(dp0 * m))
    keep = order[m[order]][:N_PAIRS]
    src = np.concatenate([ii[keep], jj[keep]])
    dst = np.concatenate([jj[keep], ii[keep]])
    npairs = len(src)

    for rd in sorted(Path(dpath).glob("run_*")):
        for site in (sites or [None]):
            if arch == "rnn":
                be = RNNBackend(rd, device, T, rev, b0, vocab)
                label = load_run_config(str(rd))["model_config"]["rnn_type"].upper()
            else:
                be = XFBackend(rd, device, T, rev, b0, vocab, site)
                label = "transformer"
            A = be.states_at_patch(toks7)
            d_eta = bel7[dst] - bel7[src]
            writes = {
                "col": (d_eta @ P_E) @ be.G,
                "full": d_eta @ be.G,
                "ker": (d_eta @ P_E_perp) @ be.G,
            }
            norms = np.linalg.norm(writes["col"], axis=1)
            rng = np.random.default_rng(0)

            # theory local targets
            ntp_w = p07[src]
            ntp_v = p07[dst]

            def eval_delta(delta):
                if arch == "rnn":
                    return local_p0(arch, be, A[src] + delta, toks7, src, branches)
                return local_p0(arch, be, delta, toks7, src, branches)

            p_unp = eval_delta(np.zeros_like(writes["col"]))
            for kind, delta in writes.items():
                p_pat = eval_delta(delta)
                tgt = ntp_v if kind in ("col", "full") else ntp_w
                klpu = kl2(p_pat, p_unp)
                tgt_frac = 1 - np.median(kl2(p_pat, ntp_v)) / max(
                    np.median(kl2(p_unp, ntp_v)), EPS)
                obs = logit(p_pat) - logit(p_unp)
                pred = logit(tgt) - logit(ntp_w)
                if np.var(pred) > 0:
                    slope = float(np.cov(obs, pred)[0, 1] / np.var(pred))
                    r2 = float(np.corrcoef(obs, pred)[0, 1] ** 2)
                else:
                    slope, r2 = float("nan"), float("nan")
                out_rows.append({
                    "arch": label, "site": site or "", "seed": be.seed, "kind": kind,
                    "kl_local": float(np.median(klpu)),
                    "tgt_frac": float(tgt_frac),
                    "slope": slope, "r2": r2,
                    "mean_abs_obs": float(np.mean(np.abs(obs))),
                    "norm_ratio": float(np.median(np.linalg.norm(delta, axis=1) /
                                                  np.linalg.norm(A[src], axis=1))),
                })
            # matched-norm random (to col write)
            kls, tf, mao = [], [], []
            for k in range(N_DRAWS):
                rd_dir = rng.standard_normal((npairs, A.shape[1]))
                rd_dir /= np.linalg.norm(rd_dir, axis=1, keepdims=True)
                rd_dir *= norms[:, None]
                p_pat = eval_delta(rd_dir)
                kls.append(np.median(kl2(p_pat, p_unp)))
                tf.append(1 - np.median(kl2(p_pat, ntp_v)) / max(
                    np.median(kl2(p_unp, ntp_v)), EPS))
                mao.append(np.mean(np.abs(logit(p_pat) - logit(p_unp))))
            out_rows.append({
                "arch": label, "site": site or "", "seed": be.seed, "kind": "random",
                "kl_local": float(np.mean(kls)), "tgt_frac": float(np.mean(tf)),
                "slope": float("nan"), "r2": float("nan"),
                "mean_abs_obs": float(np.mean(mao)),
                "norm_ratio": out_rows[-3]["norm_ratio"],
            })
            r = out_rows[-4:]
            print(f"  [{label}{'/' + site if site else ''} s{be.seed}] "
                  + "  ".join(f"{x['kind']}: kl={x['kl_local']:.4f} tgt%={100 * x['tgt_frac']:.0f} "
                              f"R2={x['r2']:.2f}" for x in r), flush=True)
            del be
            if "cuda" in str(device):
                torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/causal_transplant/local_readout.csv")
    args = ap.parse_args()
    t0 = time.time()
    rows = []
    for arch, dpath, sites in GROUPS:
        print(f"=== {dpath} ===")
        run_group(arch, dpath, sites, args.device, rows)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s)")

    # aggregate
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["arch"], r["site"], r["kind"])].append(r)
    print("\nAGGREGATE (mean over seeds):")
    print(f"{'arch':<12} {'site':<24} {'kind':<7} {'kl_local':>9} {'tgt%':>7} {'R2':>6} {'slope':>6}")
    for (a, s, k), rs in agg.items():
        kl = np.mean([r["kl_local"] for r in rs])
        tf = 100 * np.mean([r["tgt_frac"] for r in rs])
        r2 = np.nanmean([r["r2"] for r in rs])
        sl = np.nanmean([r["slope"] for r in rs])
        print(f"{a:<12} {s:<24} {k:<7} {kl:>9.4f} {tf:>7.1f} {r2:>6.2f} {sl:>6.2f}")


if __name__ == "__main__":
    main()
