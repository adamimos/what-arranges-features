"""
Transformer channel-attribution patching (full activation-space swaps, no
regression writes) — follow-up to the belief-transplant patching experiments.

Two channels carry w-vs-v prefix information to the branch predictions:
  C1 "prefix_upper":  prefix positions 0..6, resid_pre L1 <- donor run
                      (cascades: the prefix's L2/L3 streams recompute to the
                      donor's exact values, so this IS "everything above L0"
                      for the prefix)
  C2 "branch_l0attn": branch positions 7..9, blocks.0.hook_attn_out <- donor
                      run (severs the branches' only read of raw prefix
                      embeddings)
Plus:
  "allpos6_swap":     position 6 only, resid_pre L1+L2+L3 and resid_post L3
                      <- donor exact values (the strongest possible
                      single-position patch that leaves layer 0 alone)
  "both":             C1 + C2 — must reconstruct the donor run (identity check)

PRE-REGISTERED (2026-07-26, before running): allpos6 ~ null; prefix_upper ~ 18%
(replicates prefix_swap@L1); branch_l0attn carries MOST of the effect if
downstream positions re-derive beliefs from raw tokens; both ~ 100%.

Usage:
  PYTHONPATH=. uv run python scripts/activation_analysis/causal_xf_channels.py \
      --run_dir results/qrrxor_robust_transformer/run_0_... --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from epsilon_transformers.analysis.checkpoints import (
    load_run_config, list_checkpoints, load_model,
)
from epsilon_transformers.analysis.process_targets import _stationary_belief
from epsilon_transformers.analysis.shuffle_control import (
    compute_emission_matrix, compute_projections,
)
from epsilon_transformers.analysis.interventions import (
    prefix_beliefs, branch_tokens, theory_tree, tree_kl, stepwise_kls_from_probs,
    select_pairs, PATCH_P, ROLL, EPS,
)
from epsilon_transformers.process.transition_matrices import get_matrix_from_args

HOOKS_UPPER = ["blocks.1.hook_resid_pre", "blocks.2.hook_resid_pre",
               "blocks.3.hook_resid_pre", "blocks.3.hook_resid_post"]
HOOK_L0ATTN = "blocks.0.hook_attn_out"
ALL_HOOKS = HOOKS_UPPER + [HOOK_L0ATTN]

# kind -> list of (hook_name, position_slice)
KINDS = {
    "allpos6_swap": [(h, slice(PATCH_P - 1, PATCH_P)) for h in HOOKS_UPPER],
    "prefix_upper": [("blocks.1.hook_resid_pre", slice(0, PATCH_P))],
    "branch_l0attn": [(HOOK_L0ATTN, slice(PATCH_P, None))],
    "both": [("blocks.1.hook_resid_pre", slice(0, PATCH_P)),
             (HOOK_L0ATTN, slice(PATCH_P, None))],
}


def run_one(run_dir, device, out_dir):
    t0 = time.time()
    cfg = load_run_config(str(run_dir))
    seed = cfg["model_config"]["seed"]
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
    nb = len(branches)
    pi, pj, kern, dp0 = select_pairs(bel7, wt7, p07, P_E, P_E_perp)
    src = np.concatenate([pi, pj]); dst = np.concatenate([pj, pi])
    npairs = len(src)

    model = load_model(str(run_dir), list_checkpoints(str(run_dir))[-1], device)
    dt = model.cfg.dtype

    # ---- cache pass: all 128 prefixes x 8 branches ----
    seqs_all = np.concatenate(
        [np.repeat(toks7, nb, axis=0), np.tile(branches, (len(toks7), 1))], axis=1)
    cache_np, soft_np = {h: [] for h in ALL_HOOKS}, []
    with torch.no_grad():
        for s0 in range(0, len(seqs_all), 4096):
            x = torch.from_numpy(seqs_all[s0:s0 + 4096]).long().to(device)
            lg, cache = model.run_with_cache(x, names_filter=ALL_HOOKS)
            for h in ALL_HOOKS:
                cache_np[h].append(cache[h].float().cpu().numpy())
            soft_np.append(F.softmax(
                lg[:, PATCH_P - 1:PATCH_P - 1 + ROLL, :].float(), -1).cpu().numpy())
    cache_np = {h: np.concatenate(v) for h, v in cache_np.items()}
    soft_np = np.concatenate(soft_np)                       # (1024, ROLL, vocab)

    def seq_ids(prefix_idx):                                # (npairs*nb,) cache rows
        return (np.repeat(prefix_idx, nb) * nb + np.tile(np.arange(nb), npairs))

    rows_src, rows_dst = seq_ids(src), seq_ids(dst)
    br_all = np.tile(branches, (npairs, 1))

    def steps_from_soft(soft_rows):
        sp = np.zeros((len(soft_rows), ROLL))
        for s in range(ROLL):
            sp[:, s] = soft_rows[np.arange(len(soft_rows)), s, br_all[:, s]]
        return sp.reshape(npairs, nb, ROLL)

    Qu_steps = steps_from_soft(soft_np[rows_src])
    Qv_steps = steps_from_soft(soft_np[rows_dst])
    Qu_tree, Qv_tree = np.prod(Qu_steps, 2), np.prod(Qv_steps, 2)
    th_v = theory_tree(T, rev, bel7[dst], branches)
    th_w = theory_tree(T, rev, bel7[src], branches)

    # ---- patched passes ----
    x_in = torch.from_numpy(seqs_all[rows_src]).long()
    out_rows = []
    for kind, spec in KINDS.items():
        soft_p = np.zeros((npairs * nb, ROLL, vocab))
        for s0 in range(0, npairs * nb, 4096):
            sl = slice(s0, min(s0 + 4096, npairs * nb))
            patches = {h: torch.from_numpy(cache_np[h][rows_dst[sl]]).to(device, dt)
                       for h, _ in spec}

            def mk_hook(h, pos):
                def hook(t, hook):
                    t = t.clone()
                    t[:, pos, :] = patches[h][:, pos, :]
                    return t
                return hook

            with torch.no_grad():
                lg = model.run_with_hooks(
                    x_in[sl].to(device),
                    fwd_hooks=[(h, mk_hook(h, pos)) for h, pos in spec])
            soft_p[sl] = F.softmax(
                lg[:, PATCH_P - 1:PATCH_P - 1 + ROLL, :].float(), -1).cpu().numpy()
        sp = steps_from_soft(soft_p)
        Qp_tree = np.prod(sp, 2)
        klsteps = stepwise_kls_from_probs(sp, Qu_steps, branches)
        tgtv = 1 - np.median(tree_kl(Qp_tree, Qv_tree)) / max(
            np.median(tree_kl(Qu_tree, Qv_tree)), EPS)
        obs = np.log(np.clip(Qp_tree, EPS, None)) - np.log(np.clip(Qu_tree, EPS, None))
        pred = np.log(np.clip(th_v, EPS, None)) - np.log(np.clip(th_w, EPS, None))
        r2 = float(np.corrcoef(obs.ravel(), pred.ravel())[0, 1] ** 2)
        slope = float(np.cov(obs.ravel(), pred.ravel())[0, 1] / np.var(pred.ravel()))
        out_rows.append({
            "seed": seed, "kind": kind,
            "kl_step1": float(np.median(klsteps[:, 0])),
            "kl_step2": float(np.median(klsteps[:, 1])),
            "kl_step3": float(np.median(klsteps[:, 2])),
            "tgtv_frac": float(tgtv),
            "kl_to_model_v": float(np.median(tree_kl(Qp_tree, Qv_tree))),
            "kl_unpatched_to_model_v": float(np.median(tree_kl(Qu_tree, Qv_tree))),
            "dlog_r2": r2, "dlog_slope": slope, "n_pairs": npairs,
        })
        print(f"  [xf-chan s{seed}] {kind:14s} kl1={out_rows[-1]['kl_step1']:.4f} "
              f"kl2={out_rows[-1]['kl_step2']:.4f} kl3={out_rows[-1]['kl_step3']:.4f} "
              f"tgtv%={100 * tgtv:.1f} R2={r2:.3f} slope={slope:.3f}", flush=True)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    f = out / f"xf_channels_s{seed}.csv"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"  saved {f}  ({time.time()-t0:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default="results/causal_transplant")
    args = ap.parse_args()
    run_one(args.run_dir, args.device, args.out_dir)


if __name__ == "__main__":
    main()
