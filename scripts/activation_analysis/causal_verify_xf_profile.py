"""
Completes independent transformer coverage: per-site prefix-swap layer profile
(claimed L1 ~18% / L2 ~8% / L3 ~6% of the KL gap to the donor-model dist) and
per-site single-position-6 full residual swaps (claimed inert, ~0%).

Usage:
  PYTHONPATH=. uv run python scripts/activation_analysis/causal_verify_xf_profile.py --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from scripts.activation_analysis.causal_verify_xf import (
    process_setup, enum_prefixes, branch_mats, theory_tree, tree_kl,
    matched_pairs, load_xf, run_cfg, softmax_at, XF_DIR, OUT_DIR, P, ROLL, EPS,
)

SITES = ["blocks.1.hook_resid_pre", "blocks.2.hook_resid_pre", "blocks.3.hook_resid_pre"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rows = []
    run_dirs = sorted(XF_DIR.glob("run_*"))
    cfg = run_cfg(run_dirs[0])
    T, rev, b0, E, Pcol, Pker, vocab, d = process_setup(cfg["process_config"])
    toks7, bel7, wt7, p07 = enum_prefixes(T, rev, b0, P)
    branches, bmats = branch_mats(T, ROLL)
    nb = len(branches)

    for rd in run_dirs:
        seed = run_cfg(rd)["model_config"]["seed"]
        model = load_xf(rd, args.device)
        seqs = np.concatenate([np.repeat(toks7, nb, 0),
                               np.tile(branches, (len(toks7), 1))], 1)
        x_all = torch.from_numpy(seqs).long().to(args.device)
        with torch.no_grad():
            _, cache = model.run_with_cache(x_all, names_filter=SITES)
        cache_dev = {h: cache[h].to(model.cfg.dtype) for h in SITES}
        soft_clean = softmax_at(model, x_all)

        src, dst = matched_pairs(bel7, wt7, p07, Pker)
        npairs = len(src)
        rs = (np.repeat(src, nb) * nb + np.tile(np.arange(nb), npairs))
        rd_ = (np.repeat(dst, nb) * nb + np.tile(np.arange(nb), npairs))
        br_rep = np.tile(branches, (npairs, 1))

        def tree_of(soft_rows):
            sp = np.stack([soft_rows[np.arange(len(soft_rows)), s, br_rep[:, s]]
                           for s in range(ROLL)], 1)
            return np.prod(sp, 1).reshape(npairs, nb)

        Qu, Qv = tree_of(soft_clean[rs]), tree_of(soft_clean[rd_])
        gap = np.median(tree_kl(Qu, Qv))
        x_src = x_all[rs]

        for site in SITES:
            for kind, ps in (("prefix_swap", slice(0, P)), ("pos6_swap", slice(P - 1, P))):
                def mk(lo, hi, site=site, ps=ps):
                    donor = cache_dev[site][rd_[lo:hi]]
                    def fn(t, hook, donor=donor, ps=ps):
                        t = t.clone()
                        t[:, ps, :] = donor[:, ps, :]
                        return t
                    return [(site, fn)]
                Qp = tree_of(softmax_at(model, x_src, hooks=mk))
                closed = float(100 * (1 - np.median(tree_kl(Qp, Qv)) / max(gap, EPS)))
                rows.append({"seed": seed, "site": "L" + site.split(".")[1],
                             "kind": kind, "pct_closed": closed})
                print(f"[xf s{seed} {kind}@{site.split('.')[1]}] closed={closed:6.1f}%",
                      flush=True)
        del model, cache, cache_dev
        torch.cuda.empty_cache()

    f = OUT_DIR / "verify_xf_profile.csv"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"saved {f}")


if __name__ == "__main__":
    main()
