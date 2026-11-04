"""Aggregate the erasure-audit outputs (erasure_transformer.py, erasure_recurrent.py) across seeds into summary tables.

Usage:
  PYTHONPATH=. uv run python experiments/interventions/erasure_aggregate.py \
      [--dir results/causal_transplant/audit]
"""
from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict

import numpy as np


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += list(csv.DictReader(open(f)))
    return rows


def fmt(vals):
    v = np.array([x for x in vals if not np.isnan(x)])
    if len(v) == 0:
        return "      --      "
    return f"{v.mean():+.3f}±{v.std():.3f}" if len(v) > 1 else f"{v[0]:+.3f}       "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/causal_transplant/audit")
    args = ap.parse_args()

    # ---------------- transformer residue ----------------
    rows = load(f"{args.dir}/resid_xf_s*.csv")
    seeds = sorted({r["seed"] for r in rows})
    print(f"=== TRANSFORMER RESIDUE (mean±sd over {len(seeds)} seeds; positions 5/7/9 avg) ===")
    print(f"{'phase':6s} {'surgery':9s} {'cap':7s} {'role':5s} {'tgt':4s} "
          f"{'r2_lin_cv':>14s} {'r2_mlp_cv':>14s}")
    acc = defaultdict(lambda: (defaultdict(list), defaultdict(list)))
    for r in rows:
        if r["pos"] not in ("5", "7", "9"):
            continue
        key = (r["phase"], f"{r['kind']}@{r['surgery']}", r["cap_site"], r["role"], r["target"])
        acc[key][0][r["seed"]].append(float(r["r2_lin_cv"]))
        acc[key][1][r["seed"]].append(float(r["r2_mlp_cv"]))
    order = {"clean": 0, "orig": 1, "scrub": 2}
    for key in sorted(acc, key=lambda k: (order[k[0]], k[1], k[2], k[4])):
        lin, mlp = acc[key]
        lv = [np.nanmean(lin[s]) for s in seeds if s in lin]
        mv = [np.nanmean(mlp[s]) if not np.all(np.isnan(mlp[s])) else np.nan
              for s in seeds if s in mlp]
        print(f"{key[0]:6s} {key[1]:9s} {key[2]:7s} {key[3]:5s} {key[4]:4s} "
              f"{fmt(lv):>14s} {fmt(mv):>14s}")

    # ---------------- transformer scrub costs ----------------
    rows = load(f"{args.dir}/scrub_xf_s*.csv")
    print(f"\n=== TRANSFORMER COSTS: orig vs converged scrub (dloss nats; {len(seeds)} seeds) ===")
    print(f"{'config':10s} {'orig_dloss':>14s} {'scrub_dloss':>14s} {'final_r2cv':>11s} {'iters':>6s}")
    by = defaultdict(list)
    for r in rows:
        by[(r["kind"], r["sites"], r["seed"])].append(r)
    cfgs = sorted({(r["kind"], r["sites"]) for r in rows})
    for kind, sites in cfgs:
        od, sd_, fr, ni = [], [], [], []
        for s in seeds:
            rs = by.get((kind, sites, s), [])
            orig = [r for r in rs if r["stage"] == "orig"]
            scr = [r for r in rs if r["stage"] == "scrub"]
            if orig:
                od.append(float(orig[0]["dloss_nats"]))
            if scr:
                last = max(scr, key=lambda r: int(r["iter"]))
                sd_.append(float(last["dloss_nats"]))
                fr.append(float(last["r2_cv_max"]))
                ni.append(int(last["iter"]))
        print(f"{kind}@{sites:6s} {fmt(od):>14s} {fmt(sd_):>14s} {fmt(fr):>11s} "
              f"{np.mean(ni) if ni else float('nan'):>6.1f}")

    # ---------------- RNN ----------------
    rows = load(f"{args.dir}/resid_rnn_*.csv")
    if rows:
        rseeds = sorted({r["seed"] for r in rows})
        print(f"\n=== RNN RESIDUE (steps; {len(rseeds)} seeds) ===")
        print(f"{'phase':6s} {'step':4s} {'role':5s} {'tgt':4s} {'n':>5s} "
              f"{'r2_lin_cv':>14s} {'r2_mlp_cv':>14s}")
        acc = defaultdict(lambda: (defaultdict(list), defaultdict(list), []))
        for r in rows:
            key = (r["phase"], int(r["step"]), r["role"], r["target"])
            acc[key][0][r["seed"]].append(float(r["r2_lin_cv"]))
            acc[key][1][r["seed"]].append(float(r["r2_mlp_cv"]))
            acc[key][2].append(int(r["n_uniq"]))
        for key in sorted(acc, key=lambda k: (order[k[0]], k[1], k[3])):
            lin, mlp, ns = acc[key]
            lv = [np.nanmean(lin[s]) for s in rseeds if s in lin]
            mv = [np.nanmean(mlp[s]) if not np.all(np.isnan(mlp[s])) else np.nan
                  for s in rseeds if s in mlp]
            print(f"{key[0]:6s} {key[1]:<4d} {key[2]:5s} {key[3]:4s} {ns[0]:>5d} "
                  f"{fmt(lv):>14s} {fmt(mv):>14s}")

        rows = load(f"{args.dir}/scrub_rnn_*.csv")
        print(f"\n=== RNN COSTS: orig vs converged scrub (kl_future nats) ===")
        print(f"{'seed':6s} {'orig_future':>12s} {'scrub_future':>13s} {'theory':>8s} "
              f"{'orig_local':>11s} {'scrub_local':>12s} {'final_r2cv':>11s}")
        by = defaultdict(list)
        for r in rows:
            by[r["seed"]].append(r)
        for s in sorted(by):
            rs = by[s]
            orig = [r for r in rs if r["stage"] == "orig"][0]
            last = max([r for r in rs if r["stage"] == "scrub"], key=lambda r: int(r["iter"]))
            print(f"{s:6s} {float(orig['kl_future']):>12.4f} {float(last['kl_future']):>13.4f} "
                  f"{float(orig['theory_future']):>8.4f} {float(orig['kl_local']):>11.4f} "
                  f"{float(last['kl_local']):>12.4f} {float(last['r2_cv_max']):>11.4f}")


if __name__ == "__main__":
    main()
