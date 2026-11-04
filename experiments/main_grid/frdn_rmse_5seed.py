"""FRDN bar metrics, 5 seeds (App. extended-architecture figures, FRDN row).

Same as fig2_rmse_5seed.py, for every converged multi-seed FRDN run x 4 architectures:
score the FINAL checkpoint and the step-0 checkpoint (same seed = untrained control)
against the compact-geometry belief and the Markov-order-3 belief, with the exact
fold scheme of weighted_probe._weighted_r2 (5 folds on the fixed SHUFFLE_SEED
permutation, train-standardized, weighted lstsq rcond=1e-10, no PCA).

Per (run, checkpoint, target) it records:
  r2         per-fold-mean CV weighted R^2   (same scheme as r2_mess3_moon_frdn_5seed.py)
  rmse_cv    sqrt( sum_i p_i ||eta_i - etahat_i||^2 ) over pooled held-out predictions
  rmse_in    same, in-sample fit on all contexts (closest to the submitted Fig. 2)

Read-only on checkpoints. Usage (repo root):
  PYTHONPATH=. uv run python experiments/main_grid/frdn_rmse_5seed.py --device cuda:2
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from epsilon_transformers.analysis.checkpoints import load_run_config, list_checkpoints
from epsilon_transformers.analysis.multiseed_probe import (
    build_process_targets, final_val_lb, CONV_THRESH, scores, acts_for,
)

RUN_DIRS = [
    "results/multiseed_frdn_transformer", "results/multiseed_frdn_rnn",
]
PROC = {"fanizza": "FRDN"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/paper_metrics/frdn_metrics_multiseed.csv")
    args = ap.parse_args()
    t0 = time.time()
    runs = []
    for base in RUN_DIRS:
        for p in sorted(Path(base).iterdir()):
            if p.is_dir() and p.name.startswith("run_"):
                cfg = load_run_config(str(p))
                name = cfg["process_config"]["name"]
                if name in PROC:
                    runs.append((p, cfg, name))
    print(f"{len(runs)} runs", flush=True)
    targets = {}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["process", "arch", "seed", "run_dir", "converged", "checkpoint",
                     "target", "r2", "rmse_cv", "rmse_in", "target_weighted_sd"])
        for p, cfg, name in runs:
            if name not in targets:
                targets[name] = build_process_targets(cfg)
            proc = targets[name]
            mc = cfg["model_config"]
            arch = mc["rnn_type"].upper() if "rnn_type" in mc else "transformer"
            arch = "vanilla_RNN" if arch == "RNN" else arch
            conv = final_val_lb(p) < CONV_THRESH
            ck = list_checkpoints(str(p))
            perm, wts = proc["perm"], proc["weights"]
            for tag, c in (("final", ck[-1]), ("init", ck[0])):
                A = acts_for(p, c, cfg, proc, args.device)[perm]
                for tname in ("belief", "markov3"):
                    Y = proc[tname][perm]
                    wn = wts[perm] / wts.sum()
                    sd = float(np.sqrt((wn[:, None] * (Y - (Y * wn[:, None]).sum(0)) ** 2).sum()))
                    r2, rc, ri = scores(A, Y, wts[perm])
                    wr.writerow([PROC[name], arch, mc.get("seed", "?"), p.name, conv, f"{tag}:{c}",
                                 tname, r2, rc, ri, sd])
                    print(f"{PROC[name]:5s} {arch:12s} s={mc.get('seed')} {tag:5s} {tname:7s} "
                          f"r2={r2:.4f} rmse_cv={rc:.4f} rmse_in={ri:.4f}", flush=True)
            fh.flush()
    print(f"done {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
