"""Top-3 PCs of activations (appendix PCA views).

Seed 42, final checkpoint, Mess3 / Bloch Walk / Moon x 4 architectures. Activations as in the
probe (concatenated all layers, last position) plus the last layer alone. Probability-weighted
PCA over all contexts. Saves PC coordinates + ground-truth beliefs + weights to npz so the
figure can be styled locally; also a quick PNG.

Usage (repo root): CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python \
    scripts/paper_metrics/pca_views.py --device cuda:0
"""
import argparse
from pathlib import Path

import numpy as np

from scripts.activation_analysis.run_shuffle_analysis import load_run_config, list_checkpoints
from scripts.activation_analysis.stage2_multiseed_weighted_r2 import build_process_targets
import importlib.util, sys
spec = importlib.util.spec_from_file_location("fm", "scripts/paper_metrics/fig2_metrics_multiseed.py")
fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)

OUT = Path("results/paper_metrics/pca")
DIRS = {"tom_quantum": ["results/multiseed_bloch_transformer", "results/multiseed_bloch_rnn"],
        "mess3": ["results/multiseed_mess3_transformer", "results/multiseed_stage2_rnn"],
        "post_quantum": ["results/multiseed_moon_transformer", "results/multiseed_stage2_rnn"]}


def wpca(X, w, k=3):
    w = w / w.sum()
    mu = (w[:, None] * X).sum(0)
    Xc = (X - mu) * np.sqrt(w)[:, None]
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2
    return (X - mu) @ Vt[:k].T, var[:k] / var.sum(), float(var.sum() ** 2 / (var ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for pname, dirs in DIRS.items():
        proc = None
        for d in dirs:
            for p in sorted(Path(d).iterdir()):
                if not (p.is_dir() and p.name.startswith("run_")):
                    continue
                cfg = load_run_config(str(p))
                if cfg["process_config"]["name"] != pname or cfg["model_config"].get("seed") != a.seed:
                    continue
                if proc is None:
                    proc = build_process_targets(cfg)
                mc = cfg["model_config"]
                arch = mc["rnn_type"].upper() if "rnn_type" in mc else "transformer"
                arch = "vanilla_RNN" if arch == "RNN" else arch
                A = fm.acts_for(p, list_checkpoints(str(p))[-1], cfg, proc, a.device)
                nl = 4
                last = A[:, -A.shape[1] // nl:]
                res = {}
                for tag, X in (("concat", A), ("lastlayer", last)):
                    pcs, frac, pr = wpca(X.astype(np.float64), proc["weights"])
                    res[f"{tag}_pcs"], res[f"{tag}_frac"], res[f"{tag}_pr"] = pcs, frac, pr
                    print(f"{pname:12s} {arch:12s} {tag:9s} top3 var {frac.round(3)} sum {frac.sum():.3f} PR {pr:.2f}", flush=True)
                np.savez_compressed(OUT / f"{pname}_{arch}_s{a.seed}.npz", belief=proc["belief"],
                                    weights=proc["weights"], **res)
    print("done")


if __name__ == "__main__":
    main()
