"""
Order-k Markov classical-baseline sweep (paper App. on higher-order classical baselines).

Read-only over already-trained multi-seed runs. For every converged run, at its
FINAL checkpoint, computes the SAME headline probe used in
bloch_multiseed_weighted_r2.py / stage2_multiseed_weighted_r2.py:

  representation  = concat per-layer activations at the last context position
                    - transformer: hook_resid_pre across all n_layers
                    - RNN (LSTM/GRU/vanilla): h_n across all num_layers
  probe           = NO-PCA, full 256-dim concat, CV weighted R² (Eq.48,
                    shuffle_control_simple._weighted_r2, 5-fold, standardized per
                    fold, rcond=1e-10), ONE fixed-seed row permutation (seed 0).

Two targets per run:
  * belief  : minimal/GHMM belief states (enumerate_process_data) -- the reference,
              recomputed here with the identical probe so its CV folds are identical
              to the order-k folds (cross-checked vs the existing per-run CSVs).
  * order-k : the order-k Markov approximation is state-deterministic, so the order-k
              belief at the last position is EXACTLY one-hot(last k tokens) -- a single
              category in 0..|X|^k-1. We regress onto that one-hot target WITHOUT ever
              building a dense |X|^k tensor, via the verified sparse closed form from
              a one-off check script (not included) (matches the dense GHMM walk to ~1e-14 and
              the dense weighted-R² to ~1e-16). Computed for k = 1..8.

Processes (all n_ctx=8, vocab from the process transition tensor):
  Bloch = tom_quantum                 vocab 4
  Mess3 = mess3(x=0.05,a=0.85)        vocab 3   (classical)
  Moon  = post_quantum(alpha=e,b=0.5) vocab 3   (post-quantum)

Usage:
    PYTHONPATH=. uv run python scripts/activation_analysis/orderk_sweep.py --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from epsilon_transformers.analysis.checkpoints import (
    load_run_config, list_checkpoints, load_model,
)
from epsilon_transformers.analysis.process_targets import enumerate_process_data
from epsilon_transformers.analysis.weighted_probe import _weighted_r2
from epsilon_transformers.analysis.multiseed_probe import (
    extract_transformer_concat, load_rnn, extract_rnn_concat,
)
from epsilon_transformers.process.transition_matrices import get_matrix_from_args

N_FOLDS = 5
RCOND = 1e-10
CONV_THRESH = 1.005      # last-row loss.csv val_loss_mean < 1.005 => converged
SHUFFLE_SEED = 0         # fixed row permutation for CV folds (matches existing scripts)
KMAX = 8

ARCH_ORDER = ["transformer", "LSTM", "GRU", "vanilla_RNN"]
PROC_ORDER = ["Bloch", "Mess3", "Moon"]

# process display label -> (list of run-dir roots, process_config.name filter or None)
PROCESSES = {
    "Bloch": {
        "dirs": ["results/multiseed_bloch_transformer", "results/multiseed_bloch_rnn"],
        "name_filter": None,          # every run in these dirs is Bloch
    },
    "Mess3": {
        "dirs": ["results/multiseed_mess3_transformer", "results/multiseed_stage2_rnn"],
        "name_filter": "mess3",
    },
    "Moon": {
        "dirs": ["results/multiseed_moon_transformer", "results/multiseed_stage2_rnn"],
        "name_filter": "post_quantum",
    },
}


# --------------------------------------------------------------------------
# Sparse order-k one-hot CV weighted R2 (verbatim from the verified probe at
# a one-off check script (not included); matches dense _weighted_r2 to ~1e-16).
# --------------------------------------------------------------------------
def suffix_labels(contexts, k, vocab):
    """Category index of the last k tokens, base-|X|; == the order-k one-hot class."""
    lab = np.zeros(len(contexts), dtype=np.int64)
    for j in range(k):
        lab = lab * vocab + contexts[:, contexts.shape[1] - k + j]
    return lab


def onehot_cv_weighted_r2(X, labels, M, weights, n_folds=N_FOLDS, rcond=RCOND,
                          row_chunk=4096):
    """CV weighted R2 against a one-hot(M) target, never materializing it densely.
    Fold logic mirrors shuffle_control_simple._weighted_r2 exactly."""
    N = X.shape[0]
    w = weights / weights.sum()
    idx = np.arange(N)
    fold = N // n_folds
    r2s = []
    for f in range(n_folds):
        s, e = f * fold, (f + 1) * fold if f < n_folds - 1 else N
        te, tr = idx[s:e], np.concatenate([idx[:s], idx[e:]])
        Xtr, Xte, wtr, wte = X[tr], X[te], w[tr], w[te]
        ltr, lte = labels[tr], labels[te]
        mean = Xtr.mean(0); std = Xtr.std(0); std[std < 1e-12] = 1.0
        A = np.hstack([np.ones((len(tr), 1)), (Xtr - mean) / std])
        B = np.hstack([np.ones((len(te), 1)), (Xte - mean) / std])
        sw = np.sqrt(wtr / wtr.sum())[:, None]
        U, S, Vt = np.linalg.svd(A * sw, full_matrices=False)
        keep = S > rcond * S.max()
        acc = np.zeros((M, U.shape[1]))            # (onehot*sw)^T U, scatter-add
        np.add.at(acc, ltr, U * sw)
        beta = Vt.T[:, keep] @ (acc.T[keep] / S[keep, None])   # (d+1, M)
        wte_n = wte / wte.sum()
        m = np.zeros(M); np.add.at(m, lte, wte_n)  # weighted target mean
        ss_tot = 1.0 - 2.0 * m[lte] @ wte_n + (m @ m)
        ss_res = 0.0
        for cs in range(0, len(te), row_chunk):
            P = B[cs:cs + row_chunk] @ beta
            rows = np.arange(P.shape[0])
            ss_res += (wte_n[cs:cs + row_chunk] *
                       ((P * P).sum(1) - 2.0 * P[rows, lte[cs:cs + row_chunk]] + 1.0)).sum()
        r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    return float(np.mean(r2s))


# --------------------------------------------------------------------------
# Process targets + run discovery
# --------------------------------------------------------------------------
def final_val_lb(run_dir: Path) -> float:
    f = run_dir / "loss.csv"
    if not f.exists():
        return float("inf")
    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    return float(rows[-1]["val_loss_mean"]) if rows else float("inf")


def build_process_targets(cfg):
    proc = enumerate_process_data(cfg)
    n_ctx = cfg["model_config"]["n_ctx"]
    contexts = proc["seqs"].numpy()[:, :n_ctx]
    belief = proc["beliefs"]
    weights = proc["weights"]
    T = get_matrix_from_args(**cfg["process_config"])
    vocab = int(T.shape[0])
    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(len(weights))
    return {
        "contexts": contexts, "n_ctx": n_ctx, "vocab": vocab,
        "belief": belief, "weights": weights, "perm": perm,
        "belief_dim": int(belief.shape[1]), "N": int(len(weights)),
    }


def classify_run(cfg):
    mc = cfg["model_config"]
    if "rnn_type" in mc:
        arch = mc["rnn_type"].upper()
        if arch == "RNN":
            arch = "vanilla_RNN"
    else:
        arch = "transformer"
    return arch, mc.get("seed", "?")


def discover_runs(process):
    """Return list of (run_dir Path, arch, seed, is_rnn, cfg) for a process."""
    spec = PROCESSES[process]
    out = []
    for base in spec["dirs"]:
        for p in sorted(Path(base).iterdir()):
            if not (p.is_dir() and p.name.startswith("run_")):
                continue
            cfg = load_run_config(str(p))
            if spec["name_filter"] and cfg["process_config"]["name"] != spec["name_filter"]:
                continue
            arch, seed = classify_run(cfg)
            is_rnn = "rnn_type" in cfg["model_config"]
            out.append((p, arch, seed, is_rnn, cfg))
    return out


# --------------------------------------------------------------------------
# Per-run analysis
# --------------------------------------------------------------------------
def analyze_run(run_dir, arch, seed, is_rnn, proc, device):
    val_lb = final_val_lb(run_dir)
    ckpts = list_checkpoints(str(run_dir))
    if not ckpts:
        raise RuntimeError("no checkpoints")
    ckpt = ckpts[-1]

    if is_rnn:
        model, _ = load_rnn(run_dir, ckpt, device, proc["vocab"])
        acts = extract_rnn_concat(model, proc["contexts"], proc["vocab"], device)
        del model
        if "cuda" in str(device):
            torch.cuda.empty_cache()
    else:
        acts = extract_transformer_concat(run_dir, ckpt, proc["contexts"], device)

    p, w = proc["perm"], proc["weights"]
    Xp, wp = acts[p], w[p]
    belief_r2 = _weighted_r2(Xp, proc["belief"][p], wp, n_folds=N_FOLDS, rcond=RCOND)

    orderk = {}
    for k in range(1, KMAX + 1):
        lab = suffix_labels(proc["contexts"], k, proc["vocab"])
        M = proc["vocab"] ** k
        orderk[k] = onehot_cv_weighted_r2(Xp, lab[p], M, wp)

    return {
        "arch": arch, "seed": seed, "run_dir": run_dir.name,
        "vocab": proc["vocab"], "val_loss_mean": val_lb,
        "converged": bool(val_lb < CONV_THRESH),
        "tokens_seen": int(ckpt.replace(".pt", "")),
        "concat_dim": int(acts.shape[1]),
        "belief_r2": float(belief_r2), "orderk": {k: float(v) for k, v in orderk.items()},
    }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def t_interval(vals, alpha=0.05):
    from scipy import stats
    v = np.asarray(vals, dtype=float)
    n = len(v)
    m = float(v.mean())
    if n < 2:
        return m, float("nan")
    sd = float(v.std(ddof=1))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return m, float(tcrit * sd / np.sqrt(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_csv", default="results/orderk_sweep.csv")
    ap.add_argument("--agg_csv", default="results/orderk_sweep_agg.csv")
    ap.add_argument("--joblib", default="results/orderk_sweep.joblib")
    args = ap.parse_args()

    t0 = time.time()
    results = []       # per-run dicts, with process key added
    failures = []
    proc_meta = {}
    for process in PROC_ORDER:
        runs = discover_runs(process)
        # process targets from the first run's cfg (process identical across runs)
        proc = build_process_targets(runs[0][4])
        proc_meta[process] = {"vocab": proc["vocab"], "N": proc["N"],
                              "belief_dim": proc["belief_dim"], "n_ctx": proc["n_ctx"]}
        print(f"\n=== {process}  vocab={proc['vocab']} N={proc['N']} "
              f"belief_dim={proc['belief_dim']} n_ctx={proc['n_ctx']}  ({len(runs)} runs) ===",
              flush=True)
        for (rd, arch, seed, is_rnn, _cfg) in runs:
            t = time.time()
            try:
                r = analyze_run(rd, arch, seed, is_rnn, proc, args.device)
                r["process"] = process
                results.append(r)
                ks = r["orderk"]
                print(f"  [{arch:<12}] seed={str(seed):<3} conv={'Y' if r['converged'] else 'N'} "
                      f"bel={r['belief_r2']:.4f} | k1={ks[1]:.3f} k3={ks[3]:.3f} "
                      f"k5={ks[5]:.3f} k8={ks[8]:.3f}  ({time.time()-t:.1f}s)", flush=True)
            except Exception as ex:
                import traceback
                traceback.print_exc()
                failures.append({"process": process, "run_dir": rd.name,
                                 "error": f"{type(ex).__name__}: {ex}"})
                print(f"  [{rd.name}] FAILED: {type(ex).__name__}: {ex}", flush=True)
        joblib.dump({"per_run": results, "failures": failures}, args.joblib + ".partial")

    # ----- long per-run CSV -----
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["process", "arch", "seed", "run_dir", "vocab", "val_loss_mean",
                    "converged", "k", "orderk_r2", "belief_r2"])
        for r in results:
            for k in range(1, KMAX + 1):
                w.writerow([r["process"], r["arch"], r["seed"], r["run_dir"], r["vocab"],
                            r["val_loss_mean"], r["converged"], k, r["orderk"][k],
                            r["belief_r2"]])

    # ----- aggregate CSV: mean +/- 95% t per (process, arch, k), plus belief -----
    agg = {}
    with open(args.agg_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["process", "arch", "n_converged", "vocab", "quantity", "k",
                    "mean", "ci95_hw", "ceiling_257_over_Xk"])
        for process in PROC_ORDER:
            vocab = proc_meta[process]["vocab"]
            for arch in ARCH_ORDER:
                conv = [r for r in results if r["process"] == process
                        and r["arch"] == arch and r["converged"]]
                if not conv:
                    continue
                bel_vals = [r["belief_r2"] for r in conv]
                bm, bhw = t_interval(bel_vals)
                agg[(process, arch, "belief")] = (bm, bhw)
                w.writerow([process, arch, len(conv), vocab, "belief", "", bm, bhw, ""])
                for k in range(1, KMAX + 1):
                    vals = [r["orderk"][k] for r in conv]
                    m, hw = t_interval(vals)
                    ceil = min(1.0, 257.0 / vocab ** k)
                    agg[(process, arch, k)] = (m, hw)
                    w.writerow([process, arch, len(conv), vocab, "orderk", k, m, hw, ceil])

    payload = {
        "per_run": results, "failures": failures, "proc_meta": proc_meta,
        "meta": {
            "representation": "concat all-layer (transformer resid_pre / RNN h_n) @ last pos",
            "regime": "no-PCA full-concat (App-F consistent)",
            "probe_dim": 256, "probe_dim_with_intercept": 257,
            "n_folds": N_FOLDS, "rcond": RCOND, "kmax": KMAX,
            "shuffle_seed": SHUFFLE_SEED, "conv_thresh": CONV_THRESH,
            "orderk_target": "one-hot(last k tokens) == exact order-k Markov belief",
            "r2_fn": "shuffle_control_simple._weighted_r2 / sparse one-hot equivalent",
        },
    }
    joblib.dump(payload, args.joblib)

    # ----- console summary table -----
    print("\n" + "=" * 100)
    print("ORDER-k SWEEP SUMMARY  (converged runs; mean over seeds, NO-PCA CV weighted R²)")
    print("=" * 100)
    for process in PROC_ORDER:
        vocab = proc_meta[process]["vocab"]
        print(f"\n--- {process}  (vocab={vocab}) ---")
        hdr = f"{'arch':<12} {'belief':>8} | " + " ".join(f"k{k:>6}" for k in range(1, KMAX + 1))
        print(hdr)
        print("-" * len(hdr))
        for arch in ARCH_ORDER:
            if (process, arch, "belief") not in agg:
                continue
            bm = agg[(process, arch, "belief")][0]
            cells = " ".join(f"{agg[(process, arch, k)][0]:7.3f}" for k in range(1, KMAX + 1))
            print(f"{arch:<12} {bm:8.4f} | {cells}")
        ceilrow = " ".join(f"{min(1.0, 257.0/vocab**k):7.3f}" for k in range(1, KMAX + 1))
        print(f"{'CEIL 257/Xk':<12} {'--':>8} | {ceilrow}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f['process']} {f['run_dir']}: {f['error']}")
    print(f"\nSaved: {args.out_csv} , {args.agg_csv} , {args.joblib}")
    print(f"Total runs analyzed: {len(results)}  |  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
