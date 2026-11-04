"""
Weighted-R² aggregation for Stage-1 Bloch-Walk multi-seed runs.

Read-only analysis of already-finished 20000-epoch runs. For every run, at its
FINAL checkpoint, computes the paper's §3.2 headline concat-layer probe:

  representation  = concatenated per-layer activations at the last context position
                    - transformer: hook_resid_pre across all n_layers
                    - RNN (LSTM/GRU/vanilla): h_n across all num_layers
  preprocessing   = PCA-95% (as in concat_layers_belief_cv.py / _maybe_pca)
  R²              = cross-validated weighted R² (shuffle_control_simple._weighted_r2,
                    paper Eq.48, 5-fold, standardized per fold, rcond=1e-10)
  targets         - belief  : minimal/GHMM belief states (enumerate_process_data)
                  - markov3 : classical Markov-order-3 belief states, aligned to the
                              SAME enumerated contexts + weights

Vocab BUG FIX: quick_lstm_analyze / batch_sweep_analyze hardcode vocab_size=2. The
Bloch (tom_quantum) process has vocab_size=4 (d_vocab), taken here from the process
transition tensor and asserted == 4.

Row-shuffle note: the enumerated contexts are in lexicographic order, so the default
contiguous 5-fold split of _weighted_r2 would place an entire first-token block in one
fold (train/test distribution shift). We apply ONE fixed-seed row permutation to
(X, belief, markov3, weights) before CV so folds are random; belief and markov3 share
the identical folds, keeping the paired gap exact.

Usage:
    PYTHONPATH=. uv run python scripts/activation_analysis/bloch_multiseed_weighted_r2.py \
        --device cpu
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

from scripts.activation_analysis.run_shuffle_analysis import (
    load_run_config, list_checkpoints, load_model, _maybe_pca,
)
from scripts.activation_analysis.enumerate_process import enumerate_process_data
from scripts.activation_analysis.shuffle_control_simple import _weighted_r2
from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM, markov_approximation
from epsilon_transformers.training.networks import RNNWrapper

PCA_RANK = 0.95
N_FOLDS = 5
RCOND = 1e-10
CONV_THRESH = 1.005      # val_loss_mean (LB-normalized) < 1.005 => converged
SHUFFLE_SEED = 0         # fixed row permutation for CV folds

TRANSFORMER_DIR = "results/multiseed_bloch_transformer"
RNN_DIR = "results/multiseed_bloch_rnn"


# --------------------------------------------------------------------------
# Process-level targets (identical across all runs -> computed once)
# --------------------------------------------------------------------------
def _stationary_belief(T, rev):
    P = T.sum(axis=0)
    ev, evec = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(ev - 1.0))
    b0 = np.real(evec[:, idx])
    return b0 / (b0 @ rev)


def _walk_beliefs(T, rev, b0, contexts):
    """Belief at the last position for each context, walking the GHMM."""
    n_states = T.shape[1]
    out = np.zeros((len(contexts), n_states), dtype=np.float64)
    for i, toks in enumerate(contexts):
        b = b0.copy()
        for x in toks:
            bn = b @ T[int(x)]
            nrm = float(bn @ rev)
            if nrm > 0:
                b = bn / nrm
        out[i] = b
    return out


def build_process_targets(cfg):
    """Return dict with contexts, belief target, markov3 target, weights."""
    proc = enumerate_process_data(cfg)
    n_ctx = cfg["model_config"]["n_ctx"]
    contexts = proc["seqs"].numpy()[:, :n_ctx]     # tokens the model sees
    belief = proc["beliefs"]                        # minimal/GHMM belief
    weights = proc["weights"]

    T = get_matrix_from_args(**cfg["process_config"])
    vocab = int(T.shape[0])
    assert vocab == 4, f"expected vocab_size=4 for Bloch, got {vocab}"

    ghmm = TransitionMatrixGHMM(T)
    m3 = markov_approximation(ghmm, order=3)
    T3 = m3.transition_matrices
    rev3 = np.asarray(m3.right_eigenvector).squeeze()
    b0_3 = _stationary_belief(T3, rev3)
    markov3 = _walk_beliefs(T3, rev3, b0_3, contexts)

    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(len(weights))
    return {
        "contexts": contexts, "n_ctx": n_ctx, "vocab": vocab,
        "belief": belief, "markov3": markov3, "weights": weights,
        "perm": perm,
        "belief_dim": int(belief.shape[1]), "markov3_dim": int(markov3.shape[1]),
        "N": int(len(weights)),
    }


# --------------------------------------------------------------------------
# Activation extraction
# --------------------------------------------------------------------------
def extract_transformer_concat(run_dir, ckpt, contexts, device, bs=8192):
    """Concat hook_resid_pre across all layers, last position."""
    m = load_model(str(run_dir), ckpt, device)
    n_layers = m.cfg.n_layers
    names = [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)]
    chunks = {k: [] for k in names}
    inp = torch.from_numpy(contexts).long()
    with torch.no_grad():
        for s in range(0, inp.shape[0], bs):
            _, cache = m.run_with_cache(inp[s:s + bs].to(device), names_filter=names)
            for k in names:
                chunks[k].append(cache[k][:, -1, :].cpu().numpy())
    del m
    if "cuda" in str(device):
        torch.cuda.empty_cache()
    return np.concatenate([np.concatenate(chunks[k], 0) for k in names], axis=1)


def load_rnn(run_dir, ckpt, device, vocab):
    cfg = load_run_config(str(run_dir))
    mc = cfg["model_config"]
    cls = getattr(nn, mc["rnn_type"].upper())
    rnn = cls(input_size=vocab, hidden_size=mc["hidden_size"],
              num_layers=mc["num_layers"], dropout=mc.get("dropout", 0.0),
              batch_first=True, bidirectional=mc.get("bidirectional", False)).to(device)
    out = nn.Linear(mc["hidden_size"], vocab).to(device)
    model = RNNWrapper(rnn, out).to(device)
    state = torch.load(f"{run_dir}/{ckpt}", map_location=device, weights_only=True)
    model.load_state_dict(state)      # asserts vocab correctness via shape match
    model.eval()
    return model, cfg


def extract_rnn_concat(model, contexts, vocab, device, bs=8192):
    """Concat h_n across all layers, last time step. Handles LSTM (h_n,c_n)
    tuple and GRU/vanilla-RNN (h_n tensor)."""
    outs = []
    with torch.no_grad():
        for s in range(0, len(contexts), bs):
            x = torch.from_numpy(contexts[s:s + bs]).long().to(device)
            oh = torch.zeros(*x.shape, vocab, device=device)
            oh.scatter_(2, x.unsqueeze(-1), 1.0)
            _, hidden = model.rnn(oh)
            h_n = hidden[0] if isinstance(hidden, tuple) else hidden  # (num_layers,N,H)
            outs.append(torch.cat([h_n[i] for i in range(h_n.shape[0])], dim=1).cpu().numpy())
    return np.concatenate(outs, 0)


# --------------------------------------------------------------------------
# Per-run analysis
# --------------------------------------------------------------------------
def final_val_lb(run_dir: Path) -> float:
    f = run_dir / "loss.csv"
    if not f.exists():
        return float("inf")
    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    return float(rows[-1]["val_loss_mean"]) if rows else float("inf")


def r2_pair(acts, tgt, proc, pca_rank):
    """Return (n_features_used, CV weighted R²) for a target, at given PCA rank.
    pca_rank=None => no PCA, full concatenated activations."""
    X = _maybe_pca(acts, pca_rank)[0] if pca_rank is not None else acts
    p = proc["perm"]
    w = proc["weights"]
    return (X.shape[1],
            _weighted_r2(X[p], tgt[p], w[p], n_folds=N_FOLDS, rcond=RCOND))


def analyze_run(run_dir: Path, proc, device):
    cfg = load_run_config(str(run_dir))
    mc = cfg["model_config"]
    seed = mc.get("seed", "?")
    is_rnn = "rnn_type" in mc
    arch = mc["rnn_type"].upper() if is_rnn else "transformer"
    if arch == "RNN":
        arch = "vanilla_RNN"
    val_lb = final_val_lb(run_dir)
    ckpts = list_checkpoints(str(run_dir))
    if not ckpts:
        raise RuntimeError("no checkpoints")
    ckpt = ckpts[-1]

    if is_rnn:
        model, _ = load_rnn(run_dir, ckpt, device, proc["vocab"])
        acts = extract_rnn_concat(model, proc["contexts"], proc["vocab"], device)
        del model
    else:
        acts = extract_transformer_concat(run_dir, ckpt, proc["contexts"], device)

    # PCA-95% regime (headline)
    d_pca_b, r2_bel_p = r2_pair(acts, proc["belief"], proc, PCA_RANK)
    _,       r2_m3_p  = r2_pair(acts, proc["markov3"], proc, PCA_RANK)
    # No-PCA regime (full 256-dim concat — strongest form for both targets)
    d_full, r2_bel_n = r2_pair(acts, proc["belief"], proc, None)
    _,      r2_m3_n  = r2_pair(acts, proc["markov3"], proc, None)

    return {
        "arch": arch, "seed": seed, "run_dir": run_dir.name,
        "val_loss_lb": val_lb, "converged": bool(val_lb < CONV_THRESH),
        "tokens_seen": int(ckpt.replace(".pt", "")),
        "concat_dim": int(acts.shape[1]), "d_pca": int(d_pca_b),
        "d_full": int(d_full),
        # PCA-95% regime
        "belief_r2": float(r2_bel_p), "markov3_r2": float(r2_m3_p),
        "gap": float(r2_bel_p - r2_m3_p),
        # No-PCA regime
        "belief_r2_nopca": float(r2_bel_n), "markov3_r2_nopca": float(r2_m3_n),
        "gap_nopca": float(r2_bel_n - r2_m3_n),
    }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def t_interval(vals, alpha=0.05):
    """Return (mean, half_width) for a two-sided (1-alpha) Student-t CI.
    half_width = nan if n < 2."""
    from scipy import stats
    v = np.asarray(vals, dtype=float)
    n = len(v)
    m = float(v.mean())
    if n < 2:
        return m, float("nan")
    sd = float(v.std(ddof=1))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return m, tcrit_hw(tcrit, sd, n)


def tcrit_hw(tcrit, sd, n):
    return float(tcrit * sd / np.sqrt(n))


def _regime_stats(conv, bkey, mkey, gkey):
    bel = np.array([r[bkey] for r in conv])
    m3 = np.array([r[mkey] for r in conv])
    gap = np.array([r[gkey] for r in conv])
    gm, ghw = t_interval(gap)
    return {
        "belief_mean": float(bel.mean()),
        "belief_sd": float(bel.std(ddof=1)) if len(bel) > 1 else float("nan"),
        "markov3_mean": float(m3.mean()),
        "markov3_sd": float(m3.std(ddof=1)) if len(m3) > 1 else float("nan"),
        "markov3_min": float(m3.min()), "markov3_max": float(m3.max()),
        "gap_min": float(gap.min()), "gap_mean": gm, "gap_ci95_hw": ghw,
        "gap_values": gap.tolist(),
    }


def aggregate(results):
    groups = {}
    order = ["transformer", "LSTM", "GRU", "vanilla_RNN"]
    for arch in order:
        runs = [r for r in results if r["arch"] == arch]
        conv = [r for r in runs if r["converged"]]
        failed_seeds = sorted([r["seed"] for r in runs if not r["converged"]],
                              key=lambda s: (s == "?", s))
        entry = {"n_total": len(runs), "n_converged": len(conv),
                 "failed_seeds": failed_seeds}
        if conv:
            entry["pca"] = _regime_stats(conv, "belief_r2", "markov3_r2", "gap")
            entry["nopca"] = _regime_stats(conv, "belief_r2_nopca",
                                           "markov3_r2_nopca", "gap_nopca")
        groups[arch] = entry
    return groups


def _mpm(mean, sd):
    return f"{mean:.4f}" if np.isnan(sd) else f"{mean:.4f}±{sd:.4f}"


def _gapci(g):
    return (f"{g['gap_mean']:.4f} (n<2)" if np.isnan(g["gap_ci95_hw"])
            else f"{g['gap_mean']:.4f}±{g['gap_ci95_hw']:.4f}")


def print_report(results, groups, proc):
    print("\n" + "=" * 110)
    print("PER-RUN RESULTS (final ckpt, concat all-layer probe, CV weighted R²) — PCA-95% vs NO-PCA")
    print(f"N contexts={proc['N']}  vocab={proc['vocab']}  "
          f"belief_dim={proc['belief_dim']}  markov3_dim={proc['markov3_dim']}  concat_dim=256")
    print("=" * 110)
    print(f"{'arch':<12} {'seed':>4} {'conv':>4} | "
          f"{'bel(pca)':>9} {'mk3(pca)':>9} {'gap(pca)':>9} | "
          f"{'bel(full)':>9} {'mk3(full)':>9} {'gap(full)':>9}")
    print("-" * 110)
    order = {"transformer": 0, "LSTM": 1, "GRU": 2, "vanilla_RNN": 3}
    for r in sorted(results, key=lambda r: (order.get(r["arch"], 9), str(r["seed"]))):
        print(f"{r['arch']:<12} {str(r['seed']):>4} "
              f"{'Y' if r['converged'] else 'N':>4} | "
              f"{r['belief_r2']:>9.4f} {r['markov3_r2']:>9.4f} {r['gap']:>9.4f} | "
              f"{r['belief_r2_nopca']:>9.4f} {r['markov3_r2_nopca']:>9.4f} "
              f"{r['gap_nopca']:>9.4f}")

    print("\n" + "=" * 110)
    print("PER-ARCHITECTURE COMPARISON (converged runs only): PCA-95% vs NO-PCA (full 256-dim)")
    print("=" * 110)
    print(f"{'arch':<12} {'Nconv':>5} {'regime':>7} | "
          f"{'belief_R2':>16} {'markov3_R2':>16} {'gap_min':>8} {'gap_mean±95%t':>18}")
    print("-" * 110)
    for arch in ["transformer", "LSTM", "GRU", "vanilla_RNN"]:
        g = groups[arch]
        if g["n_converged"] == 0:
            print(f"{arch:<12} {g['n_converged']:>2}/{g['n_total']:<2}  (no converged runs)")
            continue
        for regime, tag in [("pca", "PCA-95"), ("nopca", "no-PCA")]:
            s = g[regime]
            print(f"{arch:<12} {g['n_converged']:>2}/{g['n_total']:<2} {tag:>7} | "
                  f"{_mpm(s['belief_mean'], s['belief_sd']):>16} "
                  f"{_mpm(s['markov3_mean'], s['markov3_sd']):>16} "
                  f"{s['gap_min']:>8.4f} {_gapci(s):>18}")
        if g["failed_seeds"]:
            print(f"             failed seeds: {g['failed_seeds']}")

    # Verdicts
    print("\n" + "=" * 110)
    print("VERDICTS")
    print("=" * 110)
    all_pos = True
    for arch in ["transformer", "LSTM", "GRU", "vanilla_RNN"]:
        g = groups[arch]
        if g["n_converged"] == 0:
            continue
        gm = g["nopca"]["gap_min"]
        if gm <= 0:
            all_pos = False
        print(f"  {arch:<12} no-PCA gap_min = {gm:+.4f}  "
              f"({'>0 ✓' if gm > 0 else '<=0 ✗'})")
    print(f"  -> no-PCA min-over-seeds gap clearly > 0 for EVERY architecture: "
          f"{'YES' if all_pos else 'NO'}")
    m3_lo = min(g["nopca"]["markov3_min"] for g in groups.values() if g["n_converged"])
    m3_hi = max(g["nopca"]["markov3_max"] for g in groups.values() if g["n_converged"])
    print(f"  -> no-PCA markov3 R² range across all converged runs: "
          f"[{m3_lo:.3f}, {m3_hi:.3f}]  (paper App-F classical range: 0.36–0.88)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_prefix", default="results/bloch_multiseed_weighted_r2")
    args = ap.parse_args()

    t0 = time.time()
    # Process targets (shared) — build from a transformer config (process is identical)
    ref_cfg = load_run_config(f"{TRANSFORMER_DIR}/run_0_L4_H4_DH16_DM64_tom_quantum")
    print("Building shared process targets (enumerate + markov3)...")
    proc = build_process_targets(ref_cfg)
    print(f"  N={proc['N']} vocab={proc['vocab']} belief_dim={proc['belief_dim']} "
          f"markov3_dim={proc['markov3_dim']}  ({time.time()-t0:.1f}s)")

    run_dirs = []
    for d in (Path(TRANSFORMER_DIR), Path(RNN_DIR)):
        run_dirs += sorted([p for p in d.iterdir()
                            if p.is_dir() and p.name.startswith("run_")])

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    for rd in run_dirs:
        t = time.time()
        try:
            r = analyze_run(rd, proc, args.device)
            results.append(r)
            print(f"  [{r['arch']:<12}] seed={str(r['seed']):<3} "
                  f"val/LB={r['val_loss_lb']:.5f} conv={'Y' if r['converged'] else 'N'} "
                  f"belief={r['belief_r2']:.4f} markov3={r['markov3_r2']:.4f} "
                  f"gap={r['gap']:.4f}  ({time.time()-t:.1f}s)", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            failures.append({"run_dir": rd.name, "error": f"{type(e).__name__}: {e}"})
            print(f"  [{rd.name}] FAILED: {type(e).__name__}: {e}", flush=True)
        # incremental save so partial results survive any interruption
        joblib.dump({"per_run": results, "failures": failures},
                    args.out_prefix + ".partial.joblib")

    groups = aggregate(results)
    print_report(results, groups, proc)
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f['run_dir']}: {f['error']}")

    # Save
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "per_run": results, "groups": groups, "failures": failures,
        "meta": {
            "N": proc["N"], "vocab": proc["vocab"],
            "belief_dim": proc["belief_dim"], "markov3_dim": proc["markov3_dim"],
            "representation": "concat all-layer (transformer resid_pre / RNN h_n) @ last pos",
            "pca_rank": PCA_RANK, "n_folds": N_FOLDS, "rcond": RCOND,
            "r2_fn": "shuffle_control_simple._weighted_r2 (CV weighted R², paper Eq.48)",
            "conv_thresh": CONV_THRESH, "shuffle_seed": SHUFFLE_SEED,
        },
    }
    joblib.dump(payload, args.out_prefix + ".joblib")
    # PCA-95% headline CSV (unchanged schema)
    with open(args.out_prefix + ".csv", "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["arch", "seed", "run_dir", "val_loss_lb", "converged",
                       "tokens_seen", "concat_dim", "d_pca",
                       "belief_r2", "markov3_r2", "gap"])
        for r in results:
            wcsv.writerow([r["arch"], r["seed"], r["run_dir"], r["val_loss_lb"],
                           r["converged"], r["tokens_seen"], r["concat_dim"],
                           r["d_pca"], r["belief_r2"], r["markov3_r2"], r["gap"]])
    # No-PCA (full-dim) CSV + a combined per-run CSV with both regimes
    with open(args.out_prefix + "_nopca.csv", "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["arch", "seed", "run_dir", "val_loss_lb", "converged",
                       "concat_dim", "belief_r2_nopca", "markov3_r2_nopca",
                       "gap_nopca"])
        for r in results:
            wcsv.writerow([r["arch"], r["seed"], r["run_dir"], r["val_loss_lb"],
                           r["converged"], r["concat_dim"], r["belief_r2_nopca"],
                           r["markov3_r2_nopca"], r["gap_nopca"]])
    with open(args.out_prefix + "_both.csv", "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["arch", "seed", "converged", "d_pca",
                       "belief_r2_pca", "markov3_r2_pca", "gap_pca",
                       "belief_r2_nopca", "markov3_r2_nopca", "gap_nopca"])
        for r in results:
            wcsv.writerow([r["arch"], r["seed"], r["converged"], r["d_pca"],
                           r["belief_r2"], r["markov3_r2"], r["gap"],
                           r["belief_r2_nopca"], r["markov3_r2_nopca"],
                           r["gap_nopca"]])
    print(f"\nSaved: {args.out_prefix}.joblib , .csv , _nopca.csv , _both.csv")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
