"""
Stage-2 multi-seed weighted-R² aggregation (Mess3 / Moon / FRDN, 5 seeds).

Read-only analysis of already-finished Stage-2 runs. Adapted DIRECTLY from
scripts/activation_analysis/bloch_multiseed_weighted_r2.py: same representation
(concat all-layer activations at last position), same CV weighted-R² (Eq.48,
5-fold, standardized per fold, rcond=1e-10), same convergence filter, same fixed
row-permutation so belief and markov3 share identical folds (exact paired gap).

Differences from the Bloch script:
  * THREE processes, each with its own vocab (read from the process transition
    tensor T.shape[0] -- NOT hardcoded; the stock quick_lstm/batch_sweep vocab=2
    is wrong for Mess3/Moon):
        Mess3 = mess3(x=0.05, a=0.85)              vocab 3   (classical)
        Moon  = post_quantum(alpha=e, beta=0.5)    vocab 3   (post-quantum)
        FRDN  = fanizza(alpha=2000, lamb=0.49)     vocab 2   (demoted quantum)
    n_ctx = 8 for all.
  * Grouping is by (process, architecture). RNN sweep dirs mix processes, so each
    run's process is taken from process_config.name and its arch from
    model_config.rnn_type.
  * Reports the NO-PCA (full-concat) regime only -- the faithful/App-F-consistent
    method (do NOT use PCA-95%).

Usage:
    PYTHONPATH=. uv run python scripts/activation_analysis/stage2_multiseed_weighted_r2.py \
        --device cuda:3
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
    load_run_config, list_checkpoints, load_model,
)
from scripts.activation_analysis.enumerate_process import enumerate_process_data
from scripts.activation_analysis.shuffle_control_simple import _weighted_r2
from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM, markov_approximation
from epsilon_transformers.training.networks import RNNWrapper

N_FOLDS = 5
RCOND = 1e-10
CONV_THRESH = 1.005      # last-row loss.csv val_loss_mean < 1.005 => converged
MARKOV_ORDER = 3
SHUFFLE_SEED = 0         # fixed row permutation for CV folds

# process_config.name -> display label
PROC_LABEL = {"mess3": "Mess3", "post_quantum": "Moon", "fanizza": "FRDN"}
PROC_ORDER = ["Mess3", "Moon", "FRDN"]
ARCH_ORDER = ["transformer", "LSTM", "GRU", "vanilla_RNN"]

# Every Stage-2 run directory to score.
RUN_DIRS = [
    "results/multiseed_mess3_transformer",
    "results/multiseed_moon_transformer",
    "results/multiseed_frdn_transformer",
    "results/multiseed_stage2_rnn",
    "results/multiseed_frdn_rnn",
]


# --------------------------------------------------------------------------
# Process-level targets (one set per process -> computed once, cached)
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

    ghmm = TransitionMatrixGHMM(T)
    m3 = markov_approximation(ghmm, order=MARKOV_ORDER)
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


def r2_nopca(acts, tgt, proc):
    """No-PCA (full-concat) CV weighted R² for a target, on the fixed perm."""
    p = proc["perm"]
    w = proc["weights"]
    return _weighted_r2(acts[p], tgt[p], w[p], n_folds=N_FOLDS, rcond=RCOND)


def classify_run(cfg):
    mc = cfg["model_config"]
    is_rnn = "rnn_type" in mc
    if is_rnn:
        arch = mc["rnn_type"].upper()
        if arch == "RNN":
            arch = "vanilla_RNN"
    else:
        arch = "transformer"
    proc_name = cfg["process_config"]["name"]
    return arch, PROC_LABEL[proc_name], proc_name, mc.get("seed", "?"), is_rnn


def analyze_run(run_dir: Path, proc, arch, process, seed, is_rnn, device):
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

    r2_bel = r2_nopca(acts, proc["belief"], proc)
    r2_m3 = r2_nopca(acts, proc["markov3"], proc)

    return {
        "process": process, "arch": arch, "seed": seed, "run_dir": run_dir.name,
        "vocab": proc["vocab"], "val_loss_mean": val_lb,
        "converged": bool(val_lb < CONV_THRESH),
        "tokens_seen": int(ckpt.replace(".pt", "")),
        "concat_dim": int(acts.shape[1]), "n_contexts": int(proc["N"]),
        # no-PCA lstsq is underdetermined when unique contexts <= feature dim
        "underdetermined": bool(proc["N"] <= acts.shape[1]),
        "belief_r2": float(r2_bel), "markov3_r2": float(r2_m3),
        "gap": float(r2_bel - r2_m3),
    }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def t_interval(vals, alpha=0.05):
    """(mean, half_width) for a two-sided (1-alpha) Student-t CI; hw=nan if n<2."""
    from scipy import stats
    v = np.asarray(vals, dtype=float)
    n = len(v)
    m = float(v.mean())
    if n < 2:
        return m, float("nan")
    sd = float(v.std(ddof=1))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return m, float(tcrit * sd / np.sqrt(n))


def aggregate(results):
    groups = {}
    for process in PROC_ORDER:
        for arch in ARCH_ORDER:
            runs = [r for r in results if r["process"] == process and r["arch"] == arch]
            if not runs:
                continue
            conv = [r for r in runs if r["converged"]]
            failed_seeds = sorted([r["seed"] for r in runs if not r["converged"]],
                                  key=lambda s: (s == "?", s))
            entry = {"process": process, "arch": arch,
                     "n_total": len(runs), "n_converged": len(conv),
                     "failed_seeds": failed_seeds,
                     "vocab": runs[0]["vocab"],
                     "n_contexts": runs[0]["n_contexts"],
                     "underdetermined": bool(runs[0].get("underdetermined", False))}
            if conv:
                bel = np.array([r["belief_r2"] for r in conv])
                m3 = np.array([r["markov3_r2"] for r in conv])
                gap = np.array([r["gap"] for r in conv])
                gm, ghw = t_interval(gap)
                entry.update({
                    "belief_mean": float(bel.mean()),
                    "belief_sd": float(bel.std(ddof=1)) if len(bel) > 1 else float("nan"),
                    "markov3_mean": float(m3.mean()),
                    "markov3_sd": float(m3.std(ddof=1)) if len(m3) > 1 else float("nan"),
                    "gap_min": float(gap.min()), "gap_mean": gm, "gap_ci95_hw": ghw,
                    "gap_values": gap.tolist(),
                })
            groups[(process, arch)] = entry
    return groups


def _mpm(mean, sd):
    return f"{mean:.4f}" if np.isnan(sd) else f"{mean:.4f}±{sd:.4f}"


def _gapci(g):
    if np.isnan(g["gap_ci95_hw"]):
        return f"{g['gap_mean']:.4f} (n<2)"
    return f"{g['gap_mean']:.4f}±{g['gap_ci95_hw']:.4f}"


def print_report(results, groups, proc_meta):
    # -------- Full 60-run table, per-process blocks --------
    print("\n" + "=" * 104)
    print("PER-RUN RESULTS (final ckpt, concat all-layer probe, NO-PCA CV weighted R²)")
    print("=" * 104)
    for process in PROC_ORDER:
        m = proc_meta[process]
        print(f"\n--- {process}  (vocab={m['vocab']}  N_ctx-seqs={m['N']}  "
              f"belief_dim={m['belief_dim']}  markov{MARKOV_ORDER}_dim={m['markov3_dim']}) ---")
        print(f"{'arch':<12} {'seed':>4} {'conv':>4} {'val_loss':>9} | "
              f"{'belief_R2':>10} {'markov3_R2':>11} {'gap':>9} {'concat':>7}")
        print("-" * 84)
        block = [r for r in results if r["process"] == process]
        block.sort(key=lambda r: (ARCH_ORDER.index(r["arch"]) if r["arch"] in ARCH_ORDER else 9,
                                  str(r["seed"])))
        for r in block:
            print(f"{r['arch']:<12} {str(r['seed']):>4} "
                  f"{'Y' if r['converged'] else 'N':>4} {r['val_loss_mean']:>9.5f} | "
                  f"{r['belief_r2']:>10.4f} {r['markov3_r2']:>11.4f} "
                  f"{r['gap']:>9.4f} {r['concat_dim']:>7}")

    # -------- Per (process x arch) summary --------
    print("\n" + "=" * 104)
    print("PER (PROCESS × ARCH) SUMMARY  (converged runs only, NO-PCA)")
    print("=" * 104)
    print(f"{'process':<8} {'arch':<12} {'Nconv':>6} | {'belief_R2':>16} "
          f"{'markov3_R2':>16} {'gap_min':>8} {'gap_mean±95%t':>18}")
    print("-" * 104)
    for process in PROC_ORDER:
        for arch in ARCH_ORDER:
            g = groups.get((process, arch))
            if g is None:
                continue
            if g["n_converged"] == 0:
                print(f"{process:<8} {arch:<12} {g['n_converged']:>2}/{g['n_total']:<2}"
                      f"   (no converged runs; failed seeds: {g['failed_seeds']})")
                continue
            flag = ""
            if g["n_converged"] < g["n_total"]:
                flag += f"  [!{g['n_total']-g['n_converged']} nonconv: {g['failed_seeds']}]"
            if g.get("underdetermined"):
                flag += f"  [!UNDERDET N={g['n_contexts']}<=dim256: markov3/gap unreliable]"
            if g["gap_min"] <= 0:
                flag += "  [!gap_min<=0]"
            print(f"{process:<8} {arch:<12} {g['n_converged']:>2}/{g['n_total']:<2}   | "
                  f"{_mpm(g['belief_mean'], g['belief_sd']):>16} "
                  f"{_mpm(g['markov3_mean'], g['markov3_sd']):>16} "
                  f"{g['gap_min']:>8.4f} {_gapci(g):>18}{flag}")

    # -------- Verdicts --------
    print("\n" + "=" * 104)
    print("NOTES / VERDICTS")
    print("=" * 104)
    print("  Moon  = post-quantum: expect belief ≫ markov3 (positive gap), like Bloch.")
    print("  Mess3 = classical: belief & markov3 both classical; gap may be ~0 (EXPECTED, honest).")
    print("  FRDN  = demoted quantum process.")
    for process in PROC_ORDER:
        for arch in ARCH_ORDER:
            g = groups.get((process, arch))
            if g is None or g["n_converged"] == 0:
                continue
            tag = []
            if g["gap_min"] > 0.02:
                tag.append("gap clearly>0")
            elif g["gap_min"] > 0:
                tag.append("thin gap>0")
            else:
                tag.append("gap<=0")
            if g["n_converged"] < g["n_total"]:
                tag.append(f"LOW CONV {g['n_converged']}/{g['n_total']}")
            print(f"  {process:<6} {arch:<12} belief={g['belief_mean']:.3f} "
                  f"markov3={g['markov3_mean']:.3f} gap_min={g['gap_min']:+.4f} "
                  f"gap_mean={g['gap_mean']:+.4f}  -> {', '.join(tag)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_prefix", default="results/stage2_multiseed_weighted_r2")
    ap.add_argument("--smoke", action="store_true",
                    help="One run per (process,arch) group, for validation.")
    args = ap.parse_args()

    t0 = time.time()
    # Collect all run dirs + classify
    all_runs = []          # (run_dir Path, arch, process, proc_name, seed, is_rnn)
    for base in RUN_DIRS:
        for p in sorted(Path(base).iterdir()):
            if p.is_dir() and p.name.startswith("run_"):
                cfg = load_run_config(str(p))
                arch, process, proc_name, seed, is_rnn = classify_run(cfg)
                all_runs.append((p, arch, process, proc_name, seed, is_rnn, cfg))
    print(f"Discovered {len(all_runs)} Stage-2 runs across {len(RUN_DIRS)} dirs.")

    # Build one target set per process (cache by proc_name)
    proc_targets = {}
    proc_meta = {}
    for (_, _, process, proc_name, _, _, cfg) in all_runs:
        if proc_name in proc_targets:
            continue
        t = time.time()
        pt = build_process_targets(cfg)
        proc_targets[proc_name] = pt
        proc_meta[process] = pt
        print(f"  targets[{process:<5} {proc_name:<13}] vocab={pt['vocab']} N={pt['N']} "
              f"belief_dim={pt['belief_dim']} markov{MARKOV_ORDER}_dim={pt['markov3_dim']} "
              f"({time.time()-t:.1f}s)", flush=True)

    if args.smoke:
        seen = set()
        picked = []
        for r in all_runs:
            key = (r[2], r[1])
            if key not in seen:
                seen.add(key)
                picked.append(r)
        all_runs = picked
        print(f"SMOKE: {len(all_runs)} runs (one per group).")

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    for (p, arch, process, proc_name, seed, is_rnn, cfg) in all_runs:
        t = time.time()
        try:
            r = analyze_run(p, proc_targets[proc_name], arch, process, seed,
                            is_rnn, args.device)
            results.append(r)
            print(f"  [{process:<5} {arch:<12}] seed={str(seed):<3} "
                  f"val={r['val_loss_mean']:.5f} conv={'Y' if r['converged'] else 'N'} "
                  f"belief={r['belief_r2']:.4f} markov3={r['markov3_r2']:.4f} "
                  f"gap={r['gap']:+.4f}  ({time.time()-t:.1f}s)", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            failures.append({"run_dir": p.name, "error": f"{type(e).__name__}: {e}"})
            print(f"  [{p.name}] FAILED: {type(e).__name__}: {e}", flush=True)
        joblib.dump({"per_run": results, "failures": failures},
                    args.out_prefix + ".partial.joblib")

    groups = aggregate(results)
    print_report(results, groups, proc_meta)
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f['run_dir']}: {f['error']}")

    # Save joblib (rich) + summary joblib groups serializable
    groups_ser = {f"{k[0]}|{k[1]}": v for k, v in groups.items()}
    meta_ser = {proc: {kk: proc_meta[proc][kk]
                       for kk in ("vocab", "N", "belief_dim", "markov3_dim")}
                for proc in proc_meta}
    payload = {
        "per_run": results, "groups": groups_ser, "failures": failures,
        "meta": {
            "proc_meta": meta_ser,
            "representation": "concat all-layer (transformer resid_pre / RNN h_n) @ last pos",
            "regime": "no-PCA full-concat (App-F consistent)",
            "markov_order": MARKOV_ORDER, "n_folds": N_FOLDS, "rcond": RCOND,
            "r2_fn": "shuffle_control_simple._weighted_r2 (CV weighted R², Eq.48)",
            "conv_thresh": CONV_THRESH, "shuffle_seed": SHUFFLE_SEED,
        },
    }
    joblib.dump(payload, args.out_prefix + ".joblib")

    # Per-run CSV
    with open(args.out_prefix + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["process", "arch", "seed", "run_dir", "vocab", "val_loss_mean",
                    "converged", "tokens_seen", "concat_dim",
                    "belief_r2", "markov3_r2", "gap"])
        for r in results:
            w.writerow([r["process"], r["arch"], r["seed"], r["run_dir"], r["vocab"],
                        r["val_loss_mean"], r["converged"], r["tokens_seen"],
                        r["concat_dim"], r["belief_r2"], r["markov3_r2"], r["gap"]])
    # Per (process x arch) summary CSV
    with open(args.out_prefix + "_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["process", "arch", "n_converged", "n_total", "vocab",
                    "belief_mean", "belief_sd", "markov3_mean", "markov3_sd",
                    "gap_min", "gap_mean", "gap_ci95_hw", "failed_seeds"])
        for process in PROC_ORDER:
            for arch in ARCH_ORDER:
                g = groups.get((process, arch))
                if g is None:
                    continue
                if g["n_converged"] == 0:
                    w.writerow([process, arch, 0, g["n_total"], g["vocab"],
                                "", "", "", "", "", "", "",
                                ";".join(map(str, g["failed_seeds"]))])
                    continue
                w.writerow([process, arch, g["n_converged"], g["n_total"], g["vocab"],
                            g["belief_mean"], g["belief_sd"], g["markov3_mean"],
                            g["markov3_sd"], g["gap_min"], g["gap_mean"],
                            g["gap_ci95_hw"], ";".join(map(str, g["failed_seeds"]))])
    print(f"\nSaved: {args.out_prefix}.joblib , .csv , _summary.csv")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
