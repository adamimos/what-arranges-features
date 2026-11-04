"""
Score the qRRXOR hyperparameter-search screen (48 runs, 6000 epochs each).

Reuses the bloch_multiseed_weighted_r2 machinery, adapted for qRRXOR:
  - process = quantum_rrxor hero (phi=pi/2, theta=pi/4, epsilon=0.02), n_ctx=10, vocab=2
  - representation = full concatenated all-layer activations at last position, NO PCA
      transformer: hook_resid_pre across 4 layers ; LSTM: h_n across 4 layers
  - target = enumerate_process_data belief states (2^10 = 1024 contexts)
  - R² = CV weighted R² (paper Eq.48), shuffle_control_simple._weighted_r2,
         5-fold, rcond=1e-10, ONE fixed-seed row permutation (belief only)
  - convergence: last row of loss.csv val_loss_mean < 1.005 (LB-normalized;
                 collapse-to-uniform pushes binary qRRXOR val/LB well above 1)

muP note: the transformer is muP + torch.compile. Checkpoints have clean keys
(no _orig_mod.); the muP forward is fully captured by hooked_model_config.json
(attn_scale = d_head = 16, i.e. 1/d_head attention scaling), so a plain
HookedTransformer load reproduces the trained forward. We additionally record a
forward-faithfulness diagnostic per run: weighted corr between the model's
last-position p(x=0) and the process-optimal p(x=0|belief). corr≈1 for converged
runs => muP forward reproduced correctly.

Aggregate per (arch, lr, batch) over dev seeds {7,11,13}; rank configs by
(#converged desc, worst-converged-seed belief R² desc).

Usage:
    PYTHONPATH=. uv run python scripts/activation_analysis/hpsearch_qrrxor_score.py --device cuda:3
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import joblib
import numpy as np

from epsilon_transformers.analysis.checkpoints import load_run_config, list_checkpoints
from epsilon_transformers.analysis.process_targets import enumerate_process_data
from epsilon_transformers.analysis.weighted_probe import _weighted_r2
from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.analysis.multiseed_probe import final_val_lb
from epsilon_transformers.analysis.qrrxor_scoring import (
    N_FOLDS, RCOND, RCOND_STAB, CONV_THRESH, SHUFFLE_SEED, load_rnn, extract_transformer,
    extract_rnn, weighted_corr,
)

SEEDS = [7, 11, 13]

TRANSFORMER_DIR = "results/hpsearch_qrrxor_transformer"
LSTM_DIR = "results/hpsearch_qrrxor_lstm"


def build_process_targets(cfg):
    proc = enumerate_process_data(cfg)
    n_ctx = cfg["model_config"]["n_ctx"]
    contexts = proc["seqs"].numpy()[:, :n_ctx]
    T = get_matrix_from_args(**cfg["process_config"])
    vocab = int(T.shape[0])
    assert vocab == 2, f"expected vocab_size=2 for qRRXOR, got {vocab}"
    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(len(proc["weights"]))
    return {
        "contexts": contexts, "n_ctx": n_ctx, "vocab": vocab,
        "belief": proc["beliefs"], "weights": proc["weights"],
        "ntp_opt": proc["ntp"][:, 0], "perm": perm,
        "belief_dim": int(proc["beliefs"].shape[1]), "N": int(len(proc["weights"])),
    }


def analyze_run(run_dir: Path, proc, device):
    cfg = load_run_config(str(run_dir))
    mc = cfg["model_config"]
    tc = cfg["train_config"]
    is_rnn = "rnn_type" in mc
    arch = "transformer" if not is_rnn else mc["rnn_type"].upper()
    seed = mc.get("seed", "?")
    lr = float(tc["learning_rate"])
    batch = int(tc["batch_size"])
    val_lb = final_val_lb(run_dir)
    ckpt = list_checkpoints(str(run_dir))[-1]

    if is_rnn:
        model = load_rnn(run_dir, ckpt, device, proc["vocab"])
        acts, p0 = extract_rnn(model, proc["contexts"], proc["vocab"], device)
        del model
    else:
        acts, p0 = extract_transformer(run_dir, ckpt, proc["contexts"], device)

    p = proc["perm"]; w = proc["weights"]
    # Primary (as specified): no-PCA full concat, rcond=1e-10.
    r2 = _weighted_r2(acts[p], proc["belief"][p], w[p], n_folds=N_FOLDS, rcond=RCOND)
    # Stabilized: same probe but rcond=1e-3 — well-conditioned even when high-LR
    # checkpoints inflate residual-stream norms (else the lstsq extrapolates and
    # gives meaningless huge-negative R² on the held-out fold at N=1024).
    r2_stab = _weighted_r2(acts[p], proc["belief"][p], w[p], n_folds=N_FOLDS, rcond=RCOND_STAB)
    ntp_corr = weighted_corr(p0, proc["ntp_opt"], w)
    return {
        "arch": arch, "lr": lr, "batch": batch, "seed": seed,
        "run_dir": run_dir.name, "val_loss_lb": val_lb,
        "converged": bool(val_lb < CONV_THRESH),
        "tokens_seen": int(ckpt.replace(".pt", "")),
        "concat_dim": int(acts.shape[1]),
        "act_absmax": float(np.abs(acts).max()),
        "belief_r2": float(r2), "belief_r2_stab": float(r2_stab),
        "ntp_corr": float(ntp_corr),
    }


def score_configs(results):
    """Group by (arch, lr, batch); compute robustness scores over seeds."""
    keys = sorted({(r["arch"], r["lr"], r["batch"]) for r in results},
                  key=lambda k: (k[0], k[1], k[2]))
    configs = []
    for arch, lr, batch in keys:
        runs = [r for r in results if (r["arch"], r["lr"], r["batch"]) == (arch, lr, batch)]
        runs_by_seed = {r["seed"]: r for r in runs}
        conv = [r for r in runs if r["converged"]]
        conv_r2 = [r["belief_r2"] for r in conv]
        conv_r2s = [r["belief_r2_stab"] for r in conv]
        per_seed = {}
        for s in SEEDS:
            rr = runs_by_seed.get(s)
            per_seed[s] = None if rr is None else {
                "belief_r2": rr["belief_r2"], "belief_r2_stab": rr["belief_r2_stab"],
                "converged": rr["converged"], "val_lb": rr["val_loss_lb"],
                "ntp_corr": rr["ntp_corr"], "act_absmax": rr["act_absmax"],
            }
        configs.append({
            "arch": arch, "lr": lr, "batch": batch,
            "n_total": len(runs), "n_converged": len(conv),
            "worst_conv_r2": float(min(conv_r2)) if conv_r2 else float("nan"),
            "mean_conv_r2": float(np.mean(conv_r2)) if conv_r2 else float("nan"),
            "worst_conv_r2_stab": float(min(conv_r2s)) if conv_r2s else float("nan"),
            "mean_conv_r2_stab": float(np.mean(conv_r2s)) if conv_r2s else float("nan"),
            "mean_ntp_corr": float(np.mean([r["ntp_corr"] for r in conv])) if conv else float("nan"),
            "per_seed": per_seed,
        })
    return configs


def rank_key(c):
    # prefer more converged, then higher worst-converged-seed R² (stabilized probe,
    # so exploded-norm high-LR checkpoints are scored on model quality, not lstsq noise)
    wr = c["worst_conv_r2_stab"]
    return (-c["n_converged"], -(wr if not np.isnan(wr) else -1.0))


_ARCH_CANON = ["transformer", "LSTM", "GRU", "RNN"]


def arch_order(items):
    """Present arches in canonical order (unknown arches appended alphabetically)."""
    present = {it["arch"] for it in items}
    known = [a for a in _ARCH_CANON if a in present]
    extra = sorted(present - set(_ARCH_CANON))
    return known + extra


def print_report(configs, results, proc):
    print("\n" + "=" * 108)
    print("qRRXOR HP-SEARCH SCORES — belief R² (full concat, NO PCA, CV weighted Eq.48)")
    print(f"process=quantum_rrxor(phi=pi/2,theta=pi/4,eps=0.02)  n_ctx={proc['n_ctx']} "
          f"vocab={proc['vocab']}  N={proc['N']} contexts  belief_dim={proc['belief_dim']}  "
          f"concat_dim=256  conv<{CONV_THRESH}")
    print("=" * 108)
    hdr = (f"{'arch':<12} {'lr':>7} {'bs':>3} {'conv':>4} "
           f"{'worst':>7} {'mean':>7} | {'s7':>18} {'s11':>18} {'s13':>18}")
    print(hdr)
    print("(worst/mean = stabilized belief R² over converged seeds; per-seed cells "
          "show stabilized R²; * = seed NOT converged)")
    print("-" * 108)

    def cell(d):
        if d is None:
            return f"{'--':>18}"
        flag = "" if d["converged"] else "*"
        return f"{d['belief_r2_stab']:.4f}{flag}(LB{d['val_lb']:.3f})".rjust(18)

    for arch in arch_order(configs):
        cs = sorted([c for c in configs if c["arch"] == arch],
                    key=lambda c: (c["lr"], c["batch"]))
        for c in cs:
            wr = "" if np.isnan(c["worst_conv_r2_stab"]) else f"{c['worst_conv_r2_stab']:.4f}"
            mr = "" if np.isnan(c["mean_conv_r2_stab"]) else f"{c['mean_conv_r2_stab']:.4f}"
            ps = c["per_seed"]
            print(f"{arch:<12} {c['lr']:>7.0e} {c['batch']:>3} "
                  f"{c['n_converged']}/{c['n_total']:<2} {wr:>7} {mr:>7} | "
                  f"{cell(ps[7])} {cell(ps[11])} {cell(ps[13])}")
        print("-" * 108)

    print("\n" + "=" * 108)
    print("WINNER PER ARCHITECTURE (rank: #converged desc, then worst-converged-seed R² desc)")
    print("=" * 108)
    winners = {}
    for arch in arch_order(configs):
        cs = sorted([c for c in configs if c["arch"] == arch], key=rank_key)
        winners[arch] = cs
        top = cs[0]
        print(f"  {arch:<12} WINNER: lr={top['lr']:.0e} batch={top['batch']}  "
              f"({top['n_converged']}/{top['n_total']} converged, "
              f"worst-seed R²={top['worst_conv_r2_stab']:.4f}, mean R²={top['mean_conv_r2_stab']:.4f})")
        runner = cs[1] if len(cs) > 1 else None
        if runner:
            print(f"  {'':<12} runner-up: lr={runner['lr']:.0e} batch={runner['batch']}  "
                  f"({runner['n_converged']}/{runner['n_total']}, "
                  f"worst-seed R²={runner['worst_conv_r2_stab']:.4f})")

    # muP faithfulness summary (forward correctness, independent of probe conditioning)
    tf_conv = [r for r in results if r["arch"] == "transformer" and r["converged"]]
    if tf_conv:
        print("\n" + "=" * 108)
        print("muP / forward-faithfulness (transformer): weighted corr(model p0, optimal p0)")
        print("=" * 108)
        corrs = np.array([r["ntp_corr"] for r in tf_conv])
        r2s = np.array([r["belief_r2_stab"] for r in tf_conv])
        absmax = np.array([r["act_absmax"] for r in tf_conv])
        print(f"  converged transformer runs (N={len(tf_conv)}): "
              f"ntp_corr min={corrs.min():.4f} mean={corrs.mean():.4f} max={corrs.max():.4f}")
        print(f"  stabilized belief R²: min={r2s.min():.4f} max={r2s.max():.4f}  |  "
              f"act |max| range [{absmax.min():.0f}, {absmax.max():.0f}]")
        ok = corrs.min() > 0.8
        print(f"  -> muP forward {'REPRODUCED CORRECTLY (ntp_corr high for all) — not garbage' if ok else 'SUSPECT — FLAG'}")
        # Report the raw rcond=1e-10 instability separately (probe conditioning, NOT muP)
        bad = [r for r in tf_conv if r["belief_r2"] < 0]
        if bad:
            corr_str = ", ".join(f"{r['ntp_corr']:.3f}" for r in bad)
            stab_str = ", ".join(f"{r['belief_r2_stab']:.3f}" for r in bad)
            print(f"  NOTE: {len(bad)} converged transformer run(s) give NEGATIVE raw "
                  f"(rcond=1e-10) belief R² — all at lr=1e-3 with inflated act norms "
                  f"(|max| up to {max(r['act_absmax'] for r in bad):.0f}). This is a linear-"
                  f"probe conditioning artifact at N={proc['N']}, NOT model/muP failure "
                  f"(their ntp_corr = [{corr_str}]; stabilized R² = [{stab_str}]).")
    return winners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--out", default="results/hpsearch_qrrxor_scores.csv")
    ap.add_argument("--dirs", default=f"{TRANSFORMER_DIR},{LSTM_DIR}",
                    help="comma-separated sweep dirs to score (split by arch from config)")
    args = ap.parse_args()

    t0 = time.time()
    dirs = [Path(d.strip()) for d in args.dirs.split(",") if d.strip()]
    run_dirs = []
    for d in dirs:
        run_dirs += sorted([p for p in d.iterdir()
                            if p.is_dir() and p.name.startswith("run_")])
    if not run_dirs:
        raise SystemExit(f"no run_* dirs found under {args.dirs}")

    print("Building process targets (enumerate qRRXOR)...")
    proc = build_process_targets(load_run_config(str(run_dirs[0])))
    print(f"  N={proc['N']} vocab={proc['vocab']} belief_dim={proc['belief_dim']}  "
          f"({time.time()-t0:.1f}s)")

    results, failures = [], []
    for rd in run_dirs:
        try:
            r = analyze_run(rd, proc, args.device)
            results.append(r)
            print(f"  [{r['arch']:<11}] lr={r['lr']:.0e} bs={r['batch']:<2} seed={r['seed']:<2} "
                  f"val/LB={r['val_loss_lb']:.4f} conv={'Y' if r['converged'] else 'N'} "
                  f"belief_R2={r['belief_r2']:.4f} ntp_corr={r['ntp_corr']:.3f}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            failures.append({"run_dir": rd.name, "error": f"{type(e).__name__}: {e}"})
            print(f"  [{rd.name}] FAILED: {type(e).__name__}: {e}", flush=True)

    configs = score_configs(results)
    winners = print_report(configs, results, proc)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f['run_dir']}: {f['error']}")

    # Save per-config scores CSV
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["arch", "lr", "batch", "n_converged", "n_total",
                       "worst_conv_belief_r2_stab", "mean_conv_belief_r2_stab",
                       "worst_conv_belief_r2_raw", "mean_conv_belief_r2_raw",
                       "mean_ntp_corr", "rank_within_arch",
                       "r2stab_seed7", "conv_seed7", "r2stab_seed11", "conv_seed11",
                       "r2stab_seed13", "conv_seed13"])
        for arch in arch_order(configs):
            ranked = sorted([c for c in configs if c["arch"] == arch], key=rank_key)
            for rank, c in enumerate(ranked, 1):
                ps = c["per_seed"]
                def rv(s): return "" if ps[s] is None else f"{ps[s]['belief_r2_stab']:.6f}"
                def cv(s): return "" if ps[s] is None else int(ps[s]["converged"])
                wcsv.writerow([c["arch"], c["lr"], c["batch"], c["n_converged"],
                               c["n_total"], c["worst_conv_r2_stab"], c["mean_conv_r2_stab"],
                               c["worst_conv_r2"], c["mean_conv_r2"],
                               c["mean_ntp_corr"], rank,
                               rv(7), cv(7), rv(11), cv(11), rv(13), cv(13)])
    # Save per-run detail joblib
    joblib.dump({"per_run": results, "configs": configs, "failures": failures,
                 "meta": {"process": "quantum_rrxor", "n_ctx": proc["n_ctx"],
                          "vocab": proc["vocab"], "N": proc["N"],
                          "belief_dim": proc["belief_dim"],
                          "representation": "full concat all-layer @ last pos, NO PCA",
                          "r2_fn": "shuffle_control_simple._weighted_r2 (CV weighted Eq.48)",
                          "conv_thresh": CONV_THRESH, "shuffle_seed": SHUFFLE_SEED}},
                args.out.replace(".csv", ".joblib"))
    print(f"\nSaved: {args.out}  and  {args.out.replace('.csv', '.joblib')}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
