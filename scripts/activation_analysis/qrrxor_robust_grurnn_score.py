"""
Score the GRU + vanilla-RNN qRRXOR ROBUST report runs (20k, 5 seeds each), exactly
like the transformer+LSTM robust scoring (qrrxor_robust_score.py), then produce the
CONSOLIDATED 4-architecture qRRXOR table by merging with the earlier transformer+LSTM
robust results (results/qrrxor_robust_scores.csv).

  GRU  robust @ lr=3e-4/bs64  -> results/qrrxor_robust_gru
  RNN  robust @ lr=3e-5/bs64  -> results/qrrxor_robust_rnn

Per run at FINAL checkpoint: convergence (val_loss_mean<1.005), belief R²
(stabilized rcond=1e-3 CV weighted R², no-PCA concat h_n×4), shuffled R²
(geometric-shuffle target), gap = belief − shuffled. Aggregate per arch over
converged runs: N/5, mean belief R² (±sd), mean shuffled R² (±sd), paired gap
min-over-seeds + mean ± 95% t (df = N_conv−1).

Usage:
    PYTHONPATH=. uv run python scripts/activation_analysis/qrrxor_robust_grurnn_score.py --device cuda:5
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import joblib
import numpy as np

from scripts.activation_analysis.run_shuffle_analysis import load_run_config
from scripts.activation_analysis.qrrxor_robust_score import (
    build_targets, analyze_run, t_interval, _msd, RCOND_STAB, CONV_THRESH,
)

# arch -> (robust dir, lr, batch)
NEW_GROUPS = [
    ("GRU", "results/qrrxor_robust_gru", "3e-4", 64),
    ("RNN", "results/qrrxor_robust_rnn", "3e-5", 64),
]
MERGE_CSV = "results/qrrxor_robust_scores.csv"   # earlier transformer+LSTM robust
ARCH_LR = {"transformer": ("3e-4", 64), "LSTM": ("3e-4", 64),
           "GRU": ("3e-4", 64), "RNN": ("3e-5", 64)}


def agg_arch(runs):
    """Aggregate a list of per-run dicts (keys: belief_r2, shuffled_r2, gap,
    converged, seed) for one architecture over its converged runs."""
    conv = [r for r in runs if r["converged"]]
    e = {"n_total": len(runs), "n_converged": len(conv),
         "failed_seeds": [r["seed"] for r in runs if not r["converged"]]}
    if conv:
        e["belief_mean"], e["belief_sd"] = _msd([r["belief_r2"] for r in conv])
        e["shuffled_mean"], e["shuffled_sd"] = _msd([r["shuffled_r2"] for r in conv])
        gap = [r["gap"] for r in conv]
        e["gap_min"] = float(min(gap))
        e["gap_mean"], e["gap_ci95_hw"] = t_interval(gap)
    return e


def _mpm(m, sd):
    return f"{m:.4f}" if np.isnan(sd) else f"{m:.4f}±{sd:.4f}"


def _gci(e):
    return (f"{e['gap_mean']:.4f}(n<2)" if np.isnan(e["gap_ci95_hw"])
            else f"{e['gap_mean']:.4f}±{e['gap_ci95_hw']:.4f}")


def load_merge_rows(path):
    """Return per-run dicts (transformer+LSTM robust) from earlier CSV."""
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["config"] != "robust":
                continue
            rows.append({
                "arch": r["arch"], "seed": r["seed"],
                "converged": r["converged"] == "True",
                "belief_r2": float(r["belief_r2"]),
                "shuffled_r2": float(r["shuffled_r2"]),
                "gap": float(r["gap"]),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--out", default="results/qrrxor_robust_grurnn_scores.csv")
    ap.add_argument("--merge_csv", default=MERGE_CSV)
    args = ap.parse_args()

    t0 = time.time()
    # process targets (arch-agnostic); build from first GRU run
    first_run = sorted([p for p in Path(NEW_GROUPS[0][1]).iterdir()
                        if p.is_dir() and p.name.startswith("run_")])[0]
    proc = build_targets(load_run_config(str(first_run)))
    print(f"targets: N={proc['N']} vocab={proc['vocab']} belief_dim={proc['belief_dim']} "
          f"({time.time()-t0:.1f}s)")

    # Score GRU + RNN robust
    new_runs, failures = [], []
    for arch, dpath, lr, bs in NEW_GROUPS:
        for rd in sorted([p for p in Path(dpath).iterdir()
                          if p.is_dir() and p.name.startswith("run_")]):
            try:
                r = analyze_run(rd, "robust", arch, proc, args.device)
                new_runs.append(r)
                print(f"  [{arch:<3}/robust] seed={str(r['seed']):<5} "
                      f"conv={'Y' if r['converged'] else 'N'} "
                      f"belief={r['belief_r2']:.4f} shuffled={r['shuffled_r2']:.4f} "
                      f"gap={r['gap']:.4f} ntp_corr={r['ntp_corr']:.3f} "
                      f"val/LB={r['val_loss_lb']:.4f}", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                failures.append({"run_dir": f"{dpath}/{rd.name}", "error": str(e)})

    # ---- GRU/RNN per-arch summary ----
    print("\n" + "=" * 100)
    print("GRU / vanilla-RNN ROBUST report runs — per-arch summary (converged only)")
    print("=" * 100)
    print(f"{'arch':<6} {'lr/bs':>9} {'Nconv':>6} {'belief_R2':>16} {'shuffled_R2':>16} "
          f"{'gap_min':>8} {'gap_mean±95%t':>18}")
    print("-" * 100)
    grurnn_agg = {}
    for arch, _, lr, bs in NEW_GROUPS:
        runs = [r for r in new_runs if r["arch"] == arch]
        e = agg_arch(runs); grurnn_agg[arch] = e
        if e["n_converged"] == 0:
            print(f"{arch:<6} {lr+'/'+str(bs):>9} {e['n_converged']}/{e['n_total']}  (none converged)")
            continue
        print(f"{arch:<6} {lr+'/'+str(bs):>9} {e['n_converged']:>2}/{e['n_total']:<3} "
              f"{_mpm(e['belief_mean'], e['belief_sd']):>16} "
              f"{_mpm(e['shuffled_mean'], e['shuffled_sd']):>16} "
              f"{e['gap_min']:>8.4f} {_gci(e):>18}")
        if e["failed_seeds"]:
            print(f"       failed seeds: {e['failed_seeds']}")

    # ---- CONSOLIDATED 4-arch table ----
    merge_rows = load_merge_rows(args.merge_csv)
    all_rows = merge_rows + [{"arch": r["arch"], "seed": r["seed"],
                              "converged": r["converged"], "belief_r2": r["belief_r2"],
                              "shuffled_r2": r["shuffled_r2"], "gap": r["gap"]}
                             for r in new_runs]
    consolidated = {}
    print("\n" + "=" * 108)
    print("CONSOLIDATED 4-ARCHITECTURE qRRXOR ROBUST TABLE (belief vs shuffled, §6 dissociation)")
    print(f"stabilized rcond={RCOND_STAB} CV weighted R², no-PCA concat; conv<{CONV_THRESH}; 5 seeds/arch")
    print("=" * 108)
    print(f"{'arch':<13} {'lr/bs':>9} {'Nconv':>6} {'belief_R2':>16} {'shuffled_R2':>16} "
          f"{'gap_min':>8} {'gap_mean±95%t':>18}")
    print("-" * 108)
    label = {"transformer": "transformer", "LSTM": "LSTM", "GRU": "GRU", "RNN": "vanilla_RNN"}
    for arch in ["transformer", "LSTM", "GRU", "RNN"]:
        runs = [r for r in all_rows if r["arch"] == arch]
        e = agg_arch(runs); consolidated[arch] = e
        lr, bs = ARCH_LR[arch]
        if e["n_converged"] == 0:
            print(f"{label[arch]:<13} {lr+'/'+str(bs):>9} {e['n_converged']}/{e['n_total']} (none)")
            continue
        print(f"{label[arch]:<13} {lr+'/'+str(bs):>9} {e['n_converged']:>2}/{e['n_total']:<3} "
              f"{_mpm(e['belief_mean'], e['belief_sd']):>16} "
              f"{_mpm(e['shuffled_mean'], e['shuffled_sd']):>16} "
              f"{e['gap_min']:>8.4f} {_gci(e):>18}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" ", f)

    # ---- Save ----
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # (1) GRU+RNN per-run rows
    with open(args.out, "w", newline="") as fh:
        wc = csv.writer(fh)
        wc.writerow(["arch", "config", "seed", "run_dir", "val_loss_lb", "converged",
                     "belief_r2", "belief_r2_raw", "shuffled_r2", "ntp_proj_r2", "gap",
                     "ntp_corr", "act_absmax"])
        for r in sorted(new_runs, key=lambda r: (r["arch"], str(r["seed"]))):
            wc.writerow([r["arch"], r["config"], r["seed"], r["run_dir"], r["val_loss_lb"],
                         r["converged"], r["belief_r2"], r["belief_r2_raw"], r["shuffled_r2"],
                         r["ntp_proj_r2"], r["gap"], r["ntp_corr"], r["act_absmax"]])
    # (2) consolidated 4-arch summary
    cons_path = args.out.replace(".csv", "_consolidated.csv")
    with open(cons_path, "w", newline="") as fh:
        wc = csv.writer(fh)
        wc.writerow(["arch", "lr", "batch", "n_converged", "n_total",
                     "belief_r2_mean", "belief_r2_sd", "shuffled_r2_mean", "shuffled_r2_sd",
                     "gap_min", "gap_mean", "gap_ci95_halfwidth"])
        for arch in ["transformer", "LSTM", "GRU", "RNN"]:
            e = consolidated[arch]; lr, bs = ARCH_LR[arch]
            if e["n_converged"] == 0:
                wc.writerow([label[arch], lr, bs, 0, e["n_total"], "", "", "", "", "", "", ""])
                continue
            wc.writerow([label[arch], lr, bs, e["n_converged"], e["n_total"],
                         e["belief_mean"], e["belief_sd"], e["shuffled_mean"], e["shuffled_sd"],
                         e["gap_min"], e["gap_mean"], e["gap_ci95_hw"]])
    joblib.dump({"grurnn_per_run": new_runs, "grurnn_agg": grurnn_agg,
                 "consolidated": consolidated, "failures": failures},
                args.out.replace(".csv", ".joblib"))
    print(f"\nSaved: {args.out} , {cons_path} , {args.out.replace('.csv', '.joblib')}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
