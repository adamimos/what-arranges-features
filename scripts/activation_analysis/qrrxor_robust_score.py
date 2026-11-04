"""
Score the qRRXOR ROBUST report runs (lr=3e-4/bs64, 20k epochs, 5 seeds) with the
§6 beyond-NTP dissociation, and compare to the FRAGILE submission config
(lr=1e-4/bs16). Reuses the hpsearch_qrrxor_score / bloch machinery.

Targets (all from enumerate_process_data, same weights, aligned rows):
  belief    = minimal/GHMM belief states
  shuffled  = geometric-shuffle target (preserves NTP/col(E) component, destroys
              the beyond-NTP ker(Eᵀ) component)  -> proc["shuffled"]
  ntp_proj  = belief projected onto col(E): b_mean + (belief - b_mean) @ P_E
              (a second, classical / NTP-only baseline)

Representation = full concat all-layer @ last pos, NO PCA (transformer resid_pre×4,
LSTM h_n×4). R² = CV weighted R² (Eq.48), 5-fold, fixed-seed row permutation.
Primary probe rcond = 1e-3 (stabilized; raw 1e-10 can be ill-conditioned on the
1024-context qRRXOR set); raw also reported.

beyond-NTP gap = belief R² − shuffled R².

Groups scored:
  transformer/robust  = results/qrrxor_robust_transformer   (seeds 42,123,456,789,1000)
  LSTM/robust         = results/qrrxor_robust_lstm
  transformer/fragile = results/20260424220339              (submission lr=1e-4/bs16, 3 seeds)
  LSTM/fragile        = results/20260503215800_lstm_default_cpu

Usage:
    PYTHONPATH=. uv run python scripts/activation_analysis/qrrxor_robust_score.py --device cuda:4
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from scripts.activation_analysis.run_shuffle_analysis import (
    load_run_config, list_checkpoints,
)
from scripts.activation_analysis.enumerate_process import enumerate_process_data
from scripts.activation_analysis.shuffle_control_simple import _weighted_r2
from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from scripts.activation_analysis.hpsearch_qrrxor_score import (
    load_rnn, extract_transformer, extract_rnn, weighted_corr, final_val_lb,
    N_FOLDS, RCOND, RCOND_STAB, CONV_THRESH, SHUFFLE_SEED,
)

GROUPS = [
    ("transformer", "robust",  "results/qrrxor_robust_transformer"),
    ("LSTM",        "robust",  "results/qrrxor_robust_lstm"),
    ("transformer", "fragile", "results/20260424220339"),
    ("LSTM",        "fragile", "results/20260503215800_lstm_default_cpu"),
]


def build_targets(cfg):
    proc = enumerate_process_data(cfg)
    n_ctx = cfg["model_config"]["n_ctx"]
    contexts = proc["seqs"].numpy()[:, :n_ctx]
    T = get_matrix_from_args(**cfg["process_config"])
    vocab = int(T.shape[0]); assert vocab == 2, f"expected vocab 2, got {vocab}"
    w = proc["weights"]; bel = proc["beliefs"]
    # ntp_proj: belief projected onto col(E) (NTP-relevant subspace)
    wn = w / w.sum()
    b_mean = (bel * wn[:, None]).sum(axis=0)
    ntp_proj = b_mean + (bel - b_mean) @ proc["P_E"]
    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(len(w))
    return {
        "contexts": contexts, "n_ctx": n_ctx, "vocab": vocab,
        "belief": bel, "shuffled": proc["shuffled"], "ntp_proj": ntp_proj,
        "ntp_opt": proc["ntp"][:, 0], "weights": w, "perm": perm,
        "belief_dim": int(bel.shape[1]), "N": int(len(w)),
    }


def r2(acts, tgt, proc, rcond):
    p = proc["perm"]; w = proc["weights"]
    return _weighted_r2(acts[p], tgt[p], w[p], n_folds=N_FOLDS, rcond=rcond)


def analyze_run(run_dir: Path, config_label, arch, proc, device):
    cfg = load_run_config(str(run_dir))
    seed = cfg["model_config"].get("seed", "?")
    val_lb = final_val_lb(run_dir)
    ckpt = list_checkpoints(str(run_dir))[-1]
    if arch == "transformer":
        acts, p0 = extract_transformer(run_dir, ckpt, proc["contexts"], device)
    else:
        model = load_rnn(run_dir, ckpt, device, proc["vocab"])
        acts, p0 = extract_rnn(model, proc["contexts"], proc["vocab"], device)
        del model
    bel_stab = r2(acts, proc["belief"], proc, RCOND_STAB)
    bel_raw = r2(acts, proc["belief"], proc, RCOND)
    shuf_stab = r2(acts, proc["shuffled"], proc, RCOND_STAB)
    ntpp_stab = r2(acts, proc["ntp_proj"], proc, RCOND_STAB)
    return {
        "arch": arch, "config": config_label, "seed": seed,
        "run_dir": run_dir.name, "val_loss_lb": val_lb,
        "converged": bool(val_lb < CONV_THRESH),
        "tokens_seen": int(ckpt.replace(".pt", "")),
        "act_absmax": float(np.abs(acts).max()),
        "belief_r2": float(bel_stab), "belief_r2_raw": float(bel_raw),
        "shuffled_r2": float(shuf_stab), "ntp_proj_r2": float(ntpp_stab),
        "gap": float(bel_stab - shuf_stab),
        "ntp_corr": float(weighted_corr(p0, proc["ntp_opt"], proc["weights"])),
    }


def t_interval(vals, alpha=0.05):
    from scipy import stats
    v = np.asarray(vals, float); n = len(v); m = float(v.mean())
    if n < 2:
        return m, float("nan")
    sd = float(v.std(ddof=1))
    return m, float(stats.t.ppf(1 - alpha / 2, df=n - 1) * sd / np.sqrt(n))


def _msd(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def aggregate(results):
    groups = {}
    for arch, cfg_label, _ in GROUPS:
        runs = [r for r in results if r["arch"] == arch and r["config"] == cfg_label]
        conv = [r for r in runs if r["converged"]]
        e = {"arch": arch, "config": cfg_label, "n_total": len(runs),
             "n_converged": len(conv),
             "failed_seeds": [r["seed"] for r in runs if not r["converged"]]}
        if conv:
            e["belief_mean"], e["belief_sd"] = _msd([r["belief_r2"] for r in conv])
            e["shuffled_mean"], e["shuffled_sd"] = _msd([r["shuffled_r2"] for r in conv])
            e["ntp_proj_mean"], e["ntp_proj_sd"] = _msd([r["ntp_proj_r2"] for r in conv])
            gap = [r["gap"] for r in conv]
            e["gap_min"] = float(min(gap))
            e["gap_mean"], e["gap_ci95_hw"] = t_interval(gap)
            e["ntp_corr_min"] = float(min(r["ntp_corr"] for r in conv))
        groups[(arch, cfg_label)] = e
    return groups


def _mpm(m, sd):
    return f"{m:.4f}" if np.isnan(sd) else f"{m:.4f}±{sd:.4f}"


def _gci(e):
    return (f"{e['gap_mean']:.4f}(n<2)" if np.isnan(e["gap_ci95_hw"])
            else f"{e['gap_mean']:.4f}±{e['gap_ci95_hw']:.4f}")


def print_report(results, groups, proc):
    print("\n" + "=" * 112)
    print("qRRXOR ROBUST-CONFIG SCORES — §6 beyond-NTP dissociation (belief vs shuffled), "
          "full concat NO-PCA, CV weighted R²")
    print(f"process=quantum_rrxor(phi=pi/2,theta=pi/4,eps=0.02) n_ctx={proc['n_ctx']} "
          f"vocab={proc['vocab']} N={proc['N']} belief_dim={proc['belief_dim']} "
          f"probe rcond={RCOND_STAB} conv<{CONV_THRESH}")
    print("=" * 112)
    print(f"{'arch':<12} {'config':>7} {'seed':>5} {'conv':>4} "
          f"{'belief':>8} {'shuffled':>9} {'ntp_proj':>9} {'gap':>8} "
          f"{'raw_bel':>9} {'ntp_corr':>8} {'val/LB':>8}")
    print("-" * 112)
    order = {("transformer", "robust"): 0, ("LSTM", "robust"): 1,
             ("transformer", "fragile"): 2, ("LSTM", "fragile"): 3}
    for r in sorted(results, key=lambda r: (order[(r["arch"], r["config"])], str(r["seed"]))):
        print(f"{r['arch']:<12} {r['config']:>7} {str(r['seed']):>5} "
              f"{'Y' if r['converged'] else 'N':>4} "
              f"{r['belief_r2']:>8.4f} {r['shuffled_r2']:>9.4f} {r['ntp_proj_r2']:>9.4f} "
              f"{r['gap']:>8.4f} {r['belief_r2_raw']:>9.3f} {r['ntp_corr']:>8.3f} "
              f"{r['val_loss_lb']:>8.4f}")

    print("\n" + "=" * 112)
    print("PER-ARCHITECTURE × CONFIG SUMMARY (converged runs only)")
    print("=" * 112)
    print(f"{'arch':<12} {'config':>7} {'Nconv':>6} {'belief_R2':>16} {'shuffled_R2':>16} "
          f"{'ntp_proj_R2':>16} {'gap_min':>8} {'gap_mean±95%t':>18}")
    print("-" * 112)
    for arch, cfg_label, _ in GROUPS:
        e = groups[(arch, cfg_label)]
        if e["n_converged"] == 0:
            print(f"{arch:<12} {cfg_label:>7} {e['n_converged']:>2}/{e['n_total']:<2}  "
                  f"(no converged runs; failed seeds {e['failed_seeds']})")
            continue
        print(f"{arch:<12} {cfg_label:>7} {e['n_converged']:>2}/{e['n_total']:<3} "
              f"{_mpm(e['belief_mean'], e['belief_sd']):>16} "
              f"{_mpm(e['shuffled_mean'], e['shuffled_sd']):>16} "
              f"{_mpm(e['ntp_proj_mean'], e['ntp_proj_sd']):>16} "
              f"{e['gap_min']:>8.4f} {_gci(e):>18}")
        if e["failed_seeds"]:
            print(f"{'':<20} failed seeds: {e['failed_seeds']}")

    print("\n" + "=" * 112)
    print("ROBUST vs FRAGILE (submission lr=1e-4/bs16) — did the robust hyperparams help?")
    print("=" * 112)
    for arch in ["transformer", "LSTM"]:
        rob = groups[(arch, "robust")]; fra = groups[(arch, "fragile")]
        rb = f"{rob['belief_mean']:.4f}" if rob["n_converged"] else "n/a"
        fb = f"{fra['belief_mean']:.4f}" if fra["n_converged"] else "n/a"
        print(f"  {arch:<12} convergence: robust {rob['n_converged']}/{rob['n_total']}  "
              f"vs fragile {fra['n_converged']}/{fra['n_total']}   |   "
              f"mean belief R²: robust {rb}  vs fragile {fb}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:4")
    ap.add_argument("--out", default="results/qrrxor_robust_scores.csv")
    args = ap.parse_args()

    t0 = time.time()
    ref = load_run_config(f"{GROUPS[0][2]}/run_0_L4_H4_DH16_DM64_quantum_rrxor")
    print("Building targets (enumerate qRRXOR + shuffled + ntp_proj)...")
    proc = build_targets(ref)
    print(f"  N={proc['N']} vocab={proc['vocab']} belief_dim={proc['belief_dim']}  "
          f"({time.time()-t0:.1f}s)")

    results, failures = [], []
    for arch, cfg_label, dpath in GROUPS:
        d = Path(dpath)
        if not d.exists():
            print(f"  [{arch}/{cfg_label}] dir missing: {dpath}")
            continue
        for rd in sorted([p for p in d.iterdir()
                          if p.is_dir() and p.name.startswith("run_")]):
            try:
                r = analyze_run(rd, cfg_label, arch, proc, args.device)
                results.append(r)
                print(f"  [{arch:<11}/{cfg_label:<7}] seed={str(r['seed']):<5} "
                      f"conv={'Y' if r['converged'] else 'N'} "
                      f"belief={r['belief_r2']:.4f} shuffled={r['shuffled_r2']:.4f} "
                      f"gap={r['gap']:.4f} ntp_corr={r['ntp_corr']:.3f}", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                failures.append({"run_dir": f"{dpath}/{rd.name}", "error": f"{type(e).__name__}: {e}"})
                print(f"  [{rd.name}] FAILED: {type(e).__name__}: {e}", flush=True)

    groups = aggregate(results)
    print_report(results, groups, proc)
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f['run_dir']}: {f['error']}")

    # Save — primary CSV = the 10 robust runs; joblib holds everything incl. fragile.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        wc = csv.writer(fh)
        wc.writerow(["arch", "config", "seed", "run_dir", "val_loss_lb", "converged",
                     "belief_r2", "belief_r2_raw", "shuffled_r2", "ntp_proj_r2", "gap",
                     "ntp_corr", "act_absmax"])
        for r in sorted(results, key=lambda r: (r["config"] != "robust", r["arch"], str(r["seed"]))):
            wc.writerow([r["arch"], r["config"], r["seed"], r["run_dir"], r["val_loss_lb"],
                         r["converged"], r["belief_r2"], r["belief_r2_raw"], r["shuffled_r2"],
                         r["ntp_proj_r2"], r["gap"], r["ntp_corr"], r["act_absmax"]])
    joblib.dump({"per_run": results,
                 "groups": {f"{k[0]}/{k[1]}": v for k, v in groups.items()},
                 "failures": failures,
                 "meta": {"process": "quantum_rrxor", "n_ctx": proc["n_ctx"],
                          "vocab": proc["vocab"], "N": proc["N"],
                          "belief_dim": proc["belief_dim"], "probe_rcond": RCOND_STAB,
                          "representation": "full concat all-layer @ last pos, NO PCA",
                          "conv_thresh": CONV_THRESH, "shuffle_seed": SHUFFLE_SEED}},
                args.out.replace(".csv", ".joblib"))
    print(f"\nSaved: {args.out}  and  {args.out.replace('.csv', '.joblib')}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
