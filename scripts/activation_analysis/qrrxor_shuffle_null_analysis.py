"""
FAITHFUL §6 / App-D.5 geometric-shuffle control for the qRRXOR robust report runs
(4 archs × 5 seeds). Reproduces the paper's EXACT probe/pipeline so numbers line up
with the submitted hero (~0.98 belief / ~0.17 shuffle):

  probe        = _paper_regression_r2 (SVD pinv, 10-fold CV, rcond sweep)   [shuffle_control.py]
  preprocessing= PCA-0.95 on activations (_maybe_pca)                        [run_shuffle_analysis.py]
  representation = concat all-layer @ last pos (transformer resid_pre×4 / RNN h_n×4)
  shuffle      = geometric_shuffle (preserve col(E) NTP part, permute ker(Eᵀ) perp part)
  n_shuffles   = 200  (== run_shuffle_analysis.N_SHUFFLES)  -> null mean/std + p-value

Per run reports:
  acts_to_beliefs     R² (PCA, paper probe)            — the belief probe
  perp_frac           R²(acts → Δb_perp)               — fraction of BEYOND-NTP (ker Eᵀ)
                                                          variance carried (the §6 ~81–98%)
  shuffle null (200)  mean ± std, max, and p_value = frac(null R² ≥ belief R²)
  extra curves        beliefs_to_shuffled, random_to_beliefs, ntp_to_beliefs
  reconciliation      shuffle R² under {single|null-mean} × {PCA|no-PCA} + my earlier
                      (no-PCA, single, _weighted_r2) number.

Usage:
    PYTHONPATH=. uv run python scripts/activation_analysis/qrrxor_shuffle_null_analysis.py --device cuda:3
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import os
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from scripts.activation_analysis.run_shuffle_analysis import (
    load_run_config, list_checkpoints, load_model, load_random_model, _maybe_pca,
)
from scripts.activation_analysis.enumerate_process import enumerate_process_data
from scripts.activation_analysis.shuffle_control import (
    _paper_regression_r2, geometric_shuffle, decompose_beliefs,
)
from scripts.activation_analysis.shuffle_control_simple import _weighted_r2
from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.training.networks import RNNWrapper

PCA_RANK = 0.95
N_SHUFFLES = 200          # == run_shuffle_analysis.N_SHUFFLES (PCA null, primary)
N_SHUFFLES_NOPCA = 20     # cheaper no-PCA null (mean is stable) for reconciliation
SHUFFLE_SEED = 42         # == prepare/enumerate default; first draw == proc["shuffled"]
CONV_THRESH = 1.005

GROUPS = [
    ("transformer", "results/qrrxor_robust_transformer", "3e-4", 64),
    ("LSTM",        "results/qrrxor_robust_lstm",         "3e-4", 64),
    ("GRU",         "results/qrrxor_robust_gru",          "3e-4", 64),
    ("RNN",         "results/qrrxor_robust_rnn",          "3e-5", 64),
]


def final_val_lb(run_dir: Path) -> float:
    f = run_dir / "loss.csv"
    if not f.exists():
        return float("inf")
    import csv as _csv
    with open(f) as fh:
        rows = list(_csv.DictReader(fh))
    return float(rows[-1]["val_loss_mean"]) if rows else float("inf")


# ---- activation extraction (concat all-layer @ last pos) ----
def transformer_concat(model, contexts, device, bs=8192):
    names = [f"blocks.{i}.hook_resid_pre" for i in range(model.cfg.n_layers)]
    inp = torch.from_numpy(contexts).long()
    chunks = {k: [] for k in names}
    with torch.no_grad():
        for s in range(0, inp.shape[0], bs):
            _, cache = model.run_with_cache(inp[s:s + bs].to(device), names_filter=names)
            for k in names:
                chunks[k].append(cache[k][:, -1, :].cpu().numpy())
    return np.concatenate([np.concatenate(chunks[k], 0) for k in names], axis=1)


def build_rnn(cfg, vocab, device, state=None, seed=None):
    mc = cfg["model_config"]
    if seed is not None:
        torch.manual_seed(seed)
    cls = getattr(nn, mc["rnn_type"].upper())
    rnn = cls(input_size=vocab, hidden_size=mc["hidden_size"], num_layers=mc["num_layers"],
              dropout=mc.get("dropout", 0.0), batch_first=True,
              bidirectional=mc.get("bidirectional", False)).to(device)
    out = nn.Linear(mc["hidden_size"], vocab).to(device)
    model = RNNWrapper(rnn, out).to(device)
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return model


def rnn_concat(model, contexts, vocab, device):
    x = torch.from_numpy(contexts).long().to(device)
    oh = torch.zeros(*x.shape, vocab, device=device)
    oh.scatter_(2, x.unsqueeze(-1), 1.0)
    with torch.no_grad():
        _, hidden = model.rnn(oh)
        h_n = hidden[0] if isinstance(hidden, tuple) else hidden
    return torch.cat([h_n[i] for i in range(h_n.shape[0])], dim=1).cpu().numpy()


def get_acts(arch, run_dir, ckpt, proc, device, random=False):
    cfg = load_run_config(str(run_dir))
    if arch == "transformer":
        m = (load_random_model(str(run_dir), device, seed=999) if random
             else load_model(str(run_dir), ckpt, device))
        A = transformer_concat(m, proc["contexts"], device)
        del m
        if "cuda" in str(device):
            torch.cuda.empty_cache()
    else:
        vocab = proc["vocab"]
        if random:
            m = build_rnn(cfg, vocab, device, state=None, seed=999)
        else:
            state = torch.load(f"{run_dir}/{ckpt}", map_location=device, weights_only=True)
            m = build_rnn(cfg, vocab, device, state=state)
        A = rnn_concat(m, proc["contexts"], vocab, device)
        del m
    return A


def analyze_run(arch, run_dir: Path, proc, device):
    cfg = load_run_config(str(run_dir))
    seed = cfg["model_config"].get("seed", "?")
    val_lb = final_val_lb(run_dir)
    ckpt = list_checkpoints(str(run_dir))[-1]

    A = get_acts(arch, run_dir, ckpt, proc, device, random=False)
    A_rand = get_acts(arch, run_dir, ckpt, proc, device, random=True)
    A_pca = _maybe_pca(A, PCA_RANK)[0]
    A_rand_pca = _maybe_pca(A_rand, PCA_RANK)[0]

    bel = proc["belief"]; w = proc["weights"]
    # beyond-NTP (ker Eᵀ) component of belief (unweighted mean, matching geometric_shuffle)
    _, _, delta_perp = decompose_beliefs(bel, proc["P_E"], proc["P_E_perp"])

    dev = device if "cuda" in str(device) else "cpu"
    def R2p(X, Y):   # paper probe (PCA already applied to X where relevant)
        return _paper_regression_r2(X, Y, w, device=dev)

    with contextlib.redirect_stdout(open(os.devnull, "w")):
        # --- headline paper-method quantities (PCA + paper probe) ---
        belief_pca = R2p(A_pca, bel)
        perp_frac = R2p(A_pca, delta_perp)             # fraction of beyond-NTP variance (§6 ~81–98%)
        single_pca = R2p(A_pca, proc["shuffled"])      # single-shuffle, PCA, paper
        random_to_bel = R2p(A_rand_pca, bel)
        beliefs_to_shuffled = R2p(bel, proc["shuffled"])   # theoretical control
        ntp_to_bel = R2p(proc["ntp"], bel)

        # --- 200-shuffle PCA null (primary) ---
        rng = np.random.default_rng(SHUFFLE_SEED)
        null_pca = np.array([R2p(A_pca, geometric_shuffle(bel, proc["P_E"], proc["P_E_perp"], rng))
                             for _ in range(N_SHUFFLES)])
        p_value = float(np.mean(null_pca >= belief_pca))

        # --- reconciliation: no-PCA variants ---
        single_nopca_weighted = _weighted_r2(A[proc["perm"]], proc["shuffled"][proc["perm"]],
                                             w[proc["perm"]], n_folds=5, rcond=1e-10)  # my earlier method
        rng2 = np.random.default_rng(SHUFFLE_SEED)
        null_nopca = np.array([R2p(A, geometric_shuffle(bel, proc["P_E"], proc["P_E_perp"], rng2))
                               for _ in range(N_SHUFFLES_NOPCA)])

    return {
        "arch": arch, "seed": seed, "run_dir": run_dir.name,
        "val_loss_lb": val_lb, "converged": bool(val_lb < CONV_THRESH),
        "d_pca": int(A_pca.shape[1]), "concat_dim": int(A.shape[1]),
        "belief_r2_pca": float(belief_pca),
        "perp_frac": float(perp_frac),
        "shuffle_null_mean_pca": float(null_pca.mean()),
        "shuffle_null_std_pca": float(null_pca.std()),
        "shuffle_null_max_pca": float(null_pca.max()),
        "p_value": p_value,
        "single_shuffle_pca": float(single_pca),
        "shuffle_null_mean_nopca": float(null_nopca.mean()),
        "shuffle_null_std_nopca": float(null_nopca.std()),
        "single_shuffle_nopca_weighted": float(single_nopca_weighted),
        "random_to_beliefs": float(random_to_bel),
        "beliefs_to_shuffled": float(beliefs_to_shuffled),
        "ntp_to_beliefs": float(ntp_to_bel),
    }


def _msd(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def _mpm(m, sd):
    return f"{m:.4f}" if np.isnan(sd) else f"{m:.4f}±{sd:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--out", default="results/qrrxor_shuffle_null_analysis.csv")
    args = ap.parse_args()
    t0 = time.time()

    first = sorted([p for p in Path(GROUPS[0][1]).iterdir()
                    if p.is_dir() and p.name.startswith("run_")])[0]
    ref = load_run_config(str(first))
    proc = enumerate_process_data(ref)
    proc["contexts"] = proc["seqs"].numpy()[:, :ref["model_config"]["n_ctx"]]
    proc["belief"] = proc["beliefs"]
    T = get_matrix_from_args(**ref["process_config"])
    proc["vocab"] = int(T.shape[0]); assert proc["vocab"] == 2
    rng = np.random.default_rng(0); proc["perm"] = rng.permutation(len(proc["weights"]))
    # variance diagnostics
    _, dpar, dperp = decompose_beliefs(proc["belief"], proc["P_E"], proc["P_E_perp"])
    var_par = float(np.var(dpar)); var_perp = float(np.var(dperp))
    frac_beyond = var_perp / (var_par + var_perp)
    rankE = int(np.linalg.matrix_rank(proc["E"])); dim_ker = proc["belief"].shape[1] - rankE
    print(f"process: N={len(proc['weights'])} vocab={proc['vocab']} belief_dim={proc['belief'].shape[1]} "
          f"| var_perp/var_total={frac_beyond:.3f} rank(E)={rankE} dim(ker Eᵀ)={dim_ker}  "
          f"n_shuffles(PCA)={N_SHUFFLES}  probe=PCA{PCA_RANK}+_paper_regression_r2  ({time.time()-t0:.1f}s)")

    results, failures = [], []
    for arch, dpath, lr, bs in GROUPS:
        for rd in sorted([p for p in Path(dpath).iterdir()
                          if p.is_dir() and p.name.startswith("run_")]):
            try:
                r = analyze_run(arch, rd, proc, args.device)
                results.append(r)
                print(f"  [{arch:<11} s{str(r['seed']):<5}] belief={r['belief_r2_pca']:.4f} "
                      f"perp_frac={r['perp_frac']:.4f} nullPCA={r['shuffle_null_mean_pca']:.4f}"
                      f"±{r['shuffle_null_std_pca']:.4f} p={r['p_value']:.3f} "
                      f"| nullNoPCA={r['shuffle_null_mean_nopca']:.3f} "
                      f"single_mymethod={r['single_shuffle_nopca_weighted']:.3f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                failures.append({"run_dir": f"{dpath}/{rd.name}", "error": str(e)})

    # ---------------- per-run detail ----------------
    print("\n" + "=" * 118)
    print("FULL 20-RUN DETAIL — paper-method (PCA-0.95 + _paper_regression_r2), 200-shuffle null")
    print("=" * 118)
    print(f"{'arch':<12} {'seed':>5} {'conv':>4} {'belief':>8} {'perp_frac':>9} "
          f"{'null_mean':>9} {'null_std':>8} {'null_max':>8} {'p_val':>6} {'rand→bel':>8} "
          f"{'b→shuf':>7} {'ntp→bel':>7}")
    print("-" * 118)
    ordr = {"transformer": 0, "LSTM": 1, "GRU": 2, "RNN": 3}
    for r in sorted(results, key=lambda r: (ordr[r["arch"]], str(r["seed"]))):
        print(f"{r['arch']:<12} {str(r['seed']):>5} {'Y' if r['converged'] else 'N':>4} "
              f"{r['belief_r2_pca']:>8.4f} {r['perp_frac']:>9.4f} "
              f"{r['shuffle_null_mean_pca']:>9.4f} {r['shuffle_null_std_pca']:>8.4f} "
              f"{r['shuffle_null_max_pca']:>8.4f} {r['p_value']:>6.3f} {r['random_to_beliefs']:>8.4f} "
              f"{r['beliefs_to_shuffled']:>7.4f} {r['ntp_to_beliefs']:>7.4f}")

    # ---------------- per-arch aggregation ----------------
    print("\n" + "=" * 118)
    print("PER-ARCHITECTURE (5 seeds, converged): belief R², beyond-NTP fraction, shuffle null, p-value")
    print("=" * 118)
    print(f"{'arch':<12} {'lr/bs':>8} {'N':>4} {'belief_R2':>15} {'beyond-NTP frac':>17} "
          f"{'shuffle_null_R2':>17} {'max p':>7} {'all p<.005':>10}")
    print("-" * 118)
    agg = {}
    for arch, _, lr, bs in GROUPS:
        rs = [r for r in results if r["arch"] == arch and r["converged"]]
        if not rs:
            continue
        bm, bsd = _msd([r["belief_r2_pca"] for r in rs])
        pm, psd = _msd([r["perp_frac"] for r in rs])
        nm, nsd = _msd([r["shuffle_null_mean_pca"] for r in rs])
        maxp = max(r["p_value"] for r in rs)
        allsig = all(r["p_value"] < 0.005 for r in rs)
        agg[arch] = {"belief": (bm, bsd), "perp": (pm, psd), "null": (nm, nsd),
                     "max_p": maxp, "all_sig": allsig, "n": len(rs)}
        print(f"{arch:<12} {lr+'/'+str(bs):>8} {len(rs):>2}/5 "
              f"{_mpm(bm, bsd):>15} {_mpm(pm, psd):>17} {_mpm(nm, nsd):>17} "
              f"{maxp:>7.3f} {'YES' if allsig else 'NO':>10}")

    # ---------------- reconciliation ----------------
    print("\n" + "=" * 118)
    print("RECONCILIATION — shuffle R² decomposition (concat rep held fixed): why my earlier ≈0 vs paper ~0.17")
    print("=" * 118)
    print(f"{'arch':<12} | {'(1)my:noPCA,single,wtd':>22} | {'(2)noPCA,null-mean':>18} | "
          f"{'(3)PCA,single':>14} | {'(4)PCA,null-mean=PAPER':>22}")
    print("-" * 118)
    for arch, _, _, _ in GROUPS:
        rs = [r for r in results if r["arch"] == arch and r["converged"]]
        if not rs:
            continue
        c1, _ = _msd([r["single_shuffle_nopca_weighted"] for r in rs])
        c2, _ = _msd([r["shuffle_null_mean_nopca"] for r in rs])
        c3, _ = _msd([r["single_shuffle_pca"] for r in rs])
        c4, _ = _msd([r["shuffle_null_mean_pca"] for r in rs])
        print(f"{arch:<12} | {c1:>22.4f} | {c2:>18.4f} | {c3:>14.4f} | {c4:>22.4f}")
    print("\n  factor effects (mean over archs):")
    def col(key):
        return np.mean([_msd([r[key] for r in results
                              if r["arch"] == a and r["converged"]])[0]
                        for a, _, _, _ in GROUPS
                        if any(r["arch"] == a and r["converged"] for r in results)])
    my1 = col("single_shuffle_nopca_weighted"); c2 = col("shuffle_null_mean_nopca")
    c3 = col("single_shuffle_pca"); c4 = col("shuffle_null_mean_pca")
    print(f"    PCA effect (single):   noPCA {my1:+.3f} -> PCA {c3:+.3f}   (Δ={c3-my1:+.3f})")
    print(f"    null-mean effect (PCA): single {c3:+.3f} -> null-mean {c4:+.3f}   (Δ={c4-c3:+.3f})")
    print(f"    null-mean effect (noPCA): single(my) {my1:+.3f} -> null-mean {c2:+.3f}   (Δ={c2-my1:+.3f})")
    print(f"    => PAPER method (PCA, null-mean) shuffle R² = {c4:.3f}  (hero ≈ 0.17)")

    if failures:
        print("\nFAILURES:"); [print(" ", f) for f in failures]

    # ---------------- save ----------------
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cols = ["arch", "seed", "run_dir", "val_loss_lb", "converged", "d_pca", "concat_dim",
            "belief_r2_pca", "perp_frac", "shuffle_null_mean_pca", "shuffle_null_std_pca",
            "shuffle_null_max_pca", "p_value", "single_shuffle_pca",
            "shuffle_null_mean_nopca", "shuffle_null_std_nopca",
            "single_shuffle_nopca_weighted", "random_to_beliefs", "beliefs_to_shuffled",
            "ntp_to_beliefs"]
    with open(args.out, "w", newline="") as fh:
        wc = csv.writer(fh); wc.writerow(cols)
        for r in sorted(results, key=lambda r: (ordr[r["arch"]], str(r["seed"]))):
            wc.writerow([r[c] for c in cols])
    joblib.dump({"per_run": results, "agg": agg, "failures": failures,
                 "meta": {"pca_rank": PCA_RANK, "n_shuffles_pca": N_SHUFFLES,
                          "n_shuffles_nopca": N_SHUFFLES_NOPCA, "probe": "_paper_regression_r2",
                          "var_perp_frac": frac_beyond, "dim_ker_ET": dim_ker,
                          "shuffle_seed": SHUFFLE_SEED}},
                args.out.replace(".csv", ".joblib"))
    print(f"\nSaved: {args.out}  and  {args.out.replace('.csv', '.joblib')}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
