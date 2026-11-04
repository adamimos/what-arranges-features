"""
Representational-claim audit for qRRXOR (paper §6 and App. on interventions) — trained vs UNTRAINED nets,
held-out-context generalization, and emergence over training.

Motivation: causal experiments show the ker(E^T) belief component is causally
inert in the transformer, and an adversarial audit showed an untrained GRU gives
in-sample belief-probe R^2 ~ 0.97. This script decides whether the §6 claim
("transformers linearly represent ~84% of the NTP-invisible belief variance")
is a trained, specific, generalizing structure or a probing artifact.

Three parts (all on the 1024 enumerated n_ctx=10 contexts, probability-weighted):

A. PROTOCOL RECONCILIATION + untrained controls, 5 seeds x {transformer, GRU}:
     - "audit" protocol: full concat dim (256), in-sample weighted OLS
       (rcond=1e-10) -> belief R^2, ker R^2, and a 20-draw geometric-shuffle
       null UNDER THE SAME in-sample protocol (shows decodability is cheap
       and non-specific at full capacity).
     - PAPER protocol (PCA-0.95 + _paper_regression_r2 10-fold rcond CV):
       belief R^2 and perp_frac (ker-target R^2) for trained AND untrained
       nets, plus a 200-draw geometric-shuffle null for the UNTRAINED nets
       (trained-net nulls already in results/qrrxor_shuffle_null_analysis.csv).
       Untrained nets use per-run distinct init seeds (999+idx).

B. HELD-OUT-CONTEXT generalization (the paper probe's final R^2 is in-sample;
   its CV only selects rcond): random half-split of the 1024 contexts,
   PCA fit on train half only, rcond chosen by 5-fold CV within the train
   half, refit on train half, weighted R^2 evaluated on the held-out half.
   Targets: full belief and the ker (delta_perp) component. 3 splits per run,
   trained and untrained.

C. EMERGENCE over training (transformer runs 0-2): ~14 checkpoints from 0.pt
   (the actual init) to final; paper-protocol belief R^2 + perp_frac, one
   held-out-half ker R^2 per checkpoint, and val_loss_mean from loss.csv.

Usage:
    PYTHONPATH=. uv run python scripts/activation_analysis/qrrxor_repr_audit.py \
        --device cuda:0 --parts abc
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

from scripts.activation_analysis.run_shuffle_analysis import (
    load_run_config, list_checkpoints, load_model, load_random_model, _maybe_pca,
)
from scripts.activation_analysis.enumerate_process import enumerate_process_data
from scripts.activation_analysis.shuffle_control import (
    _paper_regression_r2, geometric_shuffle, decompose_beliefs,
)
from scripts.activation_analysis.qrrxor_shuffle_null_analysis import (
    transformer_concat, build_rnn, rnn_concat, final_val_lb,
)
from scripts.activation_analysis.config import RCOND_SWEEP_LIST
from epsilon_transformers.process.transition_matrices import get_matrix_from_args

PCA_RANK = 0.95
N_SHUFFLES_PAPER = 200
N_SHUFFLES_AUDIT = 20
SHUFFLE_SEED = 42
N_SPLITS_HELDOUT = 3

XF_DIR = "results/qrrxor_robust_transformer"
GRU_DIR = "results/qrrxor_robust_gru"


# ---------------------------------------------------------------------------
# weighted R^2 helper (matches _paper_regression_r2's final metric convention)
# ---------------------------------------------------------------------------
def weighted_r2(Y_true, Y_pred, w):
    w = w / w.sum()
    mu = (Y_true * w[:, None]).sum(axis=0)
    ss_res = (w[:, None] * (Y_true - Y_pred) ** 2).sum()
    ss_tot = (w[:, None] * (Y_true - mu) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# "audit" protocol: in-sample weighted OLS on full-dim features (no PCA, no CV)
# ---------------------------------------------------------------------------
def insample_ols_r2(X, Y, w, rcond=1e-10):
    Xs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-8)
    Xb = np.concatenate([np.ones((len(Xs), 1)), Xs], axis=1)
    wn = w / w.sum()
    sw = np.sqrt(wn)[:, None]
    beta, *_ = np.linalg.lstsq(Xb * sw, Y * sw, rcond=rcond)
    return weighted_r2(Y, Xb @ beta, w)


# ---------------------------------------------------------------------------
# held-out-context probe: PCA + standardize + rcond fit on TRAIN half only
# ---------------------------------------------------------------------------
def _svd_beta_sweep(Xb_w, Y_w, rconds):
    """Return {rcond: beta} from one SVD of the weighted design matrix."""
    U, S, Vh = np.linalg.svd(Xb_w, full_matrices=False)
    UtY = U.T @ Y_w
    out = {}
    smax = S[0] if len(S) else 0.0
    for rc in rconds:
        Sinv = np.where(S > rc * smax, 1.0 / np.maximum(S, 1e-300), 0.0)
        out[rc] = Vh.T @ (Sinv[:, None] * UtY)
    return out


def heldout_r2(X, Y, w, train_idx, test_idx, rconds=RCOND_SWEEP_LIST,
               pca_rank=PCA_RANK, inner_folds=5, inner_seed=0):
    """Fit PCA + weighted linear probe on train contexts only; report
    weighted R^2 on the held-out contexts. rcond chosen by inner CV on train."""
    from sklearn.decomposition import PCA
    Xtr_raw, Xte_raw = X[train_idx], X[test_idx]
    Ytr, Yte = Y[train_idx], Y[test_idx]
    wtr, wte = w[train_idx], w[test_idx]

    pca = PCA(n_components=pca_rank, svd_solver="full").fit(Xtr_raw)
    Xtr = pca.transform(Xtr_raw)
    Xte = pca.transform(Xte_raw)

    mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-8)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr_b = np.concatenate([np.ones((len(Xtr), 1)), Xtr], axis=1)
    Xte_b = np.concatenate([np.ones((len(Xte), 1)), Xte], axis=1)

    # inner CV over rcond (train half only)
    rng = np.random.default_rng(inner_seed)
    perm = rng.permutation(len(train_idx))
    folds = np.array_split(perm, inner_folds)
    scores = {rc: [] for rc in rconds}
    for k in range(inner_folds):
        va = folds[k]
        tr = np.concatenate([folds[j] for j in range(inner_folds) if j != k])
        wf = wtr[tr] / wtr[tr].sum()
        sw = np.sqrt(wf)[:, None]
        betas = _svd_beta_sweep(Xtr_b[tr] * sw, Ytr[tr] * sw, rconds)
        for rc, beta in betas.items():
            scores[rc].append(weighted_r2(Ytr[va], Xtr_b[va] @ beta, wtr[va]))
    best_rc = max(rconds, key=lambda rc: np.mean(scores[rc]))

    wf = wtr / wtr.sum()
    sw = np.sqrt(wf)[:, None]
    beta = _svd_beta_sweep(Xtr_b * sw, Ytr * sw, [best_rc])[best_rc]
    return weighted_r2(Yte, Xte_b @ beta, wte), int(Xtr.shape[1]), float(best_rc)


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------
def xf_acts(run_dir, ckpt, proc, device, random_seed=None):
    if random_seed is not None:
        m = load_random_model(str(run_dir), device, seed=random_seed)
    else:
        m = load_model(str(run_dir), ckpt, device)
    A = transformer_concat(m, proc["contexts"], device)
    del m
    if "cuda" in str(device):
        torch.cuda.empty_cache()
    return A


def gru_acts(run_dir, ckpt, proc, device, random_seed=None):
    cfg = load_run_config(str(run_dir))
    if random_seed is not None:
        m = build_rnn(cfg, proc["vocab"], device, state=None, seed=random_seed)
    else:
        state = torch.load(f"{run_dir}/{ckpt}", map_location=device,
                           weights_only=True)
        m = build_rnn(cfg, proc["vocab"], device, state=state)
    A = rnn_concat(m, proc["contexts"], proc["vocab"], device)
    del m
    return A


# ---------------------------------------------------------------------------
# process data
# ---------------------------------------------------------------------------
def load_proc():
    first = sorted([p for p in Path(XF_DIR).iterdir()
                    if p.is_dir() and p.name.startswith("run_")])[0]
    ref = load_run_config(str(first))
    proc = enumerate_process_data(ref)
    proc["contexts"] = proc["seqs"].numpy()[:, :ref["model_config"]["n_ctx"]]
    T = get_matrix_from_args(**ref["process_config"])
    proc["vocab"] = int(T.shape[0])
    _, dpar, dperp = decompose_beliefs(proc["beliefs"], proc["P_E"], proc["P_E_perp"])
    proc["delta_perp"] = dperp
    proc["var_perp_frac"] = float(np.var(dperp) / (np.var(dpar) + np.var(dperp)))
    return proc


# ---------------------------------------------------------------------------
# Part A: protocol reconciliation + untrained under paper protocol
# ---------------------------------------------------------------------------
def part_a(proc, device, t0):
    bel, w, dperp = proc["beliefs"], proc["weights"], proc["delta_perp"]
    P_E, P_E_perp = proc["P_E"], proc["P_E_perp"]
    dev = device if "cuda" in str(device) else "cpu"

    def R2p(X, Y):
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            return _paper_regression_r2(X, Y, w, device=dev)

    rows = []
    for arch, base, getter in [("transformer", XF_DIR, xf_acts),
                               ("GRU", GRU_DIR, gru_acts)]:
        run_dirs = sorted([p for p in Path(base).iterdir()
                           if p.is_dir() and p.name.startswith("run_")])
        for idx, rd in enumerate(run_dirs):
            cfg = load_run_config(str(rd))
            seed = cfg["model_config"].get("seed", "?")
            ckpt = list_checkpoints(str(rd))[-1]
            for state, rand_seed in [("trained", None), ("untrained", 999 + idx)]:
                A = getter(rd, ckpt, proc, device, random_seed=rand_seed)
                # audit protocol (in-sample, full-dim OLS)
                aud_bel = insample_ols_r2(A, bel, w)
                aud_perp = insample_ols_r2(A, dperp, w)
                rngA = np.random.default_rng(SHUFFLE_SEED)
                aud_null = np.array([
                    insample_ols_r2(A, geometric_shuffle(bel, P_E, P_E_perp, rngA), w)
                    for _ in range(N_SHUFFLES_AUDIT)])
                # paper protocol
                A_pca = _maybe_pca(A, PCA_RANK)[0]
                pap_bel = R2p(A_pca, bel)
                pap_perp = R2p(A_pca, dperp)
                if state == "trained":
                    d_pca_trained = int(A_pca.shape[1])
                    match_bel = match_perp = float("nan")
                else:
                    # matched-dim control: give the untrained net as many PCs
                    # as the trained net kept (guards against "PCA-0.95
                    # capacity-starves the untrained baseline")
                    A_match = _maybe_pca(A, d_pca_trained)[0]
                    match_bel = R2p(A_match, bel)
                    match_perp = R2p(A_match, dperp)
                if state == "untrained":
                    rngP = np.random.default_rng(SHUFFLE_SEED)
                    null = np.array([
                        R2p(A_pca, geometric_shuffle(bel, P_E, P_E_perp, rngP))
                        for _ in range(N_SHUFFLES_PAPER)])
                    null_mean, null_std, null_max = (float(null.mean()),
                                                     float(null.std()),
                                                     float(null.max()))
                    p_val = float(np.mean(null >= pap_bel))
                else:
                    null_mean = null_std = null_max = p_val = float("nan")
                rows.append({
                    "arch": arch, "seed": seed, "run": rd.name, "state": state,
                    "d_full": int(A.shape[1]), "d_pca": int(A_pca.shape[1]),
                    "audit_insample_bel_r2": aud_bel,
                    "audit_insample_perp_r2": aud_perp,
                    "audit_insample_null_mean": float(aud_null.mean()),
                    "audit_insample_null_std": float(aud_null.std()),
                    "paper_bel_r2": pap_bel, "paper_perp_frac": pap_perp,
                    "paper_bel_r2_matchdim": match_bel,
                    "paper_perp_frac_matchdim": match_perp,
                    "paper_null_mean": null_mean, "paper_null_std": null_std,
                    "paper_null_max": null_max, "paper_p_value": p_val,
                })
                r = rows[-1]
                print(f"  [A {arch:<11} s{str(seed):<5} {state:<9}] "
                      f"audit_bel={r['audit_insample_bel_r2']:.3f} "
                      f"audit_null={r['audit_insample_null_mean']:.3f} | "
                      f"paper_bel={r['paper_bel_r2']:.3f} "
                      f"perp={r['paper_perp_frac']:.3f} "
                      f"null={null_mean if not np.isnan(null_mean) else float('nan'):.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                del A, A_pca
    return rows


# ---------------------------------------------------------------------------
# Part B: held-out-context generalization
# ---------------------------------------------------------------------------
def part_b(proc, device, t0):
    bel, w, dperp = proc["beliefs"], proc["weights"], proc["delta_perp"]
    N = len(w)
    rows = []
    for arch, base, getter in [("transformer", XF_DIR, xf_acts),
                               ("GRU", GRU_DIR, gru_acts)]:
        run_dirs = sorted([p for p in Path(base).iterdir()
                           if p.is_dir() and p.name.startswith("run_")])
        for idx, rd in enumerate(run_dirs):
            cfg = load_run_config(str(rd))
            seed = cfg["model_config"].get("seed", "?")
            ckpt = list_checkpoints(str(rd))[-1]
            for state, rand_seed in [("trained", None), ("untrained", 999 + idx)]:
                A = getter(rd, ckpt, proc, device, random_seed=rand_seed)
                for split_seed in range(N_SPLITS_HELDOUT):
                    rng = np.random.default_rng(split_seed)
                    perm = rng.permutation(N)
                    tr, te = perm[: N // 2], perm[N // 2:]
                    r2_bel, d_pca, rc_b = heldout_r2(A, bel, w, tr, te)
                    r2_perp, _, rc_p = heldout_r2(A, dperp, w, tr, te)
                    rows.append({
                        "arch": arch, "seed": seed, "run": rd.name,
                        "state": state, "split_seed": split_seed,
                        "d_pca_train": d_pca,
                        "heldout_bel_r2": r2_bel, "heldout_perp_r2": r2_perp,
                        "rcond_bel": rc_b, "rcond_perp": rc_p,
                    })
                m_b = np.mean([r["heldout_bel_r2"] for r in rows[-N_SPLITS_HELDOUT:]])
                m_p = np.mean([r["heldout_perp_r2"] for r in rows[-N_SPLITS_HELDOUT:]])
                print(f"  [B {arch:<11} s{str(seed):<5} {state:<9}] "
                      f"heldout_bel={m_b:.3f} heldout_perp={m_p:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                del A
    return rows


# ---------------------------------------------------------------------------
# Part C: emergence over training (transformer runs 0-2)
# ---------------------------------------------------------------------------
def _select_ckpts(ckpts, n=14):
    toks = np.array([int(c.replace(".pt", "")) for c in ckpts])
    keep = [0, len(toks) - 1]
    pos = toks[toks > 0]
    grid = np.geomspace(pos.min(), pos.max(), n - 1)
    for g in grid:
        keep.append(int(np.argmin(np.abs(toks - g))))
    keep = sorted(set(keep))
    return [ckpts[i] for i in keep]


def _val_loss_at(run_dir, tokens):
    import csv as _csv
    with open(Path(run_dir) / "loss.csv") as fh:
        rows = list(_csv.DictReader(fh))
    toks = np.array([int(float(r["num_tokens_seen"])) for r in rows])
    i = int(np.argmin(np.abs(toks - tokens)))
    return float(rows[i]["val_loss_mean"])


def part_c(proc, device, t0, n_runs=3):
    bel, w, dperp = proc["beliefs"], proc["weights"], proc["delta_perp"]
    dev = device if "cuda" in str(device) else "cpu"
    N = len(w)
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    tr, te = perm[: N // 2], perm[N // 2:]

    def R2p(X, Y):
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            return _paper_regression_r2(X, Y, w, device=dev)

    rows = []
    run_dirs = sorted([p for p in Path(XF_DIR).iterdir()
                       if p.is_dir() and p.name.startswith("run_")])[:n_runs]
    for rd in run_dirs:
        cfg = load_run_config(str(rd))
        seed = cfg["model_config"].get("seed", "?")
        for ckpt in _select_ckpts(list_checkpoints(str(rd))):
            tokens = int(ckpt.replace(".pt", ""))
            A = xf_acts(rd, ckpt, proc, device)
            A_pca = _maybe_pca(A, PCA_RANK)[0]
            bel_r2 = R2p(A_pca, bel)
            perp = R2p(A_pca, dperp)
            ho_perp, _, _ = heldout_r2(A, dperp, w, tr, te)
            rows.append({
                "arch": "transformer", "seed": seed, "run": rd.name,
                "checkpoint": ckpt, "tokens": tokens,
                "val_loss_mean": _val_loss_at(rd, tokens),
                "belief_r2_pca": bel_r2, "perp_frac": perp,
                "heldout_perp_r2": ho_perp, "d_pca": int(A_pca.shape[1]),
            })
            r = rows[-1]
            print(f"  [C s{str(seed):<5} tok={tokens:>11,}] loss={r['val_loss_mean']:.4f} "
                  f"bel={bel_r2:.3f} perp={perp:.3f} ho_perp={ho_perp:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del A, A_pca
    return rows


# ---------------------------------------------------------------------------
def _write_csv(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--parts", default="abc")
    ap.add_argument("--out_prefix", default="results/qrrxor_repr_audit")
    args = ap.parse_args()
    t0 = time.time()

    proc = load_proc()
    print(f"process: N={len(proc['weights'])} belief_dim={proc['beliefs'].shape[1]} "
          f"var_perp_frac={proc['var_perp_frac']:.3f} ({time.time()-t0:.1f}s)", flush=True)

    payload = {"meta": {"pca_rank": PCA_RANK, "n_shuffles_paper": N_SHUFFLES_PAPER,
                        "n_shuffles_audit": N_SHUFFLES_AUDIT,
                        "n_splits_heldout": N_SPLITS_HELDOUT,
                        "var_perp_frac": proc["var_perp_frac"]}}
    if "a" in args.parts:
        rows_a = part_a(proc, args.device, t0)
        _write_csv(f"{args.out_prefix}_protocols.csv", rows_a)
        payload["protocols"] = rows_a
    if "b" in args.parts:
        rows_b = part_b(proc, args.device, t0)
        _write_csv(f"{args.out_prefix}_heldout.csv", rows_b)
        payload["heldout"] = rows_b
    if "c" in args.parts:
        rows_c = part_c(proc, args.device, t0)
        _write_csv(f"{args.out_prefix}_emergence.csv", rows_c)
        payload["emergence"] = rows_c
    joblib.dump(payload, f"{args.out_prefix}.joblib")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
