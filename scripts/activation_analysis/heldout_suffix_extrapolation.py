"""Held-out-suffix-CLASS extrapolation under the honest no-PCA ridge probe.

Question: is the transformer's belief
decodability an EXTRAPOLATING affine code (probe generalizes to suffix classes
never seen during probe fitting, like the GRU), or an INTERPOLATIVE carrier
(probe = stitched per-suffix-class lookup that collapses on held-out classes)?

An earlier version of this test used PCA-0.95, which removes the transformer's
low-variance ker signal (0.83 -> 0.985 no-PCA); this is the same design under the
no-PCA ridge + group-CV protocol.

Design
------
- All 1024 length-10 contexts enumerated, probability-weighted (population).
- Suffix-k classes (last k tokens), k in {2,3,4}. Outer loop: leave one class
  out entirely; fit ridge probe on the rest; predict the held-out class.
  Ridge lambda chosen by INNER leave-one-training-class-out (matches the
  evaluation regime; never touches the test class). Pooled weighted R2 over
  the union of held-out predictions (every context predicted exactly once by
  a probe that never saw its suffix class).
- Targets: full belief (4d), ker(E^T) part (beyond-NTP, headline), col(E) part.
  Lambda is selected per target.
- Families: transformer trained/untrained (5 seeds each; untrained = fresh
  inits 1000+k), GRU trained/untrained. Analytic baselines: suffix-k one-hot
  lookup (definitional interpolation floor: cannot extrapolate), token-additive
  (10 position bits: extrapolation available "for free" from order-1 structure),
  token-pairwise (bits + all 45 pairwise products: cheap order-2 composition).
- Sites: transformer concat resid_pre L0..3 at last pos (matches the shuffle
  analysis concat protocol, 256-dim) plus L2/L3 resid_pre individually;
  GRU carried state after 10 tokens (h all layers, 256-dim).
- Reference column: in-distribution row-level 5-fold CV R2 (cv_ridge_r2),
  so collapse is measured as (in-dist - extrapolation) within one protocol.

Fold fits use per-class Gram/crossprod caching: G_c = Xb_c' diag(w_c) Xb_c,
B_c = Xb_c' diag(w_c) Y_c summed over the fold's classes and renormalized by
the fold weight-sum reproduce wridge (which normalizes w within the fold)
EXACTLY; this is an algebraic speedup, not an approximation.

Run:  PYTHONPATH=. uv run python scripts/activation_analysis/heldout_suffix_extrapolation.py --device cuda:0
Out:  results/heldout_suffix/heldout_suffix_extrapolation.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformer_lens import HookedTransformer, HookedTransformerConfig

from scripts.activation_analysis.causal_verify_xf import (
    process_setup, enum_prefixes, cv_ridge_r2, load_xf)
from scripts.activation_analysis.causal_verify_rnn import build, states_last

XF_DIR = Path("results/qrrxor_robust_transformer")
GRU_DIR = Path("results/qrrxor_robust_gru")
OUT_DIR = Path("results/heldout_suffix")
LAMS = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)
GRANULARITIES = (2, 3, 4)


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------

def xf_sites(model, toks, device, bs=2048):
    """Last-position resid_pre per layer + the all-layer concat."""
    names = [f"blocks.{i}.hook_resid_pre" for i in range(model.cfg.n_layers)]
    per = {n: [] for n in names}
    for s in range(0, len(toks), bs):
        x = torch.from_numpy(toks[s:s + bs]).long().to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(x, names_filter=names)
        for n in names:
            per[n].append(cache[n][:, -1, :].float().cpu().numpy())
    per = {n: np.concatenate(v) for n, v in per.items()}
    return {"concat": np.concatenate([per[n] for n in names], 1),
            "L2_resid_pre": per["blocks.2.hook_resid_pre"],
            "L3_resid_pre": per["blocks.3.hook_resid_pre"]}


def untrained_xf(run_dir, device, seed):
    with open(run_dir / "hooked_model_config.json") as f:
        mc = json.load(f)
    mc["dtype"] = getattr(torch, mc["dtype"].split(".")[-1])
    mc["device"] = device
    mc["seed"] = seed
    m = HookedTransformer(HookedTransformerConfig(**mc)).to(device)
    m.eval()
    return m


def gru_states(run_dir, device, toks, untrained_seed=None):
    with open(run_dir / "run_config.yaml") as f:
        cfg = yaml.safe_load(f)
    state = None
    if untrained_seed is None:
        pts = sorted(run_dir.glob("*.pt"), key=lambda p: int(p.stem))
        state = torch.load(pts[-1], map_location=device, weights_only=True)
    model = build(cfg["model_config"], 2, device, state=state,
                  init_seed=untrained_seed)
    X = states_last(model, toks, 2, device, is_lstm=False).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return X


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------

def _class_grams(X, Y, w, cls, classes):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    G, B, S = {}, {}, {}
    for c in classes:
        m = cls == c
        Xw = Xb[m] * w[m][:, None]
        G[c] = Xw.T @ Xb[m]
        B[c] = Xw.T @ Y[m]
        S[c] = float(w[m].sum())
    return Xb, G, B, S


def _fit(G, B, S, class_set, lam, dim):
    Gs = sum(G[c] for c in class_set)
    Bs = sum(B[c] for c in class_set)
    W = sum(S[c] for c in class_set)
    A = Gs / W + lam * np.eye(dim)
    A[0, 0] -= lam                       # don't penalize intercept
    return np.linalg.solve(A, Bs / W)


def heldout_class_r2(X, Y, w, cls):
    """Leave-one-suffix-class-out extrapolation; pooled weighted R2 + per-class."""
    classes = list(np.unique(cls))
    Xb, G, B, S = _class_grams(X, Y, w, cls, classes)
    dim = Xb.shape[1]
    preds = np.zeros_like(Y)
    lam_used = []
    for c in classes:
        train = [c2 for c2 in classes if c2 != c]
        best, blam = np.inf, LAMS[0]
        for lam in LAMS:
            ssr = 0.0
            for c2 in train:
                inner = [c3 for c3 in train if c3 != c2]
                beta = _fit(G, B, S, inner, lam, dim)
                m = cls == c2
                ssr += float((w[m][:, None] * (Y[m] - Xb[m] @ beta) ** 2).sum())
            if ssr < best:
                best, blam = ssr, lam
        beta = _fit(G, B, S, train, blam, dim)
        m = cls == c
        preds[m] = Xb[m] @ beta
        lam_used.append(blam)
    wn = w / w.sum()
    ym = (Y * wn[:, None]).sum(0)
    sst = float((wn[:, None] * (Y - ym) ** 2).sum())
    ssr = float((wn[:, None] * (Y - preds) ** 2).sum())
    pooled = 1 - ssr / sst
    per_class = []
    for c in classes:
        m = cls == c
        ssr_c = float((wn[m][:, None] * (Y[m] - preds[m]) ** 2).sum())
        sst_c = float((wn[m][:, None] * (Y[m] - ym) ** 2).sum())
        per_class.append(1 - ssr_c / sst_c if sst_c > 0 else np.nan)
    return pooled, float(np.nanmin(per_class)), float(np.nanmedian(lam_used))


def evaluate(name, seed, site, X, targets, w, toks, rows, t0,
             granularities=GRANULARITIES):
    local = []
    for tname, Y in targets.items():
        indist, _ = cv_ridge_r2(X, Y, w, seed=0, lams=LAMS)
        for k in granularities:
            cls = toks[:, -k:] @ (1 << np.arange(k - 1, -1, -1))
            pooled, worst, lam = heldout_class_r2(X, Y, w, cls)
            local.append(dict(family=name, seed=seed, site=site, target=tname,
                              suffix_k=k, extrap_r2=round(pooled, 4),
                              extrap_worst_class_r2=round(worst, 4),
                              indist_r2=round(indist, 4), lam_med=lam))
    rows.extend(local)
    ker = {r["suffix_k"]: r["extrap_r2"] for r in local if r["target"] == "ker"}
    print(f"[{time.time() - t0:7.1f}s] {name} s{seed} {site}: ker extrap "
          + " ".join(f"k{k}={v:+.3f}" for k, v in sorted(ker.items())), flush=True)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(OUT_DIR / "heldout_suffix_extrapolation.csv"))
    args = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(sorted(XF_DIR.glob("run_*"))[0] / "run_config.yaml") as f:
        ref_cfg = yaml.safe_load(f)
    T, rev, b0, E, Pcol, Pker, vocab, d = process_setup(ref_cfg["process_config"])
    n_ctx = ref_cfg["model_config"]["n_ctx"]
    toks, bel, wt, p0 = enum_prefixes(T, rev, b0, n_ctx)
    targets = {"belief": bel, "ker": bel @ Pker, "col": bel @ Pcol}
    kv = float((wt[:, None] * (bel @ Pker - (wt[:, None] * (bel @ Pker)).sum(0)) ** 2).sum())
    tv = float((wt[:, None] * (bel - (wt[:, None] * bel).sum(0)) ** 2).sum())
    print(f"process: N={len(toks)} n_ctx={n_ctx} belief_dim={d} "
          f"ker_var_frac={kv / tv:.3f}", flush=True)

    rows = []

    # transformer trained + untrained
    for rd in sorted(XF_DIR.glob("run_*")):
        with open(rd / "run_config.yaml") as f:
            seed = yaml.safe_load(f)["model_config"]["seed"]
        model = load_xf(rd, args.device)
        for site, X in xf_sites(model, toks, args.device).items():
            evaluate("XF_trained", seed, site, X, targets, wt, toks, rows, t0)
        del model
        torch.cuda.empty_cache()
    rd0 = sorted(XF_DIR.glob("run_*"))[0]
    for k in range(5):
        model = untrained_xf(rd0, args.device, seed=1000 + k)
        for site, X in xf_sites(model, toks, args.device).items():
            evaluate("XF_untrained", 1000 + k, site, X, targets, wt, toks, rows, t0)
        del model
        torch.cuda.empty_cache()

    # GRU trained + untrained
    for rd in sorted(GRU_DIR.glob("run_*")):
        with open(rd / "run_config.yaml") as f:
            seed = yaml.safe_load(f)["model_config"]["seed"]
        X = gru_states(rd, args.device, toks)
        evaluate("GRU_trained", seed, "state", X, targets, wt, toks, rows, t0)
    rd0 = sorted(GRU_DIR.glob("run_*"))[0]
    for k in range(5):
        X = gru_states(rd0, args.device, toks, untrained_seed=1000 + k)
        evaluate("GRU_untrained", 1000 + k, "state", X, targets, wt, toks, rows, t0)

    # analytic baselines
    bits = toks.astype(np.float64)
    pair = np.stack([bits[:, i] * bits[:, j]
                     for i in range(n_ctx) for j in range(i + 1, n_ctx)], 1)
    evaluate("token_additive", 0, "bits10", bits, targets, wt, toks, rows, t0)
    evaluate("token_pairwise", 0, "bits10+pairs45",
             np.concatenate([bits, pair], 1), targets, wt, toks, rows, t0)
    for k in GRANULARITIES:
        cls = toks[:, -k:] @ (1 << np.arange(k - 1, -1, -1))
        onehot = np.eye(2 ** k)[cls]
        evaluate(f"suffix{k}_lookup", 0, f"onehot{2**k}", onehot, targets, wt,
                 toks, rows, t0, granularities=(k,))

    keys = list(rows[0].keys())
    with open(args.out, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=keys)
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows, {time.time() - t0:.1f}s)", flush=True)

    print("\n=== SUMMARY: ker-target extrapolation R2 (mean ± std over seeds) ===", flush=True)
    agg = collections.defaultdict(list)
    for r in rows:
        if r["target"] == "ker":
            agg[(r["family"], r["site"], r["suffix_k"])].append(
                (r["extrap_r2"], r["indist_r2"]))
    for (fam, site, k), v in sorted(agg.items()):
        e = np.array([x[0] for x in v])
        i = np.array([x[1] for x in v])
        print(f"  {fam:16s} {site:16s} k={k}:  extrap {e.mean():+.3f} ± {e.std():.3f}"
              f"   (in-dist {i.mean():+.3f})", flush=True)


if __name__ == "__main__":
    main()
