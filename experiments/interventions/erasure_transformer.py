"""
E2-AUDIT — completeness audit of the linear surgery in causal_ker_ablation.py (the earlier single-shot erasure; not shipped).

The published result: single-shot linear ker(E^T) erasure at resid_pre L1/L2/L3
costs ~0.0004-0.008 nats (vs col erasure 0.026). The hole: that surgery only
removes what ONE ridge probe reads along ONE decoder plane. This script checks
(R) how much ker is still readable after the surgery — fresh ridge probes with
group-CV over unique prefixes, plus a 1x128 MLP probe, on clean vs ablated
activations, at the surgical site and downstream — and (S) re-measures the loss
cost under an ITERATED scrub (refit encoder on current ablated acts; decoder is
the regression of acts on the *predicted* ker coords, which zeroes the
activation/target cross-covariance at the fit points; repeat to plateau).
The col(E) positive control goes through the same audit for apples-to-apples.

Usage:
  PYTHONPATH=. uv run python experiments/interventions/erasure_transformer.py \
      --run_dir results/qrrxor_robust_transformer/run_0_... --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from epsilon_transformers.analysis.checkpoints import (
    load_run_config, list_checkpoints, load_model,
)
from epsilon_transformers.analysis.process_targets import _stationary_belief
from epsilon_transformers.analysis.shuffle_control import (
    compute_emission_matrix, compute_projections,
)
from epsilon_transformers.analysis.interventions import prefix_beliefs
from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.analysis.erasure_probes import (
    wfit, ridge_cv_best, fit_ridge_raw, probe_linear_cv, probe_mlp_cv,
)

SITES = ("blocks.1.hook_resid_pre", "blocks.2.hook_resid_pre", "blocks.3.hook_resid_pre")
CAP_SITES = SITES + ("blocks.3.hook_resid_post",)
SITE_ORDER = {"blocks.1.hook_resid_pre": 1.0, "blocks.2.hook_resid_pre": 2.0,
              "blocks.3.hook_resid_pre": 3.0, "blocks.3.hook_resid_post": 3.5}
SHORT = {"blocks.1.hook_resid_pre": "L1", "blocks.2.hook_resid_pre": "L2",
         "blocks.3.hook_resid_pre": "L3", "blocks.3.hook_resid_post": "L3post"}
LABEL2SITES = {"L1": SITES[:1], "L2": SITES[1:2], "L3": SITES[2:3], "ALL": SITES}
ABL_POS = tuple(range(2, 10))
READ_POS = tuple(range(2, 10))
PROBE_POS = tuple(range(4, 10))          # >=16 unique prefixes
MLP_POS = (5, 7, 9)
N_CTX = 10
EPS = 1e-12
MAX_IT = 12
STOP_CV = 0.02        # scrub converged when max group-CV R2 over (site,pos) < this
TERM_CV = 0.02        # append a scrub term wherever group-CV R2 > this


def w_r2_insample(X, Y, w, beta):
    w = w / w.sum()
    Yp = np.hstack([np.ones((len(X), 1)), X]) @ beta
    ym = (Y * w[:, None]).sum(0)
    ssr = (w[:, None] * (Y - Yp) ** 2).sum()
    sst = (w[:, None] * (Y - ym) ** 2).sum()
    return float(1 - ssr / sst) if sst > 0 else 0.0


def run_one(run_dir, device, out_dir, skip_mlp=False, max_it=MAX_IT):
    t0 = time.time()
    cfg = load_run_config(str(run_dir))
    seed = cfg["model_config"]["seed"]
    pc = cfg["process_config"].copy(); name = pc.pop("name")
    T = get_matrix_from_args(name=name, **pc)
    Pm = T.sum(0)
    ev, evec = np.linalg.eig(Pm)
    rev = np.real(evec[:, np.argmin(np.abs(ev - 1.0))]); rev /= rev.sum()
    b0 = _stationary_belief(T, rev)
    E = compute_emission_matrix(T, rev)
    P_E, P_E_perp = compute_projections(E)
    vocab = T.shape[0]

    # orthonormal bases of col(E) / ker(E^T)
    def basis(P):
        lam, V = np.linalg.eigh(P)
        return V[:, lam > 0.5]
    K = basis(P_E_perp)          # (4, 2)
    C = basis(P_E)               # (4, 2)

    toks, _, wt, _ = prefix_beliefs(T, rev, b0, N_CTX)
    pos_bel, true_cond = {}, {}
    for p in READ_POS:
        _, bg, _, _ = prefix_beliefs(T, rev, b0, p + 1)
        idx = np.array([int("".join(map(str, tk[:p + 1])), 2) for tk in toks])
        pos_bel[p] = bg[idx]
        pc_ = np.stack([np.clip(pos_bel[p] @ T[x] @ rev, EPS, None)
                        for x in range(vocab)], 1)
        true_cond[p] = pc_ / pc_.sum(1, keepdims=True)
    Z = {"ker": {p: pos_bel[p] @ K for p in READ_POS},
         "col": {p: pos_bel[p] @ C for p in READ_POS}}

    # dedup by prefix (group-CV unit = unique prefix)
    pow2 = 2 ** np.arange(N_CTX)[::-1]
    dedup = {}
    for p in READ_POS:
        ids = toks[:, :p + 1] @ pow2[N_CTX - p - 1:]
        uniq, first, inv = np.unique(ids, return_index=True, return_inverse=True)
        w_u = np.bincount(inv, weights=wt)
        dedup[p] = (first, w_u)

    model = load_model(str(run_dir), list_checkpoints(str(run_dir))[-1], device)
    dt = model.cfg.dtype
    x = torch.from_numpy(toks[:, :N_CTX]).long().to(device)
    with torch.no_grad():
        lg, cache = model.run_with_cache(x, names_filter=list(CAP_SITES))
        soft_clean = F.softmax(lg.float(), -1).cpu().numpy()
    A_clean = {s: cache[s].float().cpu().numpy() for s in CAP_SITES}
    del cache

    # sanity: same-prefix rows have identical activations (dedup validity)
    for p in (5, 7):
        first, _ = dedup[p]
        ids = toks[:, :p + 1] @ pow2[N_CTX - p - 1:]
        a = A_clean[SITES[1]][:, p, :]
        ref = a[first][np.searchsorted(np.sort(np.unique(ids)), ids)]
        assert np.abs(a - ref).max() < 1e-4, "dedup mismatch"

    # ORIGINAL surgery params (identical fit to causal_ker_ablation.py)
    params = {}
    for s in SITES:
        for p in ABL_POS:
            beta = wfit(A_clean[s][:, p, :], pos_bel[p], wt)
            gbeta = wfit(pos_bel[p], A_clean[s][:, p, :], wt)
            mean_bel = (pos_bel[p] * (wt / wt.sum())[:, None]).sum(0)
            params[(s, p)] = (beta[1:], beta[0], gbeta[1:], mean_bel)

    def orig_term(s, p, kind):
        W, c, G, mb = params[(s, p)]
        P_ = P_E_perp if kind == "ker" else P_E
        return (W.copy(), (c - mb).copy(), (P_ @ G).copy())

    def to_torch(terms):
        return {k: [(torch.tensor(Wt, dtype=dt, device=device),
                     torch.tensor(ct, dtype=dt, device=device),
                     torch.tensor(Gt, dtype=dt, device=device))
                    for (Wt, ct, Gt) in v] for k, v in terms.items()}

    def run_pass(surg_sites, terms):
        """Forward with stacked surgery; capture post-surgery acts at CAP_SITES."""
        tt = to_torch(terms)
        caps = {}

        def make_surg(site):
            def hook(t, hook):
                t = t.clone()
                for p in ABL_POS:
                    a = t[:, p, :]
                    for (Wt, ct, Gt) in tt[(site, p)]:
                        a = a - ((a @ Wt) + ct) @ Gt
                    t[:, p, :] = a
                caps[site] = t.float().detach().cpu().numpy()
                return t
            return hook

        def make_cap(site):
            def hook(t, hook):
                caps[site] = t.float().detach().cpu().numpy()
                return t
            return hook

        hooks = [(s, make_surg(s)) for s in surg_sites]
        hooks += [(s, make_cap(s)) for s in CAP_SITES if s not in surg_sites]
        with torch.no_grad():
            lg = model.run_with_hooks(x, fwd_hooks=hooks)
            soft = F.softmax(lg.float(), -1).cpu().numpy()
        return soft, caps

    w_ = wt / wt.sum()

    def score(soft):
        out = {}
        loss_cl = loss_ab = 0.0
        for p in READ_POS:
            pcl = np.clip(soft_clean[:, p, :], EPS, None); pcl /= pcl.sum(1, keepdims=True)
            pab = np.clip(soft[:, p, :], EPS, None); pab /= pab.sum(1, keepdims=True)
            out[f"kl_pos{p}"] = float((w_ * (pcl * (np.log(pcl) - np.log(pab))).sum(1)).sum())
            loss_cl += -(w_[:, None] * true_cond[p] * np.log(pcl)).sum()
            loss_ab += -(w_[:, None] * true_cond[p] * np.log(pab)).sum()
        out["dloss_nats"] = float((loss_ab - loss_cl) / len(READ_POS))
        out["kl_mean"] = float(np.mean([out[f"kl_pos{p}"] for p in READ_POS]))
        return out

    resid_rows, scrub_rows = [], []

    def probe_caps(caps, phase, kind, surgery, surg_sites, mlp_targets=True):
        """Probe residue on captured acts; kind target + col specificity."""
        min_order = min(SITE_ORDER[s] for s in surg_sites) if surg_sites else 0.0
        for cs in CAP_SITES:
            if SITE_ORDER[cs] < min_order:
                continue
            role = ("site" if cs in surg_sites else "down")
            if not surg_sites:
                role = "clean"
            for p in PROBE_POS:
                first, w_u = dedup[p]
                Xu = caps[cs][first, p, :]
                for tgt in ("ker", "col"):
                    if phase == "clean" and tgt == "col" and SITE_ORDER[cs] < 3.0:
                        continue
                    if phase != "clean" and tgt != kind and role != "site":
                        continue        # specificity probe only at the surgical site
                    Yu = Z[tgt][p][first]
                    r2l = probe_linear_cv(Xu, Yu, w_u, seed=1000 + p)
                    r2m = float("nan")
                    if (not skip_mlp and mlp_targets and tgt == (kind if phase != "clean" else tgt)
                            and p in MLP_POS and (phase == "clean" or tgt == kind)):
                        r2m = probe_mlp_cv(Xu, Yu, w_u, device, seed=2000 + p)
                    resid_rows.append({
                        "seed": seed, "phase": phase, "kind": kind, "surgery": surgery,
                        "cap_site": SHORT[cs], "role": role, "pos": p, "target": tgt,
                        "n_uniq": len(first), "r2_lin_cv": r2l, "r2_mlp_cv": r2m})
            print(f"    [{phase} {kind}@{surgery} cap={SHORT[cs]}] probed "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---------------- clean reference ----------------
    print(f"[s{seed}] clean reference probes", flush=True)
    probe_caps(A_clean, "clean", "none", "none", ())

    # ---------------- Phase R: residue after ORIGINAL surgery ----------------
    R_CONFIGS = [("ker", "L1"), ("ker", "L2"), ("ker", "L3"), ("col", "L2"), ("col", "L3")]
    for kind, label in R_CONFIGS:
        sites = LABEL2SITES[label]
        terms = {(s, p): [orig_term(s, p, kind)] for s in sites for p in ABL_POS}
        soft, caps = run_pass(sites, terms)
        c = score(soft)
        print(f"[s{seed}] ORIG {kind}@{label}: dloss={c['dloss_nats']:+.5f} "
              f"kl_mean={c['kl_mean']:.5f}", flush=True)
        scrub_rows.append({"seed": seed, "kind": kind, "sites": label, "iter": 0,
                           "stage": "orig", "n_terms": 1, "r2_cv_max": float("nan"),
                           **c})
        mlp_ok = not (kind == "col" and label == "L2")
        probe_caps(caps, "orig", kind, label, sites, mlp_targets=mlp_ok)

    # ---------------- Phase S: iterated scrub ----------------
    S_CONFIGS = [("ker", "L1"), ("ker", "L2"), ("ker", "L3"), ("ker", "ALL"), ("col", "L3")]
    for kind, label in S_CONFIGS:
        sites = LABEL2SITES[label]
        terms = {(s, p): [orig_term(s, p, kind)] for s in sites for p in ABL_POS}
        caps = None
        for it in range(max_it + 1):
            soft, caps = run_pass(sites, terms)
            c = score(soft)
            # residual GROUP-CV linear readability of the target on the CURRENT
            # ablated acts (unique-prefix rows; in-sample R2 is vacuous when
            # n_uniq <~ d_model)
            r2max, fits = 0.0, {}
            for s in sites:
                for p in ABL_POS:
                    first, w_u = dedup[p]
                    Xu = caps[s][first, p, :]
                    Yu = Z[kind][p][first]
                    r2cv, ridge = ridge_cv_best(Xu, Yu, w_u, seed=1000 + p)
                    fits[(s, p)] = (Xu, Yu, w_u, r2cv, ridge)
                    if not np.isnan(r2cv):
                        r2max = max(r2max, r2cv)
            scrub_rows.append({"seed": seed, "kind": kind, "sites": label, "iter": it,
                               "stage": "scrub",
                               "n_terms": len(terms[(sites[0], ABL_POS[0])]),
                               "r2_cv_max": r2max, **c})
            print(f"[s{seed}] SCRUB {kind}@{label} it={it}: r2_cv_max={r2max:.5f} "
                  f"dloss={c['dloss_nats']:+.5f} kl_mean={c['kl_mean']:.5f}", flush=True)
            if r2max < STOP_CV or it == max_it:
                break
            for (s, p), (Xu, Yu, w_u, r2cv, ridge) in fits.items():
                if np.isnan(r2cv) or r2cv <= TERM_CV:
                    continue
                beta = fit_ridge_raw(Xu, Yu, w_u, ridge)
                zhat = np.hstack([np.ones((len(Xu), 1)), Xu]) @ beta
                zbar = (zhat * (w_u / w_u.sum())[:, None]).sum(0)
                gam = wfit(zhat, Xu, w_u)      # decoder on PREDICTED coords
                terms[(s, p)].append((beta[1:], beta[0] - zbar, gam[1:]))
        probe_caps(caps, "scrub", kind, label, sites,
                   mlp_targets=(label != "ALL"))

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    f1 = out / f"resid_xf_s{seed}.csv"
    with open(f1, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(resid_rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(resid_rows)
    f2 = out / f"scrub_xf_s{seed}.csv"
    with open(f2, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(scrub_rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(scrub_rows)
    print(f"saved {f1} and {f2}  ({time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default="results/causal_transplant/audit")
    ap.add_argument("--skip_mlp", action="store_true")
    ap.add_argument("--max_it", type=int, default=MAX_IT)
    args = ap.parse_args()
    run_one(args.run_dir, args.device, args.out_dir, args.skip_mlp, args.max_it)


if __name__ == "__main__":
    main()
