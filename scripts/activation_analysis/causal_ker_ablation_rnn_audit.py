"""
E2-RNN-AUDIT — completeness audit of the carried-state ker surgery in
causal_ker_ablation_rnn.py (symmetry check for the transformer audit).

(R) After the ORIGINAL pooled-probe ker surgery at steps 3..6, how much of the
true ker(E^T) belief coordinate is still readable from the carried state —
fresh ridge probes (group-CV over unique prefixes) + a 1x128 MLP probe, at the
surgical steps and downstream (7..9), clean states as reference.
(S) Iterated per-step scrub (refit encoder on current ablated states; decoder
regressed on the predicted coords; repeat to plateau) and the resulting
kl_future vs the exact theory prediction.

Usage:
  PYTHONPATH=. uv run python scripts/activation_analysis/causal_ker_ablation_rnn_audit.py \
      --run_dir results/qrrxor_robust_rnn/run_0_... --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.activation_analysis.run_shuffle_analysis import load_run_config
from scripts.activation_analysis.enumerate_process import _stationary_belief
from scripts.activation_analysis.shuffle_control import (
    compute_emission_matrix, compute_projections,
)
from scripts.activation_analysis.causal_phase0 import prefix_beliefs
from scripts.activation_analysis.causal_belief_transplant import RNNBackend
from scripts.activation_analysis.causal_ker_ablation_audit import (
    wfit, ridge_cv_best, fit_ridge_raw, probe_linear_cv, probe_mlp_cv,
)
from epsilon_transformers.process.transition_matrices import get_matrix_from_args

ABL_STEPS = (3, 4, 5, 6)
CAP_STEPS = tuple(range(3, 10))
PROBE_STEPS = tuple(range(4, 10))
MLP_STEPS = (6, 8)
READ_STEPS = tuple(range(2, 10))
N_CTX = 10
EPS = 1e-12
MAX_IT = 12
STOP_CV = 0.02
TERM_CV = 0.02


def run_one(run_dir, device, out_dir, skip_mlp=False, max_it=MAX_IT):
    t0 = time.time()
    cfg = load_run_config(str(run_dir))
    seed = cfg["model_config"]["seed"]
    arch = cfg["model_config"]["rnn_type"].upper()
    pc = cfg["process_config"].copy(); name = pc.pop("name")
    T = get_matrix_from_args(name=name, **pc)
    Pm = T.sum(0)
    ev, evec = np.linalg.eig(Pm)
    rev = np.real(evec[:, np.argmin(np.abs(ev - 1.0))]); rev /= rev.sum()
    b0 = _stationary_belief(T, rev)
    E = compute_emission_matrix(T, rev)
    P_E, P_E_perp = compute_projections(E)
    vocab = T.shape[0]

    def basis(P):
        lam, V = np.linalg.eigh(P)
        return V[:, lam > 0.5]
    K = basis(P_E_perp)
    C = basis(P_E)

    toks, _, wt, _ = prefix_beliefs(T, rev, b0, N_CTX)
    Nseq = len(toks)
    w_ = wt / wt.sum()
    pos_bel = {}
    for t in range(1, N_CTX + 1):
        _, bg, _, _ = prefix_beliefs(T, rev, b0, t)
        idx = np.array([int("".join(map(str, tk[:t])), 2) for tk in toks])
        pos_bel[t] = bg[idx]
    mean_bel = {t: (pos_bel[t] * w_[:, None]).sum(0) for t in range(1, N_CTX + 1)}
    Zk = {t: pos_bel[t] @ K for t in range(1, N_CTX + 1)}
    Zc = {t: pos_bel[t] @ C for t in range(1, N_CTX + 1)}

    pow2 = 2 ** np.arange(N_CTX)[::-1]
    dedup = {}
    for t in CAP_STEPS:
        ids = toks[:, :t] @ pow2[N_CTX - t:]
        uniq, first, inv = np.unique(ids, return_index=True, return_inverse=True)
        dedup[t] = (first, np.bincount(inv, weights=wt))

    be = RNNBackend(run_dir, device, T, rev, b0, vocab)
    L, H = be.L, be.H
    D = L * H * (2 if be.is_lstm else 1)
    n_parts = 2 if be.is_lstm else 1
    x = torch.from_numpy(toks[:, :N_CTX]).long().to(device)
    oh = F.one_hot(x, num_classes=vocab).float()

    def flatten(h):
        parts = h if isinstance(h, tuple) else (h,)
        return torch.cat([p[i] for p in parts for i in range(L)], dim=1)

    def unflatten(v):
        blocks = v.view(-1, n_parts * L, H).permute(1, 0, 2).contiguous()
        return (blocks[:L], blocks[L:]) if be.is_lstm else blocks

    def orig_term(t_step, kind):
        P_ = P_E_perp if kind == "ker" else P_E
        return (be.Wmat.copy(), (be.beta[0] - mean_bel[t_step]).copy(),
                (P_ @ be.G).copy())

    def run_pass(terms):
        """Stepwise forward with stacked surgery; capture states at CAP_STEPS."""
        tt = None
        if terms is not None:
            tt = {k: [(torch.tensor(Wt, dtype=torch.float32, device=device),
                       torch.tensor(ct, dtype=torch.float32, device=device),
                       torch.tensor(Gt, dtype=torch.float32, device=device))
                      for (Wt, ct, Gt) in v] for k, v in terms.items()}
        caps = {}
        h = None
        logits = []
        with torch.no_grad():
            for t in range(N_CTX):
                o, h = be.model.rnn(oh[:, t:t + 1], h)
                step = t + 1
                if tt is not None and step in ABL_STEPS:
                    v = flatten(h)
                    for (Wt, ct, Gt) in tt[step]:
                        v = v - ((v @ Wt) + ct) @ Gt
                    h = unflatten(v)
                    top = v[:, (L - 1) * H:L * H]
                    logits.append(be.model.output_layer(top).unsqueeze(1))
                    if step in CAP_STEPS:
                        caps[step] = v.float().cpu().numpy()
                else:
                    logits.append(be.model.output_layer(o))
                    if step in CAP_STEPS:
                        caps[step] = flatten(h).float().cpu().numpy()
        return torch.cat(logits, dim=1), caps

    clean_lg, caps_clean = run_pass(None)
    soft_clean = F.softmax(clean_lg.float(), -1).cpu().numpy()

    def score(lg):
        soft = F.softmax(lg.float(), -1).cpu().numpy()
        out = {}
        for t in READ_STEPS:
            pcl = np.clip(soft_clean[:, t, :], EPS, None); pcl /= pcl.sum(1, keepdims=True)
            pab = np.clip(soft[:, t, :], EPS, None); pab /= pab.sum(1, keepdims=True)
            out[f"kl_step{t}"] = float((w_ * (pcl * (np.log(pcl) - np.log(pab))).sum(1)).sum())
        out["kl_local"] = out["kl_step5"]
        out["kl_future"] = float(np.mean([out[f"kl_step{t}"] for t in (6, 7, 8, 9)]))
        return out

    # exact theory of the belief-space ker collapse (same as original script)
    def theory_kls():
        bel = np.tile(b0, (Nseq, 1))
        out = {}
        for t in range(N_CTX):
            xt = toks[:, t]
            nb = np.einsum("ni,xij->nxj", bel, T)[np.arange(Nseq), xt]
            nrm = np.clip(nb @ rev, EPS, None)
            bel = nb / nrm[:, None]
            if (t + 1) in ABL_STEPS:
                bel = bel - ((bel - mean_bel[t + 1]) @ P_E_perp)
            if t in READ_STEPS:
                p_ab = np.stack([np.clip(bel @ T[v_] @ rev, EPS, None)
                                 for v_ in range(vocab)], 1)
                p_ab /= p_ab.sum(1, keepdims=True)
                p_tr = np.stack([np.clip(pos_bel[t + 1] @ T[v_] @ rev, EPS, None)
                                 for v_ in range(vocab)], 1)
                p_tr /= p_tr.sum(1, keepdims=True)
                out[t] = float((w_ * (p_tr * (np.log(p_tr) - np.log(p_ab))).sum(1)).sum())
        return out
    th = theory_kls()
    theory_future = float(np.mean([th[t] for t in (6, 7, 8, 9)]))
    theory_local = th[5]

    resid_rows, scrub_rows = [], []

    def probe_caps(caps, phase):
        for t in PROBE_STEPS:
            first, w_u = dedup[t]
            Xu = caps[t][first]
            role = "site" if t in ABL_STEPS else "down"
            if phase == "clean":
                role = "clean"
            for tgt, Zt in (("ker", Zk), ("col", Zc)):
                if phase != "clean" and tgt == "col" and role != "site":
                    continue
                Yu = Zt[t][first]
                r2l = probe_linear_cv(Xu, Yu, w_u, seed=1000 + t)
                r2m = float("nan")
                if not skip_mlp and t in MLP_STEPS and (phase == "clean" or tgt == "ker"):
                    r2m = probe_mlp_cv(Xu, Yu, w_u, device, seed=2000 + t)
                resid_rows.append({
                    "seed": seed, "arch": arch, "phase": phase, "step": t,
                    "role": role, "target": tgt, "n_uniq": len(first),
                    "r2_lin_cv": r2l, "r2_mlp_cv": r2m})
        print(f"  [{arch} s{seed}] probed phase={phase} ({time.time()-t0:.0f}s)",
              flush=True)

    probe_caps(caps_clean, "clean")

    # ---------------- Phase R: original pooled surgery ----------------
    terms = {t: [orig_term(t, "ker")] for t in ABL_STEPS}
    lg, caps = run_pass(terms)
    c = score(lg)
    print(f"  [{arch} s{seed}] ORIG ker: kl_local={c['kl_local']:.5f} "
          f"kl_future={c['kl_future']:.5f} (theory {theory_future:.4f})", flush=True)
    scrub_rows.append({"seed": seed, "arch": arch, "kind": "ker", "iter": 0,
                       "stage": "orig", "n_terms": 1, "r2_cv_max": float("nan"),
                       "theory_future": theory_future, "theory_local": theory_local,
                       **c})
    probe_caps(caps, "orig")

    # ---------------- Phase S: iterated per-step scrub ----------------
    terms = {t: [orig_term(t, "ker")] for t in ABL_STEPS}
    caps = None
    for it in range(max_it + 1):
        lg, caps = run_pass(terms)
        c = score(lg)
        r2max, fits = 0.0, {}
        for t in ABL_STEPS:
            first, w_u = dedup[t]
            Xu = caps[t][first]
            Yu = Zk[t][first]
            r2cv, ridge = ridge_cv_best(Xu, Yu, w_u, seed=1000 + t)
            fits[t] = (Xu, Yu, w_u, r2cv, ridge)
            if not np.isnan(r2cv):
                r2max = max(r2max, r2cv)
        scrub_rows.append({"seed": seed, "arch": arch, "kind": "ker", "iter": it,
                           "stage": "scrub", "n_terms": len(terms[ABL_STEPS[0]]),
                           "r2_cv_max": r2max, "theory_future": theory_future,
                           "theory_local": theory_local, **c})
        print(f"  [{arch} s{seed}] SCRUB it={it}: r2_cv_max={r2max:.5f} "
              f"kl_local={c['kl_local']:.5f} kl_future={c['kl_future']:.5f}",
              flush=True)
        if r2max < STOP_CV or it == max_it:
            break
        for t in ABL_STEPS:
            Xu, Yu, w_u, r2cv, ridge = fits[t]
            if np.isnan(r2cv) or r2cv <= TERM_CV:
                continue
            beta = fit_ridge_raw(Xu, Yu, w_u, ridge)
            zhat = np.hstack([np.ones((len(Xu), 1)), Xu]) @ beta
            zbar = (zhat * (w_u / w_u.sum())[:, None]).sum(0)
            gam = wfit(zhat, Xu, w_u)
            terms[t].append((beta[1:], beta[0] - zbar, gam[1:]))
    probe_caps(caps, "scrub")

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    f1 = out / f"resid_rnn_{arch.lower()}_s{seed}.csv"
    with open(f1, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(resid_rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(resid_rows)
    f2 = out / f"scrub_rnn_{arch.lower()}_s{seed}.csv"
    with open(f2, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(scrub_rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(scrub_rows)
    print(f"  saved {f1} and {f2}  ({time.time()-t0:.0f}s)")


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
