"""
Independent verification (from-scratch reimplementation) of the TRANSFORMER
causal-intervention numbers quoted in the paper (App. on interventions):

  V1 routing decomposition   branch_l0attn ~90% / prefix_upper ~18% / allpos6 ~10% / both=100
  V2 local readout           col-write at L3 ~96% R2~0.99, L2 ~90%, L1 inert; ker-write inert
  V3 ker erasure             iterated complete linear scrub: ker@L3 ~0.0004, ker@L2 ~0.0015,
                             col@L3 ~0.027 nats/token

Shares NOTHING with the causal_* analysis scripts: own process math, own probes,
own hooks, own metrics. Only the process definition (get_matrix_from_args) and
the checkpoint/config formats are common, by necessity.

Usage:
  PYTHONPATH=. uv run python scripts/activation_analysis/causal_verify_xf.py --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformer_lens import HookedTransformer, HookedTransformerConfig

from epsilon_transformers.process.transition_matrices import get_matrix_from_args

XF_DIR = Path("results/qrrxor_robust_transformer")
OUT_DIR = Path("results/causal_transplant/verify")
N_CTX = 10
P = 7                       # patch position: prefix length 7, position index 6
ROLL = N_CTX - P
EPS = 1e-12


# ---------------------------------------------------------------------------
# process math (independent implementation)
# ---------------------------------------------------------------------------

def process_setup(proc_cfg):
    pc = dict(proc_cfg); name = pc.pop("name")
    T = get_matrix_from_args(name=name, **pc)          # (vocab, d, d)
    vocab, d = T.shape[0], T.shape[1]
    Pm = T.sum(0)
    ev, evec = np.linalg.eig(Pm)                       # right eigvec: P rev = rev
    rev = np.real(evec[:, np.argmin(np.abs(ev - 1.0))]); rev /= rev.sum()
    ev2, evec2 = np.linalg.eig(Pm.T)                   # left eigvec: b0 P = b0
    b0 = np.real(evec2[:, np.argmin(np.abs(ev2 - 1.0))]); b0 /= b0 @ rev
    E = np.stack([T[x] @ rev for x in range(vocab)], 1)     # (d, vocab)
    Pcol = E @ np.linalg.pinv(E.T @ E) @ E.T
    Pker = np.eye(d) - Pcol
    return T, rev, b0, E, Pcol, Pker, vocab, d


def enum_prefixes(T, rev, b0, plen):
    """All vocab^plen prefixes in big-endian lex order: tokens, beliefs, weights, p0."""
    vocab, d = T.shape[0], T.shape[1]
    toks = np.array([[(i >> (plen - 1 - t)) & 1 for t in range(plen)]
                     for i in range(2 ** plen)], dtype=np.int64)
    bel = np.zeros((len(toks), d)); wt = np.zeros(len(toks))
    for i, tk in enumerate(toks):
        b = b0.copy(); pr = 1.0
        for x in tk:
            px = max(float(b @ T[x] @ rev), 0.0)
            pr *= px
            bn = b @ T[x]
            nz = float(bn @ rev)
            if nz > 0:
                b = bn / nz
        bel[i] = b; wt[i] = pr
    wt /= wt.sum()
    p0 = bel @ T[0] @ rev
    return toks, bel, wt, p0


def branch_mats(T, k):
    """Cumulative transition products for the 2^k branches (big-endian)."""
    branches = np.array([[(u >> (k - 1 - t)) & 1 for t in range(k)]
                         for u in range(2 ** k)], dtype=np.int64)
    mats = []
    for u in branches:
        M = np.eye(T.shape[1])
        for x in u:
            M = M @ T[x]
        mats.append(M)
    return branches, np.stack(mats)


def theory_tree(T, rev, bels, bmats):
    out = np.stack([bels @ M @ rev for M in bmats], 1)
    out = np.clip(out, 0, None)
    return out / out.sum(1, keepdims=True)


def tree_kl(Pt, Qt):
    Pt = np.clip(Pt, EPS, None); Pt = Pt / Pt.sum(-1, keepdims=True)
    Qt = np.clip(Qt, EPS, None); Qt = Qt / Qt.sum(-1, keepdims=True)
    return (Pt * (np.log(Pt) - np.log(Qt))).sum(-1)


# ---------------------------------------------------------------------------
# probes (independent implementation)
# ---------------------------------------------------------------------------

def wridge(X, Y, w, lam=1e-6):
    """Weighted ridge affine fit; returns beta ((D+1) x k)."""
    w = w / w.sum()
    Xb = np.hstack([np.ones((len(X), 1)), X])
    sw = np.sqrt(w)[:, None]
    A = (Xb * sw).T @ (Xb * sw) + lam * np.eye(Xb.shape[1])
    A[0, 0] -= lam                                    # don't penalize intercept
    return np.linalg.solve(A, (Xb * sw).T @ (Y * sw))


def wr2(X, Y, w, beta):
    w = w / w.sum()
    Yp = np.hstack([np.ones((len(X), 1)), X]) @ beta
    ym = (Y * w[:, None]).sum(0)
    ssr = float((w[:, None] * (Y - Yp) ** 2).sum())
    sst = float((w[:, None] * (Y - ym) ** 2).sum())
    return 1 - ssr / sst if sst > 0 else 0.0


def cv_ridge_r2(X, Y, w, seed=0, lams=(1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """5-fold CV pooled weighted R2, best over ridge grid; rows must be unique units.
    Returns (best_r2, best_lam)."""
    n = len(X)
    if n < 8:
        return float("nan"), float("nan")
    nf = 5 if n >= 20 else 4
    w = w / w.sum()
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), nf)
    ym = (Y * w[:, None]).sum(0)
    sst = float((w[:, None] * (Y - ym) ** 2).sum())
    if sst <= 0:
        return float("nan"), float("nan")
    best, blam = -np.inf, lams[0]
    for lam in lams:
        ssr = 0.0
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            beta = wridge(X[tr], Y[tr], w[tr], lam)
            Yp = np.hstack([np.ones((len(te), 1)), X[te]]) @ beta
            ssr += float((w[te][:, None] * (Y[te] - Yp) ** 2).sum())
        r2 = 1 - ssr / sst
        if r2 > best:
            best, blam = r2, lam
    return float(best), float(blam)


# ---------------------------------------------------------------------------
# model loading (config formats shared with training runs, by necessity)
# ---------------------------------------------------------------------------

def load_xf(run_dir, device):
    with open(run_dir / "hooked_model_config.json") as f:
        mc = json.load(f)
    mc["dtype"] = getattr(torch, mc["dtype"].split(".")[-1])
    mc["device"] = device
    model = HookedTransformer(HookedTransformerConfig(**mc)).to(device)
    pts = sorted(run_dir.glob("*.pt"), key=lambda p: int(p.stem))
    state = torch.load(pts[-1], map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def run_cfg(run_dir):
    with open(run_dir / "run_config.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# pair selection (replicates the published rule; robustness variant = all matched)
# ---------------------------------------------------------------------------

def matched_pairs(bel, wt, p0, Pker, n_pairs=200, tol=1e-3, all_matched=False):
    N = len(bel)
    ii, jj = np.triu_indices(N, k=1)
    dp0 = np.abs(p0[ii] - p0[jj])
    kern = np.linalg.norm((bel[ii] - bel[jj]) @ Pker, axis=1)
    ok = (dp0 < tol) & (np.minimum(wt[ii], wt[jj]) > np.quantile(wt, 0.10))
    if all_matched:
        keep = np.where(ok)[0]
    else:
        order = np.argsort(-kern)
        keep = order[ok[order]][:n_pairs]
    return np.concatenate([ii[keep], jj[keep]]), np.concatenate([jj[keep], ii[keep]])


def anti_pairs(wt, p0, n_pairs=200):
    N = len(wt)
    ii, jj = np.triu_indices(N, k=1)
    dp0 = np.abs(p0[ii] - p0[jj])
    ok = np.minimum(wt[ii], wt[jj]) > np.quantile(wt, 0.10)
    order = np.argsort(-dp0)
    keep = order[ok[order]][:n_pairs]
    return np.concatenate([ii[keep], jj[keep]]), np.concatenate([jj[keep], ii[keep]])


# ---------------------------------------------------------------------------
# V1: routing decomposition
# ---------------------------------------------------------------------------

HOOK_L1 = "blocks.1.hook_resid_pre"
HOOKS_UPPER6 = ["blocks.1.hook_resid_pre", "blocks.2.hook_resid_pre",
                "blocks.3.hook_resid_pre", "blocks.3.hook_resid_post"]
HOOK_A0 = "blocks.0.hook_attn_out"
ALL_HOOKS = HOOKS_UPPER6 + [HOOK_A0]


def softmax_at(model, x, hooks=None, bs=4096):
    outs = []
    for s in range(0, len(x), bs):
        xb = x[s:s + bs]
        with torch.no_grad():
            if hooks:
                lg = model.run_with_hooks(xb, fwd_hooks=hooks(s, s + len(xb)))
            else:
                lg = model(xb)
        outs.append(F.softmax(lg[:, P - 1:P - 1 + ROLL, :].float(), -1).cpu().numpy())
    return np.concatenate(outs)


def verify_routing(model, toks7, bel7, wt7, p07, Pker, T, rev, bmats, branches,
                   device, seed, rows, all_matched=False):
    nb = len(branches)
    seqs = np.concatenate([np.repeat(toks7, nb, 0), np.tile(branches, (len(toks7), 1))], 1)
    x_all = torch.from_numpy(seqs).long().to(device)
    cache_np = {}
    with torch.no_grad():
        _, cache = model.run_with_cache(x_all, names_filter=ALL_HOOKS)
    for h in ALL_HOOKS:
        cache_np[h] = cache[h].to(model.cfg.dtype)          # stays on device
    soft_clean = softmax_at(model, x_all)

    src, dst = matched_pairs(bel7, wt7, p07, Pker, all_matched=all_matched)
    npairs = len(src)
    rs = (np.repeat(src, nb) * nb + np.tile(np.arange(nb), npairs))
    rd = (np.repeat(dst, nb) * nb + np.tile(np.arange(nb), npairs))
    br_rep = np.tile(branches, (npairs, 1))

    def tree_of(soft_rows):
        sp = np.stack([soft_rows[np.arange(len(soft_rows)), s, br_rep[:, s]]
                       for s in range(ROLL)], 1)
        return np.prod(sp, 1).reshape(npairs, nb)

    Qu = tree_of(soft_clean[rs])
    Qv = tree_of(soft_clean[rd])
    th_w = theory_tree(T, rev, bel7[src], bmats)
    th_v = theory_tree(T, rev, bel7[dst], bmats)
    gap = np.median(tree_kl(Qu, Qv))

    KINDS = {
        "prefix_upper": [(HOOK_L1, slice(0, P))],
        "branch_l0attn": [(HOOK_A0, slice(P, None))],
        "allpos6": [(h, slice(P - 1, P)) for h in HOOKS_UPPER6],
        "both": [(HOOK_L1, slice(0, P)), (HOOK_A0, slice(P, None))],
    }
    x_src = x_all[rs]
    for kind, spec in KINDS.items():
        def mk(sl_lo, sl_hi):
            hooks = []
            for h, ps in spec:
                donor = cache_np[h][rd[sl_lo:sl_hi]]
                def fn(t, hook, donor=donor, ps=ps):
                    t = t.clone()
                    t[:, ps, :] = donor[:, ps, :]
                    return t
                hooks.append((h, fn))
            return hooks
        soft_p = softmax_at(model, x_src, hooks=mk)
        Qp = tree_of(soft_p)
        closed = 1 - np.median(tree_kl(Qp, Qv)) / max(gap, EPS)
        obs = np.log(np.clip(Qp, EPS, None)) - np.log(np.clip(Qu, EPS, None))
        pred = np.log(np.clip(th_v, EPS, None)) - np.log(np.clip(th_w, EPS, None))
        r2 = float(np.corrcoef(obs.ravel(), pred.ravel())[0, 1] ** 2)
        rows.append({"exp": "routing", "seed": seed, "kind": kind,
                     "sel": "all_matched" if all_matched else "top200",
                     "n_pairs": npairs, "pct_closed": 100 * closed, "r2": r2})
        print(f"  [xf s{seed} routing/{kind:14s} {'allm' if all_matched else 'top200'}] "
              f"closed={100*closed:6.1f}%  R2={r2:.3f}", flush=True)


# ---------------------------------------------------------------------------
# V2: local readout
# ---------------------------------------------------------------------------

def verify_local(model, toks7, bel7, wt7, p07, E, Pcol, Pker, device, seed, rows):
    src, dst = anti_pairs(wt7, p07)
    x7 = torch.from_numpy(toks7).long().to(device)
    sites = ["blocks.1.hook_resid_pre", "blocks.2.hook_resid_pre", "blocks.3.hook_resid_pre"]
    with torch.no_grad():
        _, cache = model.run_with_cache(x7, names_filter=sites)
    for site in sites:
        A = cache[site][:, -1, :].float().cpu().numpy()
        G = wridge(bel7, A, wt7)[1:]                     # belief -> resid decoder
        d_eta = bel7[dst] - bel7[src]
        writes = {"col": (d_eta @ Pcol) @ G, "ker": (d_eta @ Pker) @ G}
        x_src = x7[src]
        p_unp = p07[src]                                 # theory p0 == model p0 approx; use model
        with torch.no_grad():
            lg = model(x_src)
        p_unp = F.softmax(lg[:, P - 1, :].float(), -1)[:, 0].cpu().numpy()
        for kind, dl in writes.items():
            dlt = torch.from_numpy(dl).to(device, model.cfg.dtype)
            def fn(t, hook):
                t = t.clone()
                t[:, P - 1, :] += dlt
                return t
            with torch.no_grad():
                lg = model.run_with_hooks(x_src, fwd_hooks=[(site, fn)])
            p_pat = F.softmax(lg[:, P - 1, :].float(), -1)[:, 0].cpu().numpy()
            def kl2(a, b):
                a = np.clip(np.stack([a, 1 - a], -1), EPS, 1)
                b = np.clip(np.stack([b, 1 - b], -1), EPS, 1)
                return (a * (np.log(a) - np.log(b))).sum(-1)
            closed = 1 - np.median(kl2(p_pat, p07[dst])) / max(
                np.median(kl2(p_unp, p07[dst])), EPS)
            lo = lambda q: np.log(np.clip(q, EPS, 1 - EPS) / np.clip(1 - q, EPS, 1))
            obs = lo(p_pat) - lo(p_unp)
            pred = lo(p07[dst]) - lo(p07[src])
            r2 = float(np.corrcoef(obs, pred)[0, 1] ** 2) if kind == "col" else float("nan")
            rows.append({"exp": "local", "seed": seed, "kind": kind,
                         "sel": "L" + site.split(".")[1],
                         "n_pairs": len(src), "pct_closed": 100 * closed, "r2": r2})
            print(f"  [xf s{seed} local/{site} {kind}] closed={100*closed:6.1f}% R2={r2:.3f}",
                  flush=True)


# ---------------------------------------------------------------------------
# V3: iterated complete linear erasure
# ---------------------------------------------------------------------------

def verify_erasure(model, T, rev, b0, Pcol, Pker, device, seed, rows,
                   max_it=8, stop_cv=0.02):
    vocab = T.shape[0]
    toks, _, wt, _ = enum_prefixes(T, rev, b0, N_CTX)
    w_ = wt / wt.sum()
    x = torch.from_numpy(toks).long().to(device)

    def obasis(Pmat):
        lam, V = np.linalg.eigh(Pmat)
        return V[:, lam > 0.5]
    K, C = obasis(Pker), obasis(Pcol)

    pos_bel, true_cond, dedup = {}, {}, {}
    for p in range(2, N_CTX):
        _, bg, _, _ = enum_prefixes(T, rev, b0, p + 1)
        ids = np.array([int("".join(map(str, tk[:p + 1])), 2) for tk in toks])
        pos_bel[p] = bg[ids]
        pc = np.stack([np.clip(pos_bel[p] @ T[xx] @ rev, EPS, None) for xx in range(vocab)], 1)
        true_cond[p] = pc / pc.sum(1, keepdims=True)
        uq, first, inv = np.unique(ids, return_index=True, return_inverse=True)
        dedup[p] = (first, np.bincount(inv, weights=wt))

    with torch.no_grad():
        lg = model(x)
    soft_clean = F.softmax(lg.float(), -1).cpu().numpy()
    loss_clean = sum(-(w_[:, None] * true_cond[p] *
                       np.log(np.clip(soft_clean[:, p, :], EPS, None))).sum()
                     for p in range(2, N_CTX)) / 8

    for kind, site in (("ker", "blocks.3.hook_resid_pre"),
                       ("ker", "blocks.2.hook_resid_pre"),
                       ("ker", "blocks.1.hook_resid_pre"),
                       ("col", "blocks.3.hook_resid_pre")):
        B = K if kind == "ker" else C
        Zt = {p: pos_bel[p] @ B for p in range(2, N_CTX)}
        terms = {p: [] for p in range(2, N_CTX)}

        def fwd(capture=False):
            store = {}
            def fn(t, hook):
                t = t.clone()
                for p in range(2, N_CTX):
                    a = t[:, p, :]
                    for (bt, zbar, gt) in terms[p]:
                        ones = torch.ones(len(a), 1, dtype=a.dtype, device=a.device)
                        z = torch.cat([ones, a], 1) @ bt
                        a = a - (z - zbar) @ gt
                    t[:, p, :] = a
                if capture:
                    store["acts"] = t.float().cpu().numpy()
                return t
            with torch.no_grad():
                lgp = model.run_with_hooks(x, fwd_hooks=[(site, fn)])
            return F.softmax(lgp.float(), -1).cpu().numpy(), store.get("acts")

        r2max, it = np.inf, 0
        for it in range(max_it + 1):
            soft, acts = fwd(capture=True)
            r2max, fits = 0.0, {}
            for p in range(2, N_CTX):
                first, w_u = dedup[p]
                Xu, Yu = acts[first, p, :], Zt[p][first]
                r2cv, lam = cv_ridge_r2(Xu, Yu, w_u, seed=7 * p)
                fits[p] = (Xu, Yu, w_u, r2cv, lam)
                if not np.isnan(r2cv):
                    r2max = max(r2max, r2cv)
            if r2max < stop_cv or it == max_it:
                break
            for p, (Xu, Yu, w_u, r2cv, lam) in fits.items():
                if np.isnan(r2cv) or r2cv <= stop_cv:
                    continue
                beta = wridge(Xu, Yu, w_u, lam if not np.isnan(lam) else 1e-4)
                zhat = np.hstack([np.ones((len(Xu), 1)), Xu]) @ beta
                zbar = (zhat * (w_u / w_u.sum())[:, None]).sum(0)
                gam = wridge(zhat, Xu, w_u)[1:]        # slope only; mean preserved
                dt = model.cfg.dtype
                terms[p].append((torch.tensor(beta, dtype=dt, device=device),
                                 torch.tensor(zbar, dtype=dt, device=device),
                                 torch.tensor(gam, dtype=dt, device=device)))
        loss_abl = sum(-(w_[:, None] * true_cond[p] *
                         np.log(np.clip(soft[:, p, :], EPS, None))).sum()
                       for p in range(2, N_CTX)) / 8
        dloss = float(loss_abl - loss_clean)
        lab = "L" + site.split(".")[1]
        rows.append({"exp": "erasure", "seed": seed, "kind": kind, "sel": lab,
                     "n_pairs": it, "pct_closed": float("nan"), "r2": r2max,
                     "dloss": dloss})
        print(f"  [xf s{seed} erasure/{kind}@{lab}] dloss={dloss:+.5f} "
              f"resid_cv_r2={r2max:.4f} iters={it}", flush=True)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    run_dirs = sorted(XF_DIR.glob("run_*"))
    cfg = run_cfg(run_dirs[0])
    T, rev, b0, E, Pcol, Pker, vocab, d = process_setup(cfg["process_config"])
    toks7, bel7, wt7, p07 = enum_prefixes(T, rev, b0, P)
    branches, bmats = branch_mats(T, ROLL)

    for rd in run_dirs:
        seed = run_cfg(rd)["model_config"]["seed"]
        model = load_xf(rd, args.device)
        # sanity: clean model ~ theory at the patch position
        x7 = torch.from_numpy(toks7).long().to(args.device)
        with torch.no_grad():
            lg = model(x7)
        pm = F.softmax(lg[:, P - 1, :].float(), -1)[:, 0].cpu().numpy()
        klm = float((wt7 * (p07 * np.log(np.clip(p07, EPS, 1) / np.clip(pm, EPS, 1))
                            + (1 - p07) * np.log(np.clip(1 - p07, EPS, 1)
                                                 / np.clip(1 - pm, EPS, 1)))).sum())
        print(f"[xf s{seed}] sanity: weighted KL(theory||model) at pos6 = {klm:.5f}",
              flush=True)
        rows.append({"exp": "sanity", "seed": seed, "kind": "kl_theory_model",
                     "sel": "", "n_pairs": 128, "pct_closed": float("nan"), "r2": klm})

        verify_routing(model, toks7, bel7, wt7, p07, Pker, T, rev, bmats, branches,
                       args.device, seed, rows)
        verify_routing(model, toks7, bel7, wt7, p07, Pker, T, rev, bmats, branches,
                       args.device, seed, rows, all_matched=True)
        verify_local(model, toks7, bel7, wt7, p07, E, Pcol, Pker, args.device, seed, rows)
        verify_erasure(model, T, rev, b0, Pcol, Pker, args.device, seed, rows)
        del model
        torch.cuda.empty_cache()

    f = OUT_DIR / "verify_xf.csv"
    keys = ["exp", "seed", "kind", "sel", "n_pairs", "pct_closed", "r2", "dloss"]
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"saved {f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
