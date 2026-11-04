"""
Does ADDING readout-orthogonal noise do nothing? (Answer to author question.)

Transformer: additive gaussian noise orthogonal to the unembedding contrast
w_U = W_U[:,0]-W_U[:,1], at resid_pre L3 and resid_post L3, all positions 2..9,
row-norm-matched to the ker-erasure deformation. Compare dloss to the ker
erasure (~0.0004) and col erasure (~0.026).

Vanilla RNN: additive gaussian noise orthogonal to the exact local readout
direction u, at carried-state steps 3..6, row-norm-matched to the ker-erasure
deformation. Compare kl_future to ker erasure (0.019 = theory) and col (~0.001).

Usage:
  PYTHONPATH=. uv run python scripts/activation_analysis/causal_verify_orthonoise.py --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from scripts.activation_analysis.causal_verify_xf import (
    process_setup, enum_prefixes, wridge, cv_ridge_r2, load_xf, run_cfg,
    XF_DIR, OUT_DIR, EPS,
)
from scripts.activation_analysis.causal_verify_rnn import build, FAMS

N_CTX = 10
N_DRAWS = 3


def obasis(Pm):
    lam, V = np.linalg.eigh(Pm)
    return V[:, lam > 0.5]


def first_term_deltas(acts_by_pos, Z_by_pos, dedup, positions):
    """Fit one encoder+decoder term per position on clean acts; return the
    per-row deformation (what the erasure would subtract) and its norms."""
    deltas = {}
    for p in positions:
        first, w_u = dedup[p]
        Xu, Yu = acts_by_pos[p][first], Z_by_pos[p][first]
        r2cv, lam = cv_ridge_r2(Xu, Yu, w_u, seed=7 * p)
        beta = wridge(Xu, Yu, w_u, lam if not np.isnan(lam) else 1e-4)
        X_all = acts_by_pos[p]
        zhat = np.hstack([np.ones((len(X_all), 1)), X_all]) @ beta
        zbar = (zhat[first] * (w_u / w_u.sum())[:, None]).sum(0)
        gam = wridge(zhat[first], Xu, w_u)[1:]
        deltas[p] = (zhat - zbar) @ gam
    return deltas


def ortho_noise_like(deltas, ortho_dir, rng):
    """Gaussian noise rows orthogonal to ortho_dir, norm-matched row-wise."""
    out = {}
    u = ortho_dir / np.linalg.norm(ortho_dir)
    for p, dl in deltas.items():
        r = rng.standard_normal(dl.shape)
        r -= np.outer(r @ u, u)
        r *= (np.linalg.norm(dl, axis=1) /
              np.maximum(np.linalg.norm(r, axis=1), EPS))[:, None]
        out[p] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rows = []
    POS = tuple(range(2, N_CTX))

    # ---------------- transformer ----------------
    run_dirs = sorted(XF_DIR.glob("run_*"))
    cfg = run_cfg(run_dirs[0])
    T, rev, b0, E, Pcol, Pker, vocab, d = process_setup(cfg["process_config"])
    K = obasis(Pker)
    toks, _, wt, _ = enum_prefixes(T, rev, b0, N_CTX)
    w_ = wt / wt.sum()
    pos_bel, true_cond, dedup = {}, {}, {}
    for p in POS:
        _, bg, _, _ = enum_prefixes(T, rev, b0, p + 1)
        ids = np.array([int("".join(map(str, tk[:p + 1])), 2) for tk in toks])
        pos_bel[p] = bg[ids]
        pc = np.stack([np.clip(pos_bel[p] @ T[xx] @ rev, EPS, None) for xx in range(vocab)], 1)
        true_cond[p] = pc / pc.sum(1, keepdims=True)
        uq, first, inv = np.unique(ids, return_index=True, return_inverse=True)
        dedup[p] = (first, np.bincount(inv, weights=wt))
    Z = {p: pos_bel[p] @ K for p in POS}

    for rd in run_dirs:
        seed = run_cfg(rd)["model_config"]["seed"]
        model = load_xf(rd, args.device)
        x = torch.from_numpy(toks).long().to(args.device)
        wu = (model.W_U[:, 0] - model.W_U[:, 1]).detach().float().cpu().numpy()
        for site in ("blocks.3.hook_resid_pre", "blocks.3.hook_resid_post"):
            with torch.no_grad():
                lg, cache = model.run_with_cache(x, names_filter=[site])
            soft_clean = F.softmax(lg.float(), -1).cpu().numpy()
            A = cache[site].float().cpu().numpy()
            acts = {p: A[:, p, :] for p in POS}
            loss_clean = sum(-(w_[:, None] * true_cond[p] *
                               np.log(np.clip(soft_clean[:, p, :], EPS, None))).sum()
                             for p in POS) / len(POS)
            deltas = first_term_deltas(acts, Z, dedup, POS)
            resid_ratio = float(np.mean([np.linalg.norm(deltas[p], axis=1).mean()
                                         / np.linalg.norm(acts[p], axis=1).mean()
                                         for p in POS]))
            rng = np.random.default_rng(0)
            dls = []
            for k in range(N_DRAWS):
                noise = ortho_noise_like(deltas, wu, rng)
                nz = {p: torch.tensor(noise[p], dtype=model.cfg.dtype,
                                      device=args.device) for p in POS}
                def fn(t, hook):
                    t = t.clone()
                    for p in POS:
                        t[:, p, :] += nz[p]
                    return t
                with torch.no_grad():
                    lgp = model.run_with_hooks(x, fwd_hooks=[(site, fn)])
                soft = F.softmax(lgp.float(), -1).cpu().numpy()
                loss = sum(-(w_[:, None] * true_cond[p] *
                             np.log(np.clip(soft[:, p, :], EPS, None))).sum()
                           for p in POS) / len(POS)
                dls.append(float(loss - loss_clean))
            rows.append({"arch": "transformer", "seed": seed, "site": site,
                         "kind": "addnoise_ortho_readout",
                         "dloss": float(np.mean(dls)), "norm_vs_resid": resid_ratio})
            print(f"[xf s{seed} {site}] add-noise (ker-erasure-matched norm, "
                  f"orth. to w_U; {100*resid_ratio:.1f}% of resid) "
                  f"dloss={np.mean(dls):+.5f}", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---------------- vanilla RNN ----------------
    for rd in sorted(Path(FAMS[2][0]).glob("run_*")):
        with open(rd / "run_config.yaml") as f:
            rcfg = yaml.safe_load(f)
        seed = rcfg["model_config"]["seed"]
        T, rev, b0, E, Pcol, Pker, vocab, d = process_setup(rcfg["process_config"])
        K = obasis(Pker)
        pts = sorted(rd.glob("*.pt"), key=lambda p: int(p.stem))
        state = torch.load(pts[-1], map_location=args.device, weights_only=True)
        model = build(rcfg["model_config"], vocab, args.device, state=state)
        L, H = model.rnn.num_layers, model.rnn.hidden_size
        toks, _, wt, _ = enum_prefixes(T, rev, b0, N_CTX)
        w_ = wt / wt.sum()
        oh = F.one_hot(torch.from_numpy(toks).long().to(args.device), vocab).float()
        ABL = (3, 4, 5, 6)
        pos_bel_t, dedup_t = {}, {}
        for t in range(1, N_CTX + 1):
            _, bg, _, _ = enum_prefixes(T, rev, b0, t)
            ids = np.array([int("".join(map(str, tk[:t])), 2) for tk in toks])
            pos_bel_t[t] = bg[ids]
            uq, first, inv = np.unique(ids, return_index=True, return_inverse=True)
            dedup_t[t] = (first, np.bincount(inv, weights=wt))
        Zt = {s: pos_bel_t[s] @ K for s in ABL}
        Wout = model.output_layer.weight.detach().float().cpu().numpy()
        u = np.zeros(L * H)
        u[(L - 1) * H:L * H] = Wout[0] - Wout[1]

        def fwd(noise=None):
            h = None
            caps, logits = {}, []
            with torch.no_grad():
                for t in range(N_CTX):
                    o, h = model.rnn(oh[:, t:t + 1], h)
                    step = t + 1
                    if step in ABL:
                        v = torch.cat([h[i] for i in range(L)], 1)
                        if noise is not None:
                            v = v + noise[step]
                        h = v.view(-1, L, H).permute(1, 0, 2).contiguous()
                        caps[step] = v.cpu().numpy()
                        logits.append(model.output_layer(v[:, (L - 1) * H:]).unsqueeze(1))
                    else:
                        logits.append(model.output_layer(o))
            return F.softmax(torch.cat(logits, 1).float(), -1).cpu().numpy(), caps

        soft_clean, caps_clean = fwd(None)
        deltas = first_term_deltas(caps_clean, Zt, dedup_t, ABL)
        rng = np.random.default_rng(0)
        futs, locs = [], []
        for k in range(N_DRAWS):
            noise = ortho_noise_like(deltas, u, rng)
            nz = {s: torch.tensor(noise[s], dtype=torch.float32, device=args.device)
                  for s in ABL}
            soft, _ = fwd(nz)
            kls = {}
            for t in range(2, N_CTX):
                pcl = np.clip(soft_clean[:, t, :], EPS, None); pcl /= pcl.sum(1, keepdims=True)
                pab = np.clip(soft[:, t, :], EPS, None); pab /= pab.sum(1, keepdims=True)
                kls[t] = float((w_ * (pcl * (np.log(pcl) - np.log(pab))).sum(1)).sum())
            futs.append(np.mean([kls[t] for t in (6, 7, 8, 9)]))
            locs.append(kls[5])
        rows.append({"arch": "RNN", "seed": seed, "site": "carried_state",
                     "kind": "addnoise_ortho_readout",
                     "kl_future": float(np.mean(futs)), "kl_local": float(np.mean(locs))})
        print(f"[RNN s{seed}] add-noise (ker-erasure-matched norm, orth. to u) "
              f"kl_local={np.mean(locs):.4f} kl_future={np.mean(futs):.4f} "
              f"(ker-erasure future was 0.019-0.030; col ~0.001)", flush=True)
        del model
        torch.cuda.empty_cache()

    f = OUT_DIR / "verify_orthonoise.csv"
    keys = sorted({k for r in rows for k in r})
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"saved {f}")


if __name__ == "__main__":
    main()
