"""
Independent verification (from-scratch reimplementation) of the RECURRENT
causal-intervention numbers quoted in the paper (App. on interventions):

  V4 transplant    ntp-neutral: GRU 96.0 / LSTM 95.2 / RNN 92.7 % (R2 .87/.84/.81),
                   step-1 KL <= 5e-14 on all 16,256 pairs; 2-dim ker-decoder write:
                   GRU 89.3 / LSTM 83.4 / RNN 59.5 %; matched-norm random ~10%.
                   NEW: adjudicates whether patched runs land on the chimera target
                   (col_w + ker_v) or the donor conditional (hybrid: w's step-1,
                   v's steps 2-3) — the two differ on anti-matched pairs.
  V5 untrained     untrained GRU: belief probe still fits, transplant steers ~0%
                   (the control cited in all four responses; artifact was missing).
  V6 erasure       carried-state complete linear ker scrub at steps 3-6, all 5 seeds
                   x {RNN, GRU, LSTM}: kl_future vs exact theory; col contrast with
                   its own theory prediction (double dissociation, both directions).

Own process math, probes, stepping, and metrics throughout (same independence
policy as causal_verify_xf.py).

Usage:
  PYTHONPATH=. uv run python scripts/activation_analysis/causal_verify_rnn.py --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from epsilon_transformers.analysis.intervention_verify import (
    process_setup, enum_prefixes, branch_mats, theory_tree, tree_kl, wridge, wr2, cv_ridge_r2,
    OUT_DIR, N_CTX, P, ROLL, EPS, FAMS, build, states_last, rollout, hybrid_tree,
    pooled_probe,
)


# ---------------------------------------------------------------------------
# V4 / V5: transplant battery
# ---------------------------------------------------------------------------

def transplant(model, fam, seed, T, rev, b0, E, Pcol, Pker, vocab, device,
               rows, n_rand=3):
    is_lstm = fam == "LSTM"
    L = model.rnn.num_layers
    H = model.rnn.hidden_size
    toks7, bel7, wt7, p07 = enum_prefixes(T, rev, b0, P)
    branches, bmats = branch_mats(T, ROLL)
    A_t = states_last(model, toks7, vocab, device, is_lstm)
    A = A_t.cpu().numpy()
    D = A.shape[1]

    # identity check: rollout from state == full-sequence forward
    nb = len(branches)
    seqs = np.concatenate([np.repeat(toks7[:8], nb, 0), np.tile(branches, (8, 1))], 1)
    x = torch.from_numpy(seqs).long().to(device)
    with torch.no_grad():
        lg = model.output_layer(model.rnn(F.one_hot(x, vocab).float())[0])
    full_soft = F.softmax(lg[:, P - 1:P - 1 + ROLL, :], -1).cpu().numpy()
    full_steps = np.stack([full_soft[np.arange(len(seqs)), s,
                                     np.tile(branches, (8, 1))[:, s]]
                           for s in range(ROLL)], 1).reshape(8, nb, ROLL)
    ro = rollout(model, A_t[:8], branches, L, H, is_lstm, device)
    iden = float(np.abs(ro - full_steps).max())
    assert iden < 1e-3, f"rollout != full forward (max diff {iden:.2e})"
    print(f"  [{fam} s{seed}] identity check: rollout vs full forward "
          f"max|dp|={iden:.1e}", flush=True)

    # probe / decoder (pooled 4..9) + ntp direction
    Xp, Yp, Wp, beta = pooled_probe(model, T, rev, b0, vocab, device, is_lstm)
    G = wridge(Yp, Xp, Wp, 1e-6)[1:]
    r2_in = wr2(Xp, Yp, Wp, beta)
    r2_cv, _ = cv_ridge_r2(Xp, Yp, Wp, seed=3)
    Wout = model.output_layer.weight.detach().float().cpu().numpy()
    u = np.zeros(D)
    u[(L - 1) * H:L * H] = Wout[0] - Wout[1]
    u /= np.linalg.norm(u)

    ii, jj = np.triu_indices(len(A), k=1)
    src = np.concatenate([ii, jj]); dst = np.concatenate([jj, ii])
    d_full = A[dst] - A[src]
    d_neut = d_full - np.outer(d_full @ u, u)
    d_ker = ((bel7[dst] - bel7[src]) @ Pker) @ G

    eta_chi = bel7[src] + (bel7[dst] - bel7[src]) @ Pker
    th_chi = theory_tree(T, rev, eta_chi, bmats)
    th_w = theory_tree(T, rev, bel7[src], bmats)
    th_v = theory_tree(T, rev, bel7[dst], bmats)
    th_hyb = hybrid_tree(T, rev, bel7[src], bel7[dst], branches)

    base = rollout(model, A_t, branches, L, H, is_lstm, device)
    Qu = np.prod(base, 2)[src]
    Qu_step1 = base[:, 0, 0][src]  # p of branch-0 token at step 1 (token 0)

    rng = np.random.default_rng(0)
    jobs = [("ntp_neutral", d_neut), ("ker_write", d_ker), ("fullswap", d_full)]
    for k in range(n_rand):
        r = rng.standard_normal(d_neut.shape)
        r -= np.outer(r @ u, u)
        r *= (np.linalg.norm(d_neut, axis=1) /
              np.maximum(np.linalg.norm(r, axis=1), EPS))[:, None]
        jobs.append((f"rand_neutral_{k}", r))

    for kind, delta in jobs:
        sp = rollout(model, A[src] + delta, branches, L, H, is_lstm, device)
        Qp = np.prod(sp, 2)
        # step-1 invisibility: binary KL patched vs unpatched at step 1
        p1p, p1u = sp[:, 0, 0], Qu_step1
        def bkl(a, b):
            a = np.clip(np.stack([a, 1 - a], -1), EPS, 1)
            b = np.clip(np.stack([b, 1 - b], -1), EPS, 1)
            return (a * (np.log(a) - np.log(b))).sum(-1)
        kl1_max = float(bkl(p1p, p1u).max())
        res = {"exp": "transplant", "family": fam, "seed": seed, "kind": kind,
               "n_pairs": len(src), "kl1_max": kl1_max,
               "probe_r2_in": r2_in, "probe_r2_cv": r2_cv,
               "norm_ratio": float(np.median(np.linalg.norm(delta, axis=1)
                                             / np.linalg.norm(A[src], axis=1)))}
        for tgt_name, tht in (("chi", th_chi), ("hyb", th_hyb), ("v", th_v)):
            pct = 1 - np.median(tree_kl(Qp, tht)) / max(np.median(tree_kl(Qu, tht)), EPS)
            obs = np.log(np.clip(Qp, EPS, None)) - np.log(np.clip(Qu, EPS, None))
            pred = np.log(np.clip(tht, EPS, None)) - np.log(np.clip(th_w, EPS, None))
            r2 = float(np.corrcoef(obs.ravel(), pred.ravel())[0, 1] ** 2)
            res[f"pct_{tgt_name}"] = 100 * pct
            res[f"r2_{tgt_name}"] = r2
        rows.append(res)
        print(f"  [{fam} s{seed} {kind:14s}] kl1max={kl1_max:.1e} "
              f"chi={res['pct_chi']:5.1f}% hyb={res['pct_hyb']:5.1f}% v={res['pct_v']:5.1f}% "
              f"R2chi={res['r2_chi']:.3f} R2hyb={res['r2_hyb']:.3f} "
              f"(probe in/cv {r2_in:.3f}/{r2_cv:.3f})", flush=True)


# ---------------------------------------------------------------------------
# V6: carried-state erasure with exact theory (ker AND col)
# ---------------------------------------------------------------------------

def erasure(model, fam, seed, T, rev, b0, Pcol, Pker, vocab, device, rows,
            max_it=8, stop_cv=0.02):
    is_lstm = fam == "LSTM"
    L, H = model.rnn.num_layers, model.rnn.hidden_size
    ABL = (3, 4, 5, 6)
    toks, _, wt, _ = enum_prefixes(T, rev, b0, N_CTX)
    w_ = wt / wt.sum()
    x = torch.from_numpy(toks).long().to(device)
    oh = F.one_hot(x, vocab).float()

    def obasis(Pm):
        lam, V = np.linalg.eigh(Pm)
        return V[:, lam > 0.5]

    pos_bel, true_cond, dedup = {}, {}, {}
    for t in range(1, N_CTX + 1):
        _, bg, _, _ = enum_prefixes(T, rev, b0, t)
        ids = np.array([int("".join(map(str, tk[:t])), 2) for tk in toks])
        pos_bel[t] = bg[ids]
        if t <= N_CTX:
            pc = np.stack([np.clip(pos_bel[t] @ T[xx] @ rev, EPS, None)
                           for xx in range(vocab)], 1)
            true_cond[t] = pc / pc.sum(1, keepdims=True)
        uq, first, inv = np.unique(ids, return_index=True, return_inverse=True)
        dedup[t] = (first, np.bincount(inv, weights=wt))

    n_parts = 2 if is_lstm else 1

    def fwd(terms):
        h = None
        caps, logits = {}, []
        with torch.no_grad():
            for t in range(N_CTX):
                o, h = model.rnn(oh[:, t:t + 1], h)
                step = t + 1
                if terms is not None and step in ABL:
                    parts = h if is_lstm else (h,)
                    v = torch.cat([p[i] for p in parts for i in range(L)], 1)
                    for (bt, zbar, gt) in terms[step]:
                        ones = torch.ones(len(v), 1, device=device)
                        z = torch.cat([ones, v], 1) @ bt
                        v = v - (z - zbar) @ gt
                    blocks = v.view(-1, n_parts * L, H).permute(1, 0, 2).contiguous()
                    h = (blocks[:L].contiguous(), blocks[L:].contiguous()) \
                        if is_lstm else blocks.contiguous()
                    caps[step] = v.cpu().numpy()
                    logits.append(model.output_layer(v[:, (L - 1) * H:L * H]).unsqueeze(1))
                else:
                    logits.append(model.output_layer(o))
        return F.softmax(torch.cat(logits, 1).float(), -1).cpu().numpy(), caps

    soft_clean, _ = fwd(None)

    def score(soft):
        kls = {}
        for t in range(2, N_CTX):
            pcl = np.clip(soft_clean[:, t, :], EPS, None); pcl /= pcl.sum(1, keepdims=True)
            pab = np.clip(soft[:, t, :], EPS, None); pab /= pab.sum(1, keepdims=True)
            kls[t] = float((w_ * (pcl * (np.log(pcl) - np.log(pab))).sum(1)).sum())
        return kls

    def theory(Pmat):
        bel = np.tile(b0, (len(toks), 1))
        th = {}
        for t in range(N_CTX):
            xt = toks[:, t]
            nb = np.einsum("ni,xij->nxj", bel, T)[np.arange(len(toks)), xt]
            bel = nb / np.clip(nb @ rev, EPS, None)[:, None]
            if (t + 1) in ABL:
                mb = (pos_bel[t + 1] * w_[:, None]).sum(0)
                bel = bel - (bel - mb) @ Pmat
            if 2 <= t < N_CTX:
                pa = np.stack([np.clip(bel @ T[xx] @ rev, EPS, None)
                               for xx in range(vocab)], 1)
                pa /= pa.sum(1, keepdims=True)
                th[t] = float((w_ * (true_cond[t + 1] *
                                     (np.log(true_cond[t + 1]) - np.log(pa))).sum(1)).sum())
        return th

    for kind, Pmat in (("ker", Pker), ("col", Pcol)):
        B = obasis(Pmat)
        Zt = {s: pos_bel[s] @ B for s in ABL}
        terms = {s: [] for s in ABL}
        for it in range(max_it + 1):
            soft, caps = fwd(terms)
            r2max, fits = 0.0, {}
            for s in ABL:
                first, w_u = dedup[s]
                Xu, Yu = caps[s][first], Zt[s][first]
                r2cv, lam = cv_ridge_r2(Xu, Yu, w_u, seed=11 * s)
                fits[s] = (Xu, Yu, w_u, r2cv, lam)
                if not np.isnan(r2cv):
                    r2max = max(r2max, r2cv)
            if r2max < stop_cv or it == max_it:
                break
            for s, (Xu, Yu, w_u, r2cv, lam) in fits.items():
                if np.isnan(r2cv) or r2cv <= stop_cv:
                    continue
                beta = wridge(Xu, Yu, w_u, lam if not np.isnan(lam) else 1e-4)
                zhat = np.hstack([np.ones((len(Xu), 1)), Xu]) @ beta
                zbar = (zhat * (w_u / w_u.sum())[:, None]).sum(0)
                gam = wridge(zhat, Xu, w_u)[1:]
                terms[s].append((torch.tensor(beta, dtype=torch.float32, device=device),
                                 torch.tensor(zbar, dtype=torch.float32, device=device),
                                 torch.tensor(gam, dtype=torch.float32, device=device)))
        kls = score(soft)
        th = theory(Pmat)
        kl_future = float(np.mean([kls[t] for t in (6, 7, 8, 9)]))
        th_future = float(np.mean([th[t] for t in (6, 7, 8, 9)]))
        kl_first = kls[2]                     # right after first erased state
        rows.append({"exp": "erasure", "family": fam, "seed": seed, "kind": kind,
                     "kl_first": kl_first, "kl_local": kls[5], "kl_future": kl_future,
                     "th_local": th[5], "th_future": th_future,
                     "resid_r2": r2max, "iters": it})
        print(f"  [{fam} s{seed} erase-{kind}] kl_first={kl_first:.2e} "
              f"local={kls[5]:.4f} (th {th[5]:.4f}) future={kl_future:.4f} "
              f"(th {th_future:.4f}) resid_r2={r2max:.3f} it={it}", flush=True)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--families", default="GRU,LSTM,RNN")
    args = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    fams = args.families.split(",")

    for dpath, fam in FAMS:
        if fam not in fams:
            continue
        for rd in sorted(Path(dpath).glob("run_*")):
            with open(rd / "run_config.yaml") as f:
                cfg = yaml.safe_load(f)
            seed = cfg["model_config"]["seed"]
            T, rev, b0, E, Pcol, Pker, vocab, d = process_setup(cfg["process_config"])
            pts = sorted(rd.glob("*.pt"), key=lambda p: int(p.stem))
            state = torch.load(pts[-1], map_location=args.device, weights_only=True)
            model = build(cfg["model_config"], vocab, args.device, state=state)
            transplant(model, fam, seed, T, rev, b0, E, Pcol, Pker, vocab,
                       args.device, rows)
            erasure(model, fam, seed, T, rev, b0, Pcol, Pker, vocab, args.device, rows)
            del model
            torch.cuda.empty_cache()

    # untrained control (GRU architecture, 5 fresh inits)
    if "UNTRAINED" in fams or "GRU" in fams:
        with open(sorted(Path(FAMS[0][0]).glob("run_*"))[0] / "run_config.yaml") as f:
            cfg = yaml.safe_load(f)
        T, rev, b0, E, Pcol, Pker, vocab, d = process_setup(cfg["process_config"])
        for k in range(5):
            model = build(cfg["model_config"], vocab, args.device, init_seed=1000 + k)
            transplant(model, "GRU_untrained", 1000 + k, T, rev, b0, E, Pcol, Pker,
                       vocab, args.device, rows, n_rand=1)
            del model
            torch.cuda.empty_cache()

    keys = sorted({k for r in rows for k in r})
    f = OUT_DIR / "verify_rnn.csv"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"saved {f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
