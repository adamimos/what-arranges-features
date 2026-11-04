"""
Target adjudication on ANTI-MATCHED pairs (top-quartile |dp0|), where the
chimera target (w's next-token part + v's beyond-NTP part) and the donor
conditional (hybrid: w's pinned step-1, v's dynamics after) actually differ.

Prediction if the 2-dim ker-decoder write is a genuine belief-coordinate write:
  ker_write   -> tracks CHIMERA (cannot carry v's col part)
  ntp_neutral -> tracks DONOR/HYBRID (carries everything but the readout direction)

Usage:
  PYTHONPATH=. uv run python scripts/activation_analysis/causal_verify_target_adjudication.py --device cuda:1
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from epsilon_transformers.analysis.intervention_verify import (
    process_setup, enum_prefixes, branch_mats, theory_tree, tree_kl, wridge, build,
    states_last, rollout, hybrid_tree, pooled_probe, FAMS, OUT_DIR, P, ROLL, EPS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    rows = []
    for dpath, fam in FAMS:
        for rd in sorted(Path(dpath).glob("run_*")):
            with open(rd / "run_config.yaml") as f:
                cfg = yaml.safe_load(f)
            seed = cfg["model_config"]["seed"]
            T, rev, b0, E, Pcol, Pker, vocab, d = process_setup(cfg["process_config"])
            pts = sorted(rd.glob("*.pt"), key=lambda p: int(p.stem))
            state = torch.load(pts[-1], map_location=args.device, weights_only=True)
            model = build(cfg["model_config"], vocab, args.device, state=state)
            is_lstm = fam == "LSTM"
            L, H = model.rnn.num_layers, model.rnn.hidden_size

            toks7, bel7, wt7, p07 = enum_prefixes(T, rev, b0, P)
            branches, bmats = branch_mats(T, ROLL)
            A_t = states_last(model, toks7, vocab, args.device, is_lstm)
            A = A_t.cpu().numpy()
            Xp, Yp, Wp, beta = pooled_probe(model, T, rev, b0, vocab, args.device, is_lstm)
            G = wridge(Yp, Xp, Wp, 1e-6)[1:]
            Wout = model.output_layer.weight.detach().float().cpu().numpy()
            u = np.zeros(A.shape[1])
            u[(L - 1) * H:L * H] = Wout[0] - Wout[1]
            u /= np.linalg.norm(u)

            ii, jj = np.triu_indices(len(A), k=1)
            dp0 = np.abs(p07[ii] - p07[jj])
            anti = dp0 >= np.quantile(dp0, 0.75)
            src = np.concatenate([ii[anti], jj[anti]])
            dst = np.concatenate([jj[anti], ii[anti]])

            d_full = A[dst] - A[src]
            d_neut = d_full - np.outer(d_full @ u, u)
            d_ker = ((bel7[dst] - bel7[src]) @ Pker) @ G
            eta_chi = bel7[src] + (bel7[dst] - bel7[src]) @ Pker
            th_chi = theory_tree(T, rev, eta_chi, bmats)
            th_w = theory_tree(T, rev, bel7[src], bmats)
            th_hyb = hybrid_tree(T, rev, bel7[src], bel7[dst], branches)
            gap_targets = np.median(tree_kl(th_chi, th_hyb))

            base = rollout(model, A_t, branches, L, H, is_lstm, args.device)
            Qu = np.prod(base, 2)[src]
            for kind, delta in (("ntp_neutral", d_neut), ("ker_write", d_ker)):
                sp = rollout(model, A[src] + delta, branches, L, H, is_lstm, args.device)
                Qp = np.prod(sp, 2)
                r = {"family": fam, "seed": seed, "kind": kind,
                     "n_pairs": len(src), "kl_chi_vs_hyb_targets": float(gap_targets)}
                for name, tht in (("chi", th_chi), ("hyb", th_hyb)):
                    r[f"pct_{name}"] = float(100 * (1 - np.median(tree_kl(Qp, tht))
                                                    / max(np.median(tree_kl(Qu, tht)), EPS)))
                    r[f"klmed_{name}"] = float(np.median(tree_kl(Qp, tht)))
                    obs = np.log(np.clip(Qp, EPS, None)) - np.log(np.clip(Qu, EPS, None))
                    pred = (np.log(np.clip(tht, EPS, None))
                            - np.log(np.clip(th_w, EPS, None)))
                    r[f"r2_{name}"] = float(np.corrcoef(obs.ravel(), pred.ravel())[0, 1] ** 2)
                rows.append(r)
                print(f"[{fam} s{seed} {kind:12s} anti-matched] "
                      f"KL->chi {r['klmed_chi']:.4f} vs KL->hyb {r['klmed_hyb']:.4f} "
                      f"(targets differ by {gap_targets:.4f}) "
                      f"pct chi/hyb {r['pct_chi']:.1f}/{r['pct_hyb']:.1f} "
                      f"R2 {r['r2_chi']:.3f}/{r['r2_hyb']:.3f}", flush=True)
            del model
            torch.cuda.empty_cache()

    f = OUT_DIR / "verify_target_adjudication.csv"
    keys = list(rows[0].keys())
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"saved {f}")


if __name__ == "__main__":
    main()
