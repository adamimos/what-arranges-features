"""
Compute the full layer-curves suite (acts→bel, acts→shuf, ntp→bel,
random→bel, etc.) for default-init LSTMs at the qRRXOR hero. Mirrors the
transformer's run_shuffle_analysis output structure so figures can render
the same panels.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from epsilon_transformers.training.networks import RNNWrapper
from epsilon_transformers.analysis.process_targets import prepare_process_data
from epsilon_transformers.analysis.checkpoints import list_checkpoints, _maybe_pca
from epsilon_transformers.analysis.shuffle_control import (
    _paper_regression_r2, geometric_shuffle, local_geometric_shuffle,
)
from epsilon_transformers.analysis.lstm_checkpoints import (
    load_run_config, load_lstm, extract_lstm_activations,
)


LAYER_KEYS = ["embed", "layer_0_h_n", "layer_1_h_n", "layer_2_h_n",
              "layer_3_h_n", "layer_3_c_n", "logits"]


def make_random_lstm(cfg, device, seed=999):
    mc = cfg["model_config"]
    rnn_cls = getattr(nn, mc.get("rnn_type", "LSTM").upper())
    torch.manual_seed(seed)
    rnn = rnn_cls(
        input_size=2,
        hidden_size=mc["hidden_size"],
        num_layers=mc["num_layers"],
        dropout=mc.get("dropout", 0.0),
        batch_first=True,
        bidirectional=mc.get("bidirectional", False),
    ).to(device)
    out = nn.Linear(mc["hidden_size"], 2).to(device)
    return RNNWrapper(rnn, out).to(device)


def compute_curves_for_run(run_dir: Path, device: str = "cpu",
                            n_shuffles: int = 20, pca_rank: float = 0.95):
    cfg = load_run_config(str(run_dir))
    ckpts = list_checkpoints(str(run_dir))
    if not ckpts:
        return None

    proc = prepare_process_data(cfg)
    inputs = proc["seqs"][:, :-1].long()
    beliefs = proc["beliefs"]
    weights = proc["weights"]
    P_E, P_E_perp = proc["P_E"], proc["P_E_perp"]
    ntp = proc["ntp"]  # (N, 1) probability of token-0 at last position
    shuffled = proc["shuffled"]
    local_shuffled = proc["local_shuffled"]

    # Trained model
    model_t, _ = load_lstm(str(run_dir), ckpts[-1], device)
    acts_t = extract_lstm_activations(model_t, inputs, vocab_size=2,
                                       device=device)
    del model_t

    # Random model (untrained)
    model_r = make_random_lstm(cfg, device, seed=999)
    model_r.eval()
    acts_r = extract_lstm_activations(model_r, inputs, vocab_size=2,
                                       device=device)
    del model_r

    rng = np.random.default_rng(0)

    def _r2(X, Y):
        return _paper_regression_r2(X, Y, weights, device=device)

    curves = {}
    for lk in LAYER_KEYS:
        if lk not in acts_t:
            continue
        a_pca, _, _ = _maybe_pca(acts_t[lk], pca_rank)
        ar_pca, _, _ = _maybe_pca(acts_r[lk], pca_rank)
        curve = {
            "acts_to_beliefs": _r2(a_pca, beliefs),
            "acts_to_shuffled": _r2(a_pca, shuffled),
            "acts_to_local_shuffled": _r2(a_pca, local_shuffled),
            "ntp_to_beliefs": _r2(ntp, beliefs),
            "beliefs_to_shuffled": _r2(beliefs, shuffled),
            "random_to_beliefs": _r2(ar_pca, beliefs),
            "random_to_shuffled": _r2(ar_pca, shuffled),
            "d_pca": int(a_pca.shape[1]),
        }
        curves[lk] = curve

    # Concat (h_only)
    h_layers = [acts_t[f"layer_{i}_h_n"] for i in range(4)
                if f"layer_{i}_h_n" in acts_t]
    rh_layers = [acts_r[f"layer_{i}_h_n"] for i in range(4)
                 if f"layer_{i}_h_n" in acts_r]
    if h_layers:
        Xt = np.concatenate(h_layers, axis=1)
        Xr = np.concatenate(rh_layers, axis=1)
        Xt_pca, _, _ = _maybe_pca(Xt, pca_rank)
        Xr_pca, _, _ = _maybe_pca(Xr, pca_rank)
        curves["CONCAT_h"] = {
            "acts_to_beliefs": _r2(Xt_pca, beliefs),
            "acts_to_shuffled": _r2(Xt_pca, shuffled),
            "acts_to_local_shuffled": _r2(Xt_pca, local_shuffled),
            "ntp_to_beliefs": _r2(ntp, beliefs),
            "beliefs_to_shuffled": _r2(beliefs, shuffled),
            "random_to_beliefs": _r2(Xr_pca, beliefs),
            "random_to_shuffled": _r2(Xr_pca, shuffled),
            "d_pca": int(Xt_pca.shape[1]),
        }

    return {
        "init_scheme": cfg["model_config"].get("init_scheme", "default"),
        "seed": cfg["model_config"].get("seed", 42),
        "tokens_seen": int(ckpts[-1].replace(".pt", "")),
        "curves": curves,
        "process_config": cfg["process_config"],
        "model_config": cfg["model_config"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", choices=["default", "rich"], default="default")
    p.add_argument("--sweep_dirs", nargs="+", default=None,
                   help="explicit list of sweep dirs to scan (overrides --init)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.sweep_dirs:
        sweep_dirs = args.sweep_dirs
    else:
        sweep_dirs = {
            "default": ["results/20260503215800_lstm_default_cpu",
                        "results/20260503220112_lstm_default_cpu_more"],
            "rich":    ["results/20260503215800_lstm_rich_cpu",
                        "results/20260503220112_lstm_rich_cpu_more"],
        }[args.init]

    results = []
    for sd in sweep_dirs:
        sdp = Path(sd)
        if not sdp.exists():
            continue
        for rd in sorted(sdp.iterdir()):
            if not (rd.is_dir() and rd.name.startswith("run_")):
                continue
            print(f"  {sdp.name}/{rd.name}")
            r = compute_curves_for_run(rd, device=args.device)
            if r is not None:
                results.append(r)

    out = args.out or f"figures/beyond_ntp/lstm_curves_{args.init}.joblib"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, out)
    print(f"Saved {len(results)} runs to {out}")


if __name__ == "__main__":
    main()
