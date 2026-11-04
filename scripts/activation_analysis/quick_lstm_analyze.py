"""
Quick LSTM activation→belief regression. Sanity check on partially-trained
checkpoints to confirm the analysis pipeline works for LSTMs and to peek at
where the numbers are going.

For each layer L in [0, num_layers-1], extracts h_n[L] at last time step
(= LSTM hidden state at last position, after layer L), pools across the
n_samples sequences, applies PCA-95% on activations, then runs the paper's
rcond-CV regression to compute R²(acts → beliefs).

Usage:
    PYTHONPATH=. uv run python scripts/activation_analysis/quick_lstm_analyze.py \
        --run_dir results/<sweep>/<run> [--checkpoint last]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from epsilon_transformers.training.networks import RNNWrapper
from scripts.activation_analysis.run_shuffle_analysis import (
    prepare_process_data,
    list_checkpoints,
    _maybe_pca,
)
from scripts.activation_analysis.shuffle_control import _paper_regression_r2


def load_run_config(run_dir: str) -> dict:
    with open(Path(run_dir) / "run_config.yaml") as f:
        return yaml.safe_load(f)


def load_lstm(run_dir: str, ckpt_name: str, device: str):
    cfg = load_run_config(run_dir)
    mc = cfg["model_config"]
    rnn_cls = getattr(nn, mc.get("rnn_type", "LSTM").upper())
    vocab_size = 2  # qRRXOR is binary
    rnn = rnn_cls(
        input_size=vocab_size,
        hidden_size=mc["hidden_size"],
        num_layers=mc["num_layers"],
        dropout=mc.get("dropout", 0.0),
        batch_first=True,
        bidirectional=mc.get("bidirectional", False),
    ).to(device)
    out = nn.Linear(mc["hidden_size"], vocab_size).to(device)
    model = RNNWrapper(rnn, out).to(device)
    state = torch.load(Path(run_dir) / ckpt_name, map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def extract_lstm_activations(model, inputs, vocab_size, device):
    """Returns dict[layer_name] -> (N, hidden_size) at the last time step.

    LSTM layers don't have separate pre/post like transformer; we expose one
    representation per layer (h_n[L] = post-layer-L hidden state at last t).

    Plus an "embedding" pseudo-layer (the one-hot input itself, for parity
    with the transformer's L0_pre baseline).
    """
    # One-hot encode inputs: (N, T) int -> (N, T, vocab_size)
    inputs = inputs.to(device)
    one_hot = torch.zeros(*inputs.shape, vocab_size, device=device)
    one_hot.scatter_(2, inputs.unsqueeze(-1), 1.0)

    with torch.no_grad():
        # nn.LSTM with default num_layers returns (output, (h_n, c_n))
        # output: (N, T, H), h_n: (num_layers, N, H), c_n: (num_layers, N, H)
        output, (h_n, c_n) = model.rnn(one_hot)
        out_logits = model.output_layer(output[:, -1, :])

    n_layers = h_n.shape[0]
    acts = {}
    # Embedding-equivalent: one-hot at last position (parity with L0_pre)
    acts["embed"] = one_hot[:, -1, :].cpu().numpy()
    for i in range(n_layers):
        acts[f"layer_{i}_h_n"] = h_n[i].cpu().numpy()
    # Cell state at top layer for completeness
    acts[f"layer_{n_layers-1}_c_n"] = c_n[-1].cpu().numpy()
    acts["logits"] = out_logits.cpu().numpy()
    return acts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", default="last",
                   help="'last' or specific tokens-seen number")
    p.add_argument("--device", default="cpu")
    p.add_argument("--pca_rank", type=float, default=0.95)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpts = list_checkpoints(str(run_dir))
    if not ckpts:
        print(f"  no checkpoints in {run_dir}")
        return
    ckpt_name = ckpts[-1] if args.checkpoint == "last" else f"{args.checkpoint}.pt"

    print(f"=== {run_dir.name}  ckpt={ckpt_name} ===")
    model, cfg = load_lstm(str(run_dir), ckpt_name, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  arch: LSTM L={cfg['model_config']['num_layers']} "
          f"H={cfg['model_config']['hidden_size']} "
          f"({n_params:,} params)")
    print(f"  init_scheme: {cfg['model_config'].get('init_scheme', 'default')}")
    print(f"  process: {cfg['process_config']}")

    proc_data = prepare_process_data(cfg)
    inputs = proc_data["seqs"][:, :-1].long()
    beliefs = proc_data["beliefs"]
    weights = proc_data["weights"]

    acts = extract_lstm_activations(model, inputs, vocab_size=2,
                                     device=args.device)

    print(f"\n  N={beliefs.shape[0]}  d_belief={beliefs.shape[1]}  "
          f"tokens_seen={ckpt_name.replace('.pt','')}")
    print(f"  {'layer':<28s} {'d_acts':>7s}  {'PCA(d)':>7s}  {'R²(acts→bel)':>13s}")
    print("  " + "-" * 66)
    for lk in ["embed", "layer_0_h_n", "layer_1_h_n", "layer_2_h_n",
               "layer_3_h_n", "layer_3_c_n", "logits"]:
        if lk not in acts:
            continue
        a_raw = acts[lk]
        a_pca, evr, n = _maybe_pca(a_raw, args.pca_rank)
        d_pca = a_pca.shape[1]
        r2 = _paper_regression_r2(a_pca, beliefs, weights, device=args.device)
        print(f"  {lk:<28s} {a_raw.shape[1]:>7d}  {d_pca:>7d}  {r2:>13.3f}")

    # Concat variants
    print("  " + "-" * 66)
    h_layers = [acts[f"layer_{i}_h_n"] for i in range(4)
                if f"layer_{i}_h_n" in acts]
    if h_layers:
        a_concat_h = np.concatenate(h_layers, axis=1)
        a_pca, _, _ = _maybe_pca(a_concat_h, args.pca_rank)
        r2 = _paper_regression_r2(a_pca, beliefs, weights, device=args.device)
        print(f"  {'CONCAT all h_n (4 layers)':<28s} {a_concat_h.shape[1]:>7d}  "
              f"{a_pca.shape[1]:>7d}  {r2:>13.3f}")
    if h_layers and "layer_3_c_n" in acts:
        a_concat_hc = np.concatenate(h_layers + [acts["layer_3_c_n"]], axis=1)
        a_pca, _, _ = _maybe_pca(a_concat_hc, args.pca_rank)
        r2 = _paper_regression_r2(a_pca, beliefs, weights, device=args.device)
        print(f"  {'CONCAT h_n + c_n':<28s} {a_concat_hc.shape[1]:>7d}  "
              f"{a_pca.shape[1]:>7d}  {r2:>13.3f}")


if __name__ == "__main__":
    main()
