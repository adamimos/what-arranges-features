"""LSTM checkpoints: load an LSTM run and extract per-layer hidden and cell states at the last position."""

from pathlib import Path

import torch
import torch.nn as nn
import yaml

from epsilon_transformers.training.networks import RNNWrapper


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
