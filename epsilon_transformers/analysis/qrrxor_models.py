"""qRRXOR model helpers: build a recurrent net from a run config, concatenated last-position
activations for transformers and recurrent nets, and the convergence check."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from epsilon_transformers.training.networks import RNNWrapper


def final_val_lb(run_dir: Path) -> float:
    f = run_dir / "loss.csv"
    if not f.exists():
        return float("inf")
    import csv as _csv
    with open(f) as fh:
        rows = list(_csv.DictReader(fh))
    return float(rows[-1]["val_loss_mean"]) if rows else float("inf")


def transformer_concat(model, contexts, device, bs=8192):
    names = [f"blocks.{i}.hook_resid_pre" for i in range(model.cfg.n_layers)]
    inp = torch.from_numpy(contexts).long()
    chunks = {k: [] for k in names}
    with torch.no_grad():
        for s in range(0, inp.shape[0], bs):
            _, cache = model.run_with_cache(inp[s:s + bs].to(device), names_filter=names)
            for k in names:
                chunks[k].append(cache[k][:, -1, :].cpu().numpy())
    return np.concatenate([np.concatenate(chunks[k], 0) for k in names], axis=1)


def build_rnn(cfg, vocab, device, state=None, seed=None):
    mc = cfg["model_config"]
    if seed is not None:
        torch.manual_seed(seed)
    cls = getattr(nn, mc["rnn_type"].upper())
    rnn = cls(input_size=vocab, hidden_size=mc["hidden_size"], num_layers=mc["num_layers"],
              dropout=mc.get("dropout", 0.0), batch_first=True,
              bidirectional=mc.get("bidirectional", False)).to(device)
    out = nn.Linear(mc["hidden_size"], vocab).to(device)
    model = RNNWrapper(rnn, out).to(device)
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return model


def rnn_concat(model, contexts, vocab, device):
    x = torch.from_numpy(contexts).long().to(device)
    oh = torch.zeros(*x.shape, vocab, device=device)
    oh.scatter_(2, x.unsqueeze(-1), 1.0)
    with torch.no_grad():
        _, hidden = model.rnn(oh)
        h_n = hidden[0] if isinstance(hidden, tuple) else hidden
    return torch.cat([h_n[i] for i in range(h_n.shape[0])], dim=1).cpu().numpy()
