"""Scoring of the qRRXOR hyperparameter screen and report runs: activation extraction, belief /
Markov targets, weighted CV R^2 per run, and t-intervals over seeds."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from epsilon_transformers.analysis.checkpoints import (
    load_run_config, load_model, list_checkpoints,
)
from epsilon_transformers.training.networks import RNNWrapper
from epsilon_transformers.analysis.process_targets import enumerate_process_data
from epsilon_transformers.analysis.weighted_probe import _weighted_r2
from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.analysis.multiseed_probe import final_val_lb

N_FOLDS = 5
RCOND = 1e-10        # primary probe rcond (as used on Bloch)
RCOND_STAB = 1e-3    # stabilized probe rcond for ill-conditioned high-LR checkpoints
CONV_THRESH = 1.005
SHUFFLE_SEED = 0


def load_rnn(run_dir, ckpt, device, vocab):
    cfg = load_run_config(str(run_dir))
    mc = cfg["model_config"]
    cls = getattr(nn, mc["rnn_type"].upper())
    rnn = cls(input_size=vocab, hidden_size=mc["hidden_size"],
              num_layers=mc["num_layers"], dropout=mc.get("dropout", 0.0),
              batch_first=True, bidirectional=mc.get("bidirectional", False)).to(device)
    out = nn.Linear(mc["hidden_size"], vocab).to(device)
    model = RNNWrapper(rnn, out).to(device)
    state = torch.load(f"{run_dir}/{ckpt}", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def weighted_corr(a, b, w):
    wn = w / w.sum()
    ma = (a * wn).sum(); mb = (b * wn).sum()
    cov = (wn * (a - ma) * (b - mb)).sum()
    va = np.sqrt((wn * (a - ma) ** 2).sum()); vb = np.sqrt((wn * (b - mb) ** 2).sum())
    return float(cov / max(va * vb, 1e-12))


def extract_transformer(run_dir, ckpt, contexts, device):
    m = load_model(str(run_dir), ckpt, device)
    names = [f"blocks.{i}.hook_resid_pre" for i in range(m.cfg.n_layers)]
    inp = torch.from_numpy(contexts).long().to(device)
    with torch.no_grad():
        logits, cache = m.run_with_cache(inp, names_filter=names)
    acts = np.concatenate([cache[k][:, -1, :].cpu().numpy() for k in names], axis=1)
    p0 = torch.softmax(logits[:, -1, :], dim=-1)[:, 0].cpu().numpy()
    del m
    if "cuda" in str(device):
        torch.cuda.empty_cache()
    return acts, p0


def extract_rnn(model, contexts, vocab, device):
    x = torch.from_numpy(contexts).long().to(device)
    oh = torch.zeros(*x.shape, vocab, device=device)
    oh.scatter_(2, x.unsqueeze(-1), 1.0)
    with torch.no_grad():
        output, hidden = model.rnn(oh)
        h_n = hidden[0] if isinstance(hidden, tuple) else hidden
        logits = model.output_layer(output[:, -1, :])
    acts = torch.cat([h_n[i] for i in range(h_n.shape[0])], dim=1).cpu().numpy()
    p0 = torch.softmax(logits, dim=-1)[:, 0].cpu().numpy()
    return acts, p0


def build_targets(cfg):
    proc = enumerate_process_data(cfg)
    n_ctx = cfg["model_config"]["n_ctx"]
    contexts = proc["seqs"].numpy()[:, :n_ctx]
    T = get_matrix_from_args(**cfg["process_config"])
    vocab = int(T.shape[0]); assert vocab == 2, f"expected vocab 2, got {vocab}"
    w = proc["weights"]; bel = proc["beliefs"]
    # ntp_proj: belief projected onto col(E) (NTP-relevant subspace)
    wn = w / w.sum()
    b_mean = (bel * wn[:, None]).sum(axis=0)
    ntp_proj = b_mean + (bel - b_mean) @ proc["P_E"]
    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(len(w))
    return {
        "contexts": contexts, "n_ctx": n_ctx, "vocab": vocab,
        "belief": bel, "shuffled": proc["shuffled"], "ntp_proj": ntp_proj,
        "ntp_opt": proc["ntp"][:, 0], "weights": w, "perm": perm,
        "belief_dim": int(bel.shape[1]), "N": int(len(w)),
    }


def r2(acts, tgt, proc, rcond):
    p = proc["perm"]; w = proc["weights"]
    return _weighted_r2(acts[p], tgt[p], w[p], n_folds=N_FOLDS, rcond=rcond)


def analyze_run(run_dir: Path, config_label, arch, proc, device):
    cfg = load_run_config(str(run_dir))
    seed = cfg["model_config"].get("seed", "?")
    val_lb = final_val_lb(run_dir)
    ckpt = list_checkpoints(str(run_dir))[-1]
    if arch == "transformer":
        acts, p0 = extract_transformer(run_dir, ckpt, proc["contexts"], device)
    else:
        model = load_rnn(run_dir, ckpt, device, proc["vocab"])
        acts, p0 = extract_rnn(model, proc["contexts"], proc["vocab"], device)
        del model
    bel_stab = r2(acts, proc["belief"], proc, RCOND_STAB)
    bel_raw = r2(acts, proc["belief"], proc, RCOND)
    shuf_stab = r2(acts, proc["shuffled"], proc, RCOND_STAB)
    ntpp_stab = r2(acts, proc["ntp_proj"], proc, RCOND_STAB)
    return {
        "arch": arch, "config": config_label, "seed": seed,
        "run_dir": run_dir.name, "val_loss_lb": val_lb,
        "converged": bool(val_lb < CONV_THRESH),
        "tokens_seen": int(ckpt.replace(".pt", "")),
        "act_absmax": float(np.abs(acts).max()),
        "belief_r2": float(bel_stab), "belief_r2_raw": float(bel_raw),
        "shuffled_r2": float(shuf_stab), "ntp_proj_r2": float(ntpp_stab),
        "gap": float(bel_stab - shuf_stab),
        "ntp_corr": float(weighted_corr(p0, proc["ntp_opt"], proc["weights"])),
    }


def t_interval(vals, alpha=0.05):
    from scipy import stats
    v = np.asarray(vals, float); n = len(v); m = float(v.mean())
    if n < 2:
        return m, float("nan")
    sd = float(v.std(ddof=1))
    return m, float(stats.t.ppf(1 - alpha / 2, df=n - 1) * sd / np.sqrt(n))


def _msd(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))
