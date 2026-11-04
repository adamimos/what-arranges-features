"""
Minimal muP (Maximal Update Parameterization) for HookedTransformer.

Yang et al., "Tensor Programs V": stable feature-learning dynamics across widths
via three coordinated changes — (1) per-parameter init scaling with width,
(2) attention logit scale 1/d_head (not 1/sqrt(d_head)), (3) per-parameter
learning-rate multipliers under Adam/SGD.

Gated by YAML `mup_config.enabled=true`; when false or absent, callers must
skip both `apply_mup` and `mup_param_groups` so behaviour matches the
pre-existing TransformerLens init + flat-LR path.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer


HIDDEN_SUFFIXES = ("W_Q", "W_K", "W_V", "W_O", "W_in", "W_out")
INPUT_SUFFIXES = ("embed.W_E", "pos_embed.W_pos")
HIDDEN_BIAS_SUFFIXES = (".b_Q", ".b_K", ".b_V", ".b_O", ".b_in", ".b_out")


def _is_input_weight(name: str) -> bool:
    return any(name.endswith(s) for s in INPUT_SUFFIXES)


def _is_hidden_weight(name: str) -> bool:
    return any(name.endswith(s) for s in HIDDEN_SUFFIXES)


def _is_readout(name: str) -> bool:
    return name.startswith("unembed.")


def _is_layernorm(name: str) -> bool:
    return ".ln" in name or name.startswith("ln_final")


def apply_mup(model: HookedTransformer, mup_cfg: dict) -> float:
    """Reinitialize weights and patch attn_scale in-place per muP.

    Parameters
    ----------
    model : HookedTransformer already constructed (its prior init is discarded).
    mup_cfg : dict with keys
        enabled (bool):            must be True (caller should gate).
        base_width (int):          d_model at which HPs are tuned. width_mult = d_model / base_width.
        init_range (float = 1.0):  hidden-weight std at base width; scaled by 1/sqrt(width_mult).
        input_init_std (float=1.0): std for embed.W_E and pos_embed.W_pos (no width scaling).
        zero_init_readout (bool=True): canonical Yang et al. readout init.
        readout_init_std (float=1.0):  only used if zero_init_readout=false; scaled by 1/width_mult.
        attn_scale_mode ('mup'|'standard' = 'mup'):
            'mup'      -> attn_scores / d_head      (muP)
            'standard' -> attn_scores / sqrt(d_head) (TL default)
        seed (int|None):           if set, torch.manual_seed(seed) before re-init.

    Returns
    -------
    width_mult (float): stored on model._mup_width_mult for optimizer setup.
    """
    assert mup_cfg.get("enabled", False), "apply_mup called with enabled=false"

    base_width = int(mup_cfg["base_width"])
    width_mult = model.cfg.d_model / base_width
    init_range = float(mup_cfg.get("init_range", 1.0))
    input_std = float(mup_cfg.get("input_init_std", 1.0))
    zero_readout = bool(mup_cfg.get("zero_init_readout", True))
    readout_std = float(mup_cfg.get("readout_init_std", 1.0))
    attn_mode = mup_cfg.get("attn_scale_mode", "mup")
    seed = mup_cfg.get("seed", None)

    if seed is not None:
        torch.manual_seed(int(seed))

    hidden_std = init_range / math.sqrt(width_mult)

    with torch.no_grad():
        for name, p in model.named_parameters():
            if _is_input_weight(name):
                nn.init.normal_(p, mean=0.0, std=input_std)
            elif _is_readout(name):
                if name.endswith("W_U"):
                    if zero_readout:
                        nn.init.zeros_(p)
                    else:
                        nn.init.normal_(p, mean=0.0, std=readout_std / width_mult)
                else:  # b_U
                    nn.init.zeros_(p)
            elif _is_hidden_weight(name):
                nn.init.normal_(p, mean=0.0, std=hidden_std)
            elif any(name.endswith(s) for s in HIDDEN_BIAS_SUFFIXES):
                nn.init.zeros_(p)
            elif _is_layernorm(name):
                if name.endswith(".w"):
                    nn.init.ones_(p)
                elif name.endswith(".b"):
                    nn.init.zeros_(p)
            # Other params (if any) left untouched.

    # Attention scale. TL copies cfg.attn_scale into each block.attn at
    # construction, so patching the blocks directly is required.
    d_head = model.cfg.d_head
    if attn_mode == "mup":
        new_scale = float(d_head)
    elif attn_mode == "standard":
        new_scale = float(math.sqrt(d_head))
    else:
        raise ValueError(f"unknown attn_scale_mode: {attn_mode}")
    model.cfg.attn_scale = new_scale
    for block in model.blocks:
        if hasattr(block.attn, "attn_scale"):
            block.attn.attn_scale = new_scale

    model._mup_width_mult = float(width_mult)
    model._mup_cfg = dict(mup_cfg)
    return float(width_mult)


def mup_param_groups(
    model: HookedTransformer,
    lr: float,
    width_mult: float,
    opt_type: str = "adam",
):
    """Split params into (input, hidden, readout) with muP LR multipliers.

    Adam: input LR = lr, hidden LR = lr / width_mult, readout LR = lr / width_mult.
    SGD : input LR = lr, hidden LR = lr * width_mult, readout LR = lr * width_mult.

    Group 0 is always the input group so existing code that reads
    `optimizer.param_groups[0]['lr']` keeps reporting the base LR.
    """
    input_ps, hidden_ps, readout_ps = [], [], []
    for name, p in model.named_parameters():
        if _is_readout(name):
            readout_ps.append(p)
        elif _is_hidden_weight(name):
            hidden_ps.append(p)
        else:
            input_ps.append(p)

    if opt_type == "adam":
        hidden_lr = lr / width_mult
        readout_lr = lr / width_mult
    elif opt_type == "sgd":
        hidden_lr = lr * width_mult
        readout_lr = lr * width_mult
    else:
        raise ValueError(f"unknown opt_type: {opt_type}")

    return [
        {"params": input_ps, "lr": lr, "name": "input"},
        {"params": hidden_ps, "lr": hidden_lr, "name": "hidden"},
        {"params": readout_ps, "lr": readout_lr, "name": "readout"},
    ]
